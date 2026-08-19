"""ONNX 导出包装类

把 SAM3 子模块的 forward 接口转成 ONNX 友好的形式:
- NestedTensor 解包为普通 tensor (ONNX 不支持自定义容器)
- dict / list 输出展平为 tuple (ONNX 只支持 tensor 序列输出)
- Python 控制流开关 (multimask_output / num_obj_ptr_tokens / mask 有无)
  固化为构造参数, 让 trace 时分支静态确定

所有 wrapper 只持有子模块引用, 不复制权重; sam3 的 import 全部在
core/export/utils.py 的函数内部完成 (需先 setup_sam3_path())。
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class ImageEncoderWrapper(nn.Module):
    """A: Sam3TriViTDetNeck (ViT + 3头×3级 FPN)。

    输入:
        image: [1, 3, H, W] float32 (H=W=resolution, 336 的倍数)

    输出 (18 个 tensor, 按 head → level 交错 (feat, pos) 排列):
        {sam3,interactive,propagation}_{fpn,pos}_{0,1,2}
        fpn_i: [1, 256, S_i, S_i], S = (4/14, 2/14, 1/14) × H
        pos_i: 同尺寸的 sin/cos 位置编码

    neck 输入直接给普通 tensor (ViT.forward 对非 NestedTensor 输入走
    mask=None 分支); 输出的 NestedTensor 解包成 .tensors。
    """

    def __init__(self, neck: nn.Module):
        super().__init__()
        self.neck = neck

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        (
            sam3_feats,
            sam3_pos,
            interactive_feats,
            interactive_pos,
            propagation_feats,
            propagation_pos,
        ) = self.neck(
            image,
            need_sam3_out=True,
            need_interactive_out=True,
            need_propagation_out=True,
        )
        out = []
        for feats, poss in (
            (sam3_feats, sam3_pos),
            (interactive_feats, interactive_pos),
            (propagation_feats, propagation_pos),
        ):
            for feat, pos in zip(feats, poss):
                # NestedTensor 解包 (mask 对方形输入恒为 None, 丢弃)
                out.append(feat.tensors if hasattr(feat, "tensors") else feat)
                out.append(pos)
        return tuple(out)


class TextEncoderWrapper(nn.Module):
    """B: VETextEncoder (TextTransformer + resizer), 去掉 tokenizer。

    复刻 VETextEncoder.forward 的字符串分支, 但输入直接是 BPE token ids
    (tokenization 在端侧用 Python/JS 实现, 不进 ONNX 图)。

    输入:
        token_ids: [B, 32] int64 (BPE token ids, 0 = padding)

    输出:
        text_memory: [32, B, 256] (resizer 后, seq-first, 供 detector transformer)
        text_mask:   [B, 32] bool (True = padding, PyTorch attention 约定)
        text_embeds: [32, B, 1024] (resizer 前的 token embedding, seq-first)
    """

    def __init__(self, text_encoder: nn.Module):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(
        self, token_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoder = self.text_encoder.encoder
        # 与 VETextEncoder.forward 相同的计算, 只是跳过 tokenizer
        inputs_embeds = encoder.token_embedding(token_ids)  # [B, 32, 1024]
        _, text_memory = encoder(token_ids)  # [B, 32, 1024] (pool_type=none → tokens)
        # True = padding (VETextEncoder 里先 (ids != 0) 再 .ne(1) 取反)
        text_attention_mask = (token_ids != 0).ne(1)
        # transpose 成 seq-first 再过 resizer (与原实现顺序一致; Linear 作用在
        # 最后一维, 顺序不影响数值)
        text_memory = self.text_encoder.resizer(text_memory.transpose(0, 1))
        return text_memory, text_attention_mask, inputs_embeds.transpose(0, 1)


class PromptEncoderWrapper(nn.Module):
    """C: SAM 风格 PromptEncoder (multiplex 模型的 interactive_sam_prompt_encoder)。

    固定 boxes=None (点提示总是 pad 一个 dummy 点, 与原调用点一致)。

    输入:
        point_coords: [B, P, 2] float32 (绝对像素坐标, xy)
        point_labels: [B, P] int64 (1=前景点, 0=背景点, -1=padding)
        mask_input:   [B, 1, 4F, 4F] float32 (仅 with_mask=True; F=resolution//14)

    输出:
        sparse_embeddings: [B, P+1, 256]
        dense_embeddings:  [B, 256, F, F] (无 mask 时为 no_mask_embed 广播)
    """

    def __init__(self, prompt_encoder: nn.Module, with_mask: bool = False):
        super().__init__()
        self.prompt_encoder = prompt_encoder
        self.with_mask = with_mask

    def forward(
        self,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        mask_input: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=(point_coords, point_labels),
            boxes=None,
            masks=mask_input if self.with_mask else None,
        )
        return sparse_embeddings, dense_embeddings


class MemoryAttentionWrapper(nn.Module):
    """D: TransformerEncoderDecoupledCrossAttention (multiplex 记忆注意力)。

    输入均为 seq-first [S, B, C] (batch_first=True 的模块内部自行转置):
        image:            [HW, B, 256] 当前帧最深层特征
        src:              [HW, B, 256] 当前帧特征 (self-attention 输入)
        memory_image:     [M, B, 256] 记忆帧图像特征
        memory:           [M+P, B, 256] 记忆 (含 P 个 obj ptr token)
        image_pos:        [HW, B, 256]
        src_pos:          [HW, B, 256]
        memory_image_pos: [M, B, 256]
        memory_pos:       [M+P, B, 256]

    输出:
        fused_memory: [HW, B, 256] 融合记忆后的当前帧特征 (use_image_in_output=False)
        pos_embed:    [HW, B, 256] (透传的 src_pos)

    num_obj_ptr_tokens (P) 固化为构造参数 (单个条件帧 = multiplex_count 个),
    使 memory/memory_image 长度差的 padding 分支在 trace 时静态确定。
    """

    def __init__(self, encoder: nn.Module, num_obj_ptr_tokens: int):
        super().__init__()
        self.encoder = encoder
        self.num_obj_ptr_tokens = num_obj_ptr_tokens

    def forward(
        self,
        image: torch.Tensor,
        src: torch.Tensor,
        memory_image: torch.Tensor,
        memory: torch.Tensor,
        image_pos: torch.Tensor,
        src_pos: torch.Tensor,
        memory_image_pos: torch.Tensor,
        memory_pos: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.encoder(
            image=image,
            src=src,
            memory_image=memory_image,
            memory=memory,
            image_pos=image_pos,
            src_pos=src_pos,
            memory_image_pos=memory_image_pos,
            memory_pos=memory_pos,
            num_obj_ptr_tokens=self.num_obj_ptr_tokens,
        )
        return out["memory"], out["pos_embed"]


class MaskDecoderWrapper(nn.Module):
    """E: MultiplexMaskDecoder (传播路径, 无点/框提示输入 — 对象条件由
    extra_per_object_embeddings 携带)。

    multimask_output 固化为 True (multiplex 配置 multimask_outputs_only=True
    时必须为 True), 同时避开 dynamic_multimask_via_stability 的数据依赖分支。

    输入:
        image_embeddings:            [B, 256, F, F] (D 的输出, F=resolution//14)
        image_pe:                    [1, 256, F, F] (tracker.image_pe_layer 的 dense PE)
        feat_s0:                     [B, 32, 4F, 4F] (fpn[0] 过 conv_s0)
        feat_s1:                     [B, 64, 2F, 2F] (fpn[1] 过 conv_s1)
        extra_per_object_embeddings: [B, multiplex_count, 256] (抑制/有效嵌入)

    输出:
        masks:               [B, multiplex_count, 3, 4F, 4F] 低分辨率 mask logits
        iou_pred:            [B, multiplex_count, 3]
        sam_tokens_out:      [B, multiplex_count, 3, 256]
        object_score_logits: [B, multiplex_count, 1]
    """

    def __init__(self, mask_decoder: nn.Module, multimask_output: bool = True):
        super().__init__()
        self.mask_decoder = mask_decoder
        self.multimask_output = multimask_output

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        feat_s0: torch.Tensor,
        feat_s1: torch.Tensor,
        extra_per_object_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            multimask_output=self.multimask_output,
            high_res_features=[feat_s0, feat_s1],
            extra_per_object_embeddings=extra_per_object_embeddings,
        )
        return (
            out["masks"],
            out["iou_pred"],
            out["sam_tokens_out"],
            out["object_score_logits"],
        )


class MemoryEncoderWrapper(nn.Module):
    """F: SimpleMaskEncoder (multiplex maskmem_backbone)。

    输入:
        pix_feat: [B, 256, F, F] 当前帧最深层特征 (F=resolution//14)
        masks:    [B, 2*multiplex_count, 16F, 16F] mux 后的对象 mask 通道 +
                  条件通道 (sigmoid/mux/condition 在运行时胶水代码里完成)

    输出:
        memory_features: [B, 256, F, F]
        memory_pos_enc:  [B, 256, F, F]

    注意: mask 输入直接给 interpol_size (16F) 分辨率, 跳过
    SimpleMaskDownSampler 里的 antialias 双线性插值 (torch.onnx 不支持
    antialias=True); 端侧需自行把 4F 分辨率的 mask 双线性缩放到 16F。
    skip_mask_sigmoid 固化为 True (与 _encode_new_memory 调用点一致, sigmoid
    及 scale/bias 已在喂入前完成)。
    """

    def __init__(self, maskmem_backbone: nn.Module):
        super().__init__()
        self.maskmem_backbone = maskmem_backbone

    def forward(
        self, pix_feat: torch.Tensor, masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.maskmem_backbone(pix_feat, masks, skip_mask_sigmoid=True)
        return out["vision_features"], out["vision_pos_enc"][0]
