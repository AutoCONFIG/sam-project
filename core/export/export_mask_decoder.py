"""子模型 E: 掩码解码器 (MultiplexMaskDecoder, 传播路径)

来源: tracker.sam_mask_decoder。multimask_output 固化为 True (multiplex 配置
multimask_outputs_only=True 时也必须为 True), 同时避开
dynamic_multimask_via_stability 的数据依赖分支 (argmax/gather)。

输入里的 image_pe 是 tracker.image_pe_layer 产生的常数 dense PE
(端侧可按分辨率预生成); feat_s0/feat_s1 是 A 的 propagation 头 fpn[0]/fpn[1]
分别过 sam_mask_decoder.conv_s0/conv_s1 后的高分辨率特征 (卷积在运行时胶水
代码或单独的卷积图里完成)。
"""

from typing import Dict, List, Tuple

import torch

from .utils import ExportOptions, export_component, get_tracker_model
from .wrappers import MaskDecoderWrapper

NAME = "mask_decoder"

INPUT_NAMES: List[str] = [
    "image_embeddings",
    "image_pe",
    "feat_s0",
    "feat_s1",
    "extra_per_object_embeddings",
]
OUTPUT_NAMES: List[str] = ["masks", "iou_pred", "sam_tokens_out", "object_score_logits"]


def build_wrapper(predictor, resolution: int) -> MaskDecoderWrapper:
    tracker = get_tracker_model(predictor)
    return MaskDecoderWrapper(tracker.sam_mask_decoder, multimask_output=True)


def dummy_inputs(predictor, resolution: int, device) -> Tuple[torch.Tensor, ...]:
    tracker = get_tracker_model(predictor)
    C = tracker.hidden_dim  # 256
    multiplex_count = tracker.multiplex_count  # 16
    feat = resolution // 14
    return (
        torch.randn(1, C, feat, feat, device=device),  # image_embeddings
        torch.randn(1, C, feat, feat, device=device),  # image_pe
        torch.randn(1, C // 8, 4 * feat, 4 * feat, device=device),  # feat_s0 (conv_s0 后, 32ch)
        torch.randn(1, C // 4, 2 * feat, 2 * feat, device=device),  # feat_s1 (conv_s1 后, 64ch)
        torch.randn(1, multiplex_count, C, device=device),  # extra_per_object_embeddings
    )


def dynamic_axes() -> Dict:
    axes = {name: {0: "batch"} for name in INPUT_NAMES}
    for name in OUTPUT_NAMES:
        axes[name] = {0: "batch"}
    return axes


def export(predictor, options: ExportOptions, output_dir):
    wrapper = build_wrapper(predictor, options.resolution)
    device = next(wrapper.parameters()).device
    inputs = dummy_inputs(predictor, options.resolution, device)
    return export_component(
        NAME, wrapper, inputs, INPUT_NAMES, OUTPUT_NAMES, options, output_dir,
        dynamic_axes=dynamic_axes(),
    )
