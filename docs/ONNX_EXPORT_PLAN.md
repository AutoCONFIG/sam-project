# SAM3 ONNX 导出计划（暂不实现）

> 状态：计划已保存，暂不实现。当前聚焦训练 + 推理功能。
> 创建日期：2026-08-19

## 背景

SAM3 没有任何内置导出功能（无 ONNX、TensorRT、TorchScript 脚本或依赖）。
SAM3 multiplex 视频模型是一个有状态的多组件流水线，不能作为单一 ONNX 图导出。
需要拆分成独立的子模型分别导出，端侧运行时用 Python/C++ 胶水代码组装。

社区参考：
- SAM v1 官方仅导出 decoder（`scripts/export_onnx_model.py`）
- SAM v2 无官方导出
- `vietanhdev/samexporter` 已覆盖到 SAM3（主要面向 image predictor，非 multiplex video 跟踪）
- RKNN Toolkit2 支持 ONNX opset 19，但部分算子有 batchsize=1 限制

## 模型拆分方案

SAM3 multiplex 视频模型拆分为 **6 个可独立导出的子模型**：

| 编号 | 子模型 | 类名 | 文件 | 输入 → 输出 | 难度 |
|------|--------|------|------|-------------|------|
| A | 图像编码器 | `ViT` + `Sam3TriViTDetNeck` | `vitdet.py:759`, `necks.py:133` | `[1,3,H,W]` → 3级FPN特征 `[[1,256,H/4,W/4],[1,256,H/2,W/2],[1,256,H,W]]` × 3头 | 低 |
| B | 文本编码器 | `TextTransformer` + `resizer` | `text_encoder_ve.py:259` | `token_ids[1,32]` → `text_mem[32,1,256]`, `mask[1,32]` | 低 |
| C | 提示编码器 | `PromptEncoder` | `sam/prompt_encoder.py:14` | `points/boxes/masks` → `sparse_emb`, `dense_emb` | 低 |
| D | 记忆注意力 | `TransformerEncoderDecoupledCrossAttention` | `decoder.py:1305` | `src[HW,B,256]`, `memory[M,B,256]`, pos_enc → `out[HW,B,256]` | 中 |
| E | 掩码解码器 | `MultiplexMaskDecoder` | `multiplex_mask_decoder.py:16` | `img_emb[B,256,H,W]`, `image_pe`, `high_res`, `extra_emb` → `masks/ius/tokens` | 中 |
| F | 记忆编码器 | `SimpleMaskEncoder` | `memory.py:167` | `pix_feat[B,256,H,W]`, `masks[B,C,H,W]` → `mem_feat[B,256,H,W]` | 低 |

端侧运行时组合方式（3种场景）：
- **图像分割**（点/框提示）：A → C → E（交互式 MaskDecoder 路径）
- **文本检测+分割**：A → B → 检测器 grounding transformer → E
- **视频跟踪传播**：A(首帧) → D → E → F（逐帧循环，记忆在运行时管理）

## 文件结构（待实现）

```
sam-project/
├── commands/
│   └── export.py                   # 导出命令
├── core/
│   └── export/
│       ├── __init__.py
│       ├── wrappers.py             # ONNX 导出包装类
│       ├── export_encoder.py       # 子模型 A
│       ├── export_text_encoder.py  # 子模型 B
│       ├── export_prompt_encoder.py# 子模型 C
│       ├── export_memory_attn.py   # 子模型 D
│       ├── export_mask_decoder.py  # 子模型 E
│       ├── export_memory_encoder.py# 子模型 F
│       └── utils.py                # 模型加载、opset、动态轴、验证
├── configs/
│   └── export/
│       └── default.yaml            # 导出配置
└── docs/
    └── ONNX_EXPORT_PLAN.md         # 本文件
```

## 关键技术处理

1. **image_size 固定**: ONNX 导出时固定 image_size（如 672），RoPE freqs_cis 作为常量嵌入图内
2. **NestedTensor 解包**: SAM3 大量使用 NestedTensor，导出前用 wrapper 类解包为普通 tensor
3. **dict 输出 → tuple**: ONNX 不支持 dict 输出，wrapper 改为 tuple
4. **tokenizer 分离**: BPE tokenizer 不是 nn.Module，端侧用 Python/JS 实现相同 tokenization
5. **记忆管理在运行时**: 视频 tracking 的记忆 bank 组装有 Python 控制流，不导出。端侧运行时维护记忆状态
6. **opset 17**: 兼容 RKNN Toolkit2（上限 opset 19），GELU 用 erf（RKNN 支持）
7. **动态轴**: 默认关闭（RKNN 偏好静态形状），可选开启用于 x86/ARM ONNX Runtime
8. **验证**: 每个子模型导出后用 onnxruntime 对比 PyTorch 输出（FP32 容差 1e-5）

## RKNN 算子限制要点

- `Tile`/`Slice`/`Softmax` 仅 batchsize=1
- `repeat_interleave` 必须替换为 `tile`/`concat`（且 tile 不能 broadcast）
- 不支持 `Loop`/`Scan`/`TopK`/`NonMaxSuppression`（需 CPU 后处理）
- `Resize` 仅 nearest/bilinear（SAM 的 mask 上采样用 bilinear，OK）
- 支持 `LayerNorm`/`Conv`/`MatMul`/`Gemm`（transformer 基础算子 OK）

## 后续迭代

1. 第一版导出 FP32 ONNX，验证可导出性和数值一致性
2. 量化/剪枝后续迭代（RKNN Toolkit2 可直接对 ONNX 做 INT8 量化）
3. 端侧推理胶水代码（记忆管理、tokenizer、NMS）根据目标平台单独实现
4. 检测器 grounding transformer 导出作为可选扩展
