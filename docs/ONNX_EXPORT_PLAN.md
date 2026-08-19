# SAM3 ONNX 导出计划

> 状态：**第一版已实现**（2026-08-19，本机无 Python 环境，代码经静态走读，尚未实际运行验证）。
> 待办：在有 GPU + Python 的机器上跑 `python sam.py configs/export/default.yaml` 验证。
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

## 文件结构（第一版已实现）

```
sam-project/
├── commands/
│   └── export.py                   # 导出命令 (argparse --config + CLI 覆盖 YAML)
├── core/
│   └── export/
│       ├── __init__.py             # run_export 编排 + EXPORTERS 注册表
│       ├── wrappers.py             # ONNX 导出包装类 (NestedTensor 解包, dict→tuple)
│       ├── export_encoder.py       # 子模型 A
│       ├── export_text_encoder.py  # 子模型 B
│       ├── export_prompt_encoder.py# 子模型 C
│       ├── export_memory_attn.py   # 子模型 D
│       ├── export_mask_decoder.py  # 子模型 E
│       ├── export_memory_encoder.py# 子模型 F
│       └── utils.py                # 模型加载、opset、动态轴、simplify/fp16、onnxruntime 验证
├── configs/
│   └── export/
│       └── default.yaml            # 导出配置
└── docs/
    └── ONNX_EXPORT_PLAN.md         # 本文件
```

## 第一版实现要点（与上文计划的差异）

1. **模型来源改为 `build_sam3_multiplex_video_predictor`**（与 `core/engine.py` 推理同一条
   构建链），而不是 `build_sam3_image_model`：计划中的 A（Sam3TriViTDetNeck）、D
   （TransformerEncoderDecoupledCrossAttention）、E（MultiplexMaskDecoder）只存在于
   multiplex 模型里，image model 用的是 Sam3DualViTDetNeck（4 级）且无 D/E 对应类。
   一次构建供全部 6 个组件；`model.path: null` 时 HF 下载的是 **facebook/sam3.1**
   （不是 sam3）。提取路径见 `core/export/utils.py: load_multiplex_predictor` docstring。
2. **导出期 monkeypatch（不改子模块文件）**，`core/export/utils.py: export_patches`：
   - `vitdet.addmm_act`（aten._addmm_activation 融合算子，grad enabled 时直接 raise，
     无法在 tracer 下运行）→ 普通 fp32 Linear+激活。注意原版内部强制 bf16，导出的
     fp32 图与线上 bf16 推理存在 bf16 级数值差；验证对比的是补丁后 fp32 参考。
   - `decoder.sdpa_kernel(FLASH_ATTENTION)` → nullcontext（fp32/CPU 无 flash kernel
     会报错；让 SDPA 自动选后端，数值等价）。
3. **导出必须 `use_rope_real=True`**（复数 complex64 RoPE buffer ONNX 不支持；实数路径
   数值等价）且 **需要 CUDA**（构建链与 `PositionEmbeddingSine` 预计算硬编码 cuda）。
4. **A 的实际输出是 18 个 tensor**：3 头（sam3/interactive/propagation）× 3 级
   （stride 3.5/7/14，即计划的 H/4、H/2、H 是近似值）×（特征 + 位置编码），命名
   `{head}_{fpn,pos}_{0,1,2}`。
5. **B 多一个 `text_embeds` 输出**（resizer 前的 [32,B,1024] token 嵌入，detector
   transformer 会用到）；`text_mask` 为 bool（True=padding）。
6. **C 固定 boxes=None**（与交互调用点一致），默认导出 points-only 变体
   （`export_prompt_encoder.WITH_MASK=False`；置 True 可带 mask 提示输入）。
7. **D 的 `num_obj_ptr_tokens` 固化为 multiplex_count=16**（单条件帧场景），dummy 输入
   按 1 帧记忆构造；记忆帧数变化属于形状变化，静态图需按帧数各导一份。
8. **E 的 `multimask_output` 固化为 True**（multiplex 配置 `multimask_outputs_only=True`
   要求必须为 True，同时避开 `dynamic_multimask_via_stability` 的数据依赖分支）。
   `image_pe` 作为输入保留（运行时是常数 dense PE，端侧可按分辨率预生成）。
9. **F 的 mask 输入直接给 `interpol_size`（16×feat）分辨率**，跳过
   `SimpleMaskDownSampler` 里 `antialias=True` 的双线性插值（torch.onnx 不支持
   antialias）；端侧需自行把 4×feat 的 mask 双线性缩放到 16×feat。
   sigmoid/scale/bias 与 mux/条件通道拼装在运行时胶水代码完成（`skip_mask_sigmoid=True`）。
10. **验证**：`verify.enabled` 时用 onnxruntime（CPUExecutionProvider）跑同一随机输入，
    逐输出对比 PyTorch 参考，容差 fp32 1e-5 / fp16 1e-2。
11. **依赖**：`onnx`（必需）、`onnxruntime`（verify）、`onnxsim`（simplify）、
    `onnxconverter-common`（fp16）以注释形式列在 requirements.txt，按需安装。

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
