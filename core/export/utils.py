"""ONNX 导出公共逻辑

- load_multiplex_predictor: 构建 SAM 3.1 multiplex 模型, 6 个子模型的共同来源
  (与 core/engine.py 推理同一条构建链, checkpoint 加载逻辑由子模块自己处理)
- export_patches: 导出期 monkeypatch (运行时补丁, 不改动子模块文件)
- export_component: torch.onnx.export + onnx.checker + onnxsim + fp16 + 验证
- verify_onnx: onnxruntime 与 PyTorch 参考输出逐 tensor 对比

所有 sam3 import 都在函数内部, 调用前需 setup_sam3_path()。

导出期补丁的两个原因:
1. vitdet.Mlp.forward 走 sam3.perflib.fused.addmm_act (aten._addmm_activation),
   该函数在 torch.is_grad_enabled() 时直接 raise, 而 torch.onnx.export 的
   tracer 需要 grad graph → 替换为普通的 Linear+激活 (fp32; 原版内部强制
   bf16, 补丁版与其存在 bf16 级别的数值差, 验证对比的是补丁后的 fp32 参考)。
2. decoder.functional_attention 用 sdpa_kernel(SDPBackend.FLASH_ATTENTION)
   把 SDPA 限定到 flash 后端, fp32/CPU 下没有可用 kernel 会直接报错 →
   替换为 nullcontext, 让 SDPA 自动选后端 (数值等价, 算子照常导出)。
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from utils.config import SAM3_ROOT
from utils.constants import DEFAULT_IMAGE_SIZE

# BPE tokenizer 资源 (构建 text encoder 时需要; tokenization 本身不导出)
BPE_PATH = SAM3_ROOT / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"

# fp16 导出时验证容差下限
FP16_TOLERANCE = 1.0e-2


@dataclass
class ExportOptions:
    """一次导出运行的公共参数 (对应 YAML 的 model.resolution / export.* / verify.*)"""

    resolution: int = DEFAULT_IMAGE_SIZE
    opset: int = 17
    dynamic: bool = False
    simplify: bool = True
    fp16: bool = False
    verify: bool = True
    tolerance: float = 1.0e-5


def load_multiplex_predictor(
    checkpoint_path: Optional[str] = None,
    resolution: int = DEFAULT_IMAGE_SIZE,
    bpe_path: Optional[str] = None,
    finetune_ckpt: Optional[str] = None,
):
    """构建 SAM 3.1 multiplex 模型 (predictor 外壳), 返回组件提取入口。

    6 个子模型的提取路径:
      A image_encoder:   predictor.model.detector.backbone.vision_backbone
      B text_encoder:    predictor.model.detector.backbone.language_backbone
      C prompt_encoder:  predictor.model.tracker.model.interactive_sam_prompt_encoder
      D memory_attn:     predictor.model.tracker.model.transformer.encoder
      E mask_decoder:    predictor.model.tracker.model.sam_mask_decoder
      F memory_encoder:  predictor.model.tracker.model.maskmem_backbone

    checkpoint_path=None 时从 HuggingFace 下载 facebook/sam3.1 原版权重
    (gated repo, 需要 HF token)。

    finetune_ckpt: 训练链产出的微调 checkpoint (image model 裸 state_dict,
    可包在 {"model": ...} 里), 构建后加载进 detector (与 core/engine.py
    推理侧 finetune_ckpt 同一约定), 用于导出自定义微调的模型。

    注意: 构建链硬编码 cuda (demo_model.cuda().eval() 与位置编码预计算),
    导出必须在有 CUDA GPU 的机器上跑。
    use_fa3=False / use_rope_real=True 是 ONNX 导出的硬性要求:
    flash-attn 无法导出; 复数 RoPE buffer (complex64) ONNX 不支持,
    use_rope_real=True 走实数路径 (数值等价)。
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "ONNX 导出需要 CUDA GPU: SAM3 multiplex 构建链 "
            "(build_sam3_multiplex_video_predictor → demo_model.cuda()) 与 "
            "PositionEmbeddingSine 的预计算都硬编码 cuda"
        )
    from sam3.model_builder import build_sam3_multiplex_video_predictor

    if bpe_path is None:
        bpe_path = str(BPE_PATH)

    predictor = build_sam3_multiplex_video_predictor(
        checkpoint_path=checkpoint_path,
        bpe_path=bpe_path,
        use_fa3=False,
        use_rope_real=True,
        compile=False,
        warm_up=False,
        async_loading_frames=False,
        image_size=resolution,
    )
    if finetune_ckpt:
        load_finetune_into_detector(predictor, finetune_ckpt)
    return predictor


def load_finetune_into_detector(predictor, ckpt_path: str) -> None:
    """把微调 checkpoint (image model 裸 state_dict) 加载进 multiplex detector。

    与 core/engine.py 的推理侧同一约定 (strict=False; RoPE buffer 一律丢弃,
    用模型按当前 resolution 预计算的那份)。
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    rope_suffixes = ("freqs_cis", "freqs_cis_real", "freqs_cis_imag")
    ckpt = {k: v for k, v in ckpt.items() if not k.endswith(rope_suffixes)}
    missing, unexpected = predictor.model.detector.load_state_dict(ckpt, strict=False)
    print(f"已加载微调权重: {ckpt_path} → detector "
          f"({len(ckpt)} 键, missing {len(missing)}, unexpected {len(unexpected)})")
    if unexpected:
        print(f"  警告: 有 {len(unexpected)} 个未匹配键 (前 5 个: {unexpected[:5]})")


def get_tracker_model(predictor):
    """multiplex predictor → tracker 模型 (VideoTrackingDynamicMultiplexDemo)。"""
    return predictor.model.tracker.model


@contextlib.contextmanager
def export_patches():
    """导出期 monkeypatch 上下文 (见模块 docstring)。只 patch 进程内引用, 不改文件。"""
    import torch.nn.functional as F
    from torch import nn

    import sam3.model.decoder as decoder_mod
    import sam3.model.vitdet as vitdet_mod

    def addmm_act_export(activation, linear, mat1):
        # 原版: bf16 的 aten._addmm_activation 融合算子, grad enabled 时 raise。
        # 补丁版: 普通 fp32 Linear + 激活, 可正常 trace。
        y = F.linear(mat1, linear.weight, linear.bias)
        if activation in (F.relu, nn.ReLU):
            return F.relu(y)
        if activation in (F.gelu, nn.GELU):
            return F.gelu(y)
        raise ValueError(f"Unexpected activation {activation}")

    orig_addmm_act = vitdet_mod.addmm_act
    orig_sdpa_kernel = decoder_mod.sdpa_kernel
    vitdet_mod.addmm_act = addmm_act_export
    decoder_mod.sdpa_kernel = lambda *args, **kwargs: contextlib.nullcontext()
    try:
        yield
    finally:
        vitdet_mod.addmm_act = orig_addmm_act
        decoder_mod.sdpa_kernel = orig_sdpa_kernel


def export_component(
    name: str,
    wrapper: torch.nn.Module,
    dummy_inputs: Sequence[torch.Tensor],
    input_names: List[str],
    output_names: List[str],
    options: ExportOptions,
    output_dir: Path,
    dynamic_axes: Optional[Dict] = None,
) -> Path:
    """导出单个子模型: PyTorch 参考 → torch.onnx.export → 后处理 → 验证。"""
    try:
        import onnx
    except ImportError:
        raise ImportError("ONNX 导出需要 onnx: pip install onnx")

    wrapper.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / f"{name}.onnx"

    # PyTorch 参考输出 (与导出共用同一套 dummy 输入和补丁环境)
    with torch.no_grad():
        ref_outputs = [o.detach().cpu() for o in wrapper(*dummy_inputs)]
    assert len(ref_outputs) == len(output_names), (
        f"{name}: wrapper 输出数 ({len(ref_outputs)}) 与 output_names "
        f"({len(output_names)}) 不一致"
    )

    # 固定形状静态 trace; dynamo=False 走 legacy tracer (本代码库的控制流
    # 都是静态可解析的, legacy 行为更可预期)
    torch.onnx.export(
        wrapper,
        tuple(dummy_inputs),
        str(onnx_path),
        input_names=input_names,
        output_names=output_names,
        opset_version=options.opset,
        dynamic_axes=dynamic_axes if options.dynamic else None,
        dynamo=False,
    )

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    if options.simplify:
        onnx_model = _maybe_simplify(onnx_model, onnx_path)
    if options.fp16:
        onnx_model = _to_fp16(onnx_model, onnx_path)

    if options.verify:
        tolerance = max(options.tolerance, FP16_TOLERANCE) if options.fp16 else options.tolerance
        verify_onnx(onnx_path, ref_outputs, dummy_inputs, input_names, output_names, tolerance)

    size_mb = onnx_path.stat().st_size / 1024 / 1024
    print(f"  已导出: {onnx_path} ({size_mb:.1f} MB)")
    return onnx_path


def _maybe_simplify(onnx_model, onnx_path: Path):
    """onnxsim 简化; 未安装或简化失败时告警并保留原图。"""
    try:
        from onnxsim import simplify
    except ImportError:
        print("  警告: 未安装 onnxsim, 跳过简化 (pip install onnxsim)")
        return onnx_model
    simplified, ok = simplify(onnx_model)
    if not ok:
        print("  警告: onnxsim 简化校验失败, 保留原图")
        return onnx_model
    onnx.save(simplified, str(onnx_path))
    return simplified


def _to_fp16(onnx_model, onnx_path: Path):
    """转 fp16 (IO 保持 fp32, 便于与 fp32 参考对比)。"""
    import onnx
    try:
        from onnxconverter_common import float16
    except ImportError:
        raise ImportError(
            "fp16 转换需要 onnxconverter-common: pip install onnxconverter-common"
        )
    model_fp16 = float16.convert_float_to_float16(onnx_model, keep_io_types=True)
    onnx.save(model_fp16, str(onnx_path))
    return model_fp16


def verify_onnx(
    onnx_path: Path,
    ref_outputs: Sequence[torch.Tensor],
    dummy_inputs: Sequence[torch.Tensor],
    input_names: List[str],
    output_names: List[str],
    tolerance: float,
) -> None:
    """用 onnxruntime (CPU) 跑同一输入, 与 PyTorch 参考输出逐 tensor 对比。"""
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError("导出验证需要 onnxruntime: pip install onnxruntime")

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    feeds = {
        name: t.detach().cpu().numpy() for name, t in zip(input_names, dummy_inputs)
    }
    ort_outputs = sess.run(output_names, feeds)

    print(f"  数值验证 (onnxruntime vs PyTorch, 容差 {tolerance:g}):")
    max_diff_all = 0.0
    for name, ref, out in zip(output_names, ref_outputs, ort_outputs):
        ref_np = ref.numpy()
        if ref_np.shape != out.shape:
            raise AssertionError(
                f"  {name}: shape 不一致 torch{tuple(ref_np.shape)} vs ort{tuple(out.shape)}"
            )
        # bool 输出 (text_mask) 不能直接相减, 统一转 fp32 再比
        diff = (
            float(abs(ref_np.astype("float32") - out.astype("float32")).max())
            if ref_np.size
            else 0.0
        )
        max_diff_all = max(max_diff_all, diff)
        status = "OK" if diff <= tolerance else "超差"
        print(f"    {name}: max|diff| = {diff:.3e} [{status}]")
        if diff > tolerance:
            raise AssertionError(
                f"  {name}: 数值差异 {diff:.3e} 超过容差 {tolerance:g}"
            )
    print(f"  验证通过 (最大差异 {max_diff_all:.3e})")
