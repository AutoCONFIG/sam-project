"""SAM 3 ONNX 导出

按 docs/ONNX_EXPORT_PLAN.md 的拆分方案, 把 SAM 3.1 multiplex 模型拆成
6 个可独立导出的子模型, 分别导出 ONNX 并用 onnxruntime 做数值验证:

  A image_encoder   ViT + Sam3TriViTDetNeck   (export_encoder.py)
  B text_encoder    TextTransformer + resizer  (export_text_encoder.py)
  C prompt_encoder  PromptEncoder              (export_prompt_encoder.py)
  D memory_attn     TransformerEncoderDecoupledCrossAttention (export_memory_attn.py)
  E mask_decoder    MultiplexMaskDecoder       (export_mask_decoder.py)
  F memory_encoder  SimpleMaskEncoder          (export_memory_encoder.py)

所有组件来自同一个 multiplex predictor 构建 (见 utils.load_multiplex_predictor)。
"""

from pathlib import Path
from typing import Dict, List

from utils.config import get_nested_value
from utils.constants import DEFAULT_IMAGE_SIZE, IMAGE_SIZE_STEP

from . import (
    export_encoder,
    export_mask_decoder,
    export_memory_attn,
    export_memory_encoder,
    export_prompt_encoder,
    export_text_encoder,
)
from .utils import ExportOptions, export_patches, load_multiplex_predictor

# component 名 → 导出器模块 (每个模块提供 export(predictor, options, output_dir))
EXPORTERS = {
    "image_encoder": export_encoder,
    "text_encoder": export_text_encoder,
    "prompt_encoder": export_prompt_encoder,
    "memory_attn": export_memory_attn,
    "mask_decoder": export_mask_decoder,
    "memory_encoder": export_memory_encoder,
}


def run_export(config: Dict) -> None:
    """按配置导出 ONNX 子模型。"""
    # ── Extract config ──────────────────────────────────────────────────
    model_path = get_nested_value(config, "model", "path")
    finetune_ckpt = get_nested_value(config, "model", "finetune_ckpt")
    resolution = get_nested_value(config, "model", "resolution", default=DEFAULT_IMAGE_SIZE)

    components = get_nested_value(
        config, "export", "components", default=["image_encoder", "text_encoder"]
    )
    options = ExportOptions(
        resolution=int(resolution),
        opset=int(get_nested_value(config, "export", "opset", default=17)),
        dynamic=bool(get_nested_value(config, "export", "dynamic", default=False)),
        simplify=bool(get_nested_value(config, "export", "simplify", default=True)),
        fp16=bool(get_nested_value(config, "export", "fp16", default=False)),
        verify=bool(get_nested_value(config, "verify", "enabled", default=True)),
        tolerance=float(get_nested_value(config, "verify", "tolerance", default=1.0e-5)),
    )
    output_dir = Path(get_nested_value(config, "output", "dir", default="runs/export/default"))

    # ── Validate ────────────────────────────────────────────────────────
    if options.resolution % IMAGE_SIZE_STEP != 0:
        raise ValueError(
            f"model.resolution 必须是 {IMAGE_SIZE_STEP} 的倍数 (得到 {options.resolution})"
        )
    if isinstance(components, str):
        components = [components]
    components: List[str] = [str(c).strip() for c in components if str(c).strip()]
    if "all" in components:
        components = list(EXPORTERS)
    unknown = [c for c in components if c not in EXPORTERS]
    if unknown:
        raise ValueError(
            f"未知组件: {unknown}; 可选: {list(EXPORTERS)} 或 all"
        )
    if not components:
        raise ValueError("export.components 不能为空 (可选: image_encoder, text_encoder, "
                         "prompt_encoder, memory_attn, mask_decoder, memory_encoder, all)")

    print(f"\n{'='*60}")
    print("SAM 3.1 ONNX 导出")
    print(f"{'='*60}")
    print(f"模型权重: {model_path or 'HF 自动下载 (facebook/sam3.1)'}")
    if finetune_ckpt:
        print(f"微调权重: {finetune_ckpt} (加载进 detector)")
    print(f"导出分辨率: {options.resolution}")
    print(f"组件: {components}")
    print(f"opset: {options.opset} | dynamic: {options.dynamic} | "
          f"simplify: {options.simplify} | fp16: {options.fp16}")
    print(f"输出目录: {output_dir}")
    print(f"数值验证: {options.verify} (容差 {options.tolerance:g})")
    print(f"{'='*60}\n")

    # ── Build model once, export each component ─────────────────────────
    print("构建 multiplex 模型 (与 predict 推理同一构建链)...")
    predictor = load_multiplex_predictor(model_path, options.resolution,
                                         finetune_ckpt=finetune_ckpt)
    print("模型构建完成\n")

    results: Dict[str, Path] = {}
    with export_patches():
        for name in components:
            print(f"[{name}] 导出中...")
            results[name] = EXPORTERS[name].export(predictor, options, output_dir)

    print(f"\n{'='*60}")
    print("导出完成:")
    for name, path in results.items():
        print(f"  {name}: {path}")
    print(f"{'='*60}")
