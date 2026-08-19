"""
SAM 3 ONNX Export Module
========================
Export the SAM 3.1 multiplex model as independent ONNX sub-models
(6 components, see docs/ONNX_EXPORT_PLAN.md).

All components come from one multiplex model build (the same chain as
predict). Each component is exported as ``<component>.onnx`` under
``output.dir`` and optionally verified against PyTorch with onnxruntime.

Usage::

    python sam.py configs/export/default.yaml
    python -m commands.export --config configs/export/default.yaml --components all
    python sam.py configs/export/default.yaml --model-path pretrain/sam3.1/sam3.1_multiplex.pt --resolution 672
"""

import argparse
import sys
import traceback
from typing import Any, Dict

from utils.config import (
    config_from_args,
    load_yaml_config,
    merge_configs,
    set_boolean_argument,
    setup_sam3_path,
)
from utils.constants import DEFAULT_IMAGE_SIZE, IMAGE_SIZE_STEP

setup_sam3_path()

_COMPONENT_CHOICES = [
    "image_encoder",
    "text_encoder",
    "prompt_encoder",
    "memory_attn",
    "mask_decoder",
    "memory_encoder",
    "all",
]


# ─── Argument parser ────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SAM 3.1 ONNX 导出 (multiplex 模型拆分为 6 个子模型)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python sam.py configs/export/default.yaml
    python sam.py configs/export/default.yaml --components all
    python sam.py configs/export/default.yaml --resolution 672 --no-verify
    python -m commands.export --config configs/export/default.yaml --fp16
        """,
    )

    parser.add_argument("--config", type=str, default=None, help="YAML 配置文件路径")
    parser.add_argument("--model-path", type=str, default=None,
                        help="模型权重路径 (sam3.1 multiplex .pt; 缺省从 HF 下载原版)")
    parser.add_argument("--finetune-ckpt", type=str, default=None,
                        help="微调 checkpoint (训练链产出的 image model 权重; 构建后加载进 detector)")
    parser.add_argument("--resolution", type=int, default=None,
                        help=f"导出固定分辨率, 须为 {IMAGE_SIZE_STEP} 的倍数 "
                             f"(默认 {DEFAULT_IMAGE_SIZE}; 16G 显存建议 672)")
    parser.add_argument("--components", type=str, default=None, nargs="+",
                        choices=_COMPONENT_CHOICES,
                        help="要导出的子模型, 可多选或 all (默认 image_encoder text_encoder)")
    parser.add_argument("--opset", type=int, default=None,
                        help="ONNX opset (默认 17; RKNN Toolkit2 上限 19)")

    set_boolean_argument(parser, "dynamic", "dynamic",
                         help_true="导出动态 batch 轴 (ONNX Runtime 用; RKNN 偏好静态)",
                         help_false="静态形状 (默认)")
    set_boolean_argument(parser, "simplify", "simplify",
                         help_true="导出后用 onnxsim 简化 (默认)",
                         help_false="不用 onnxsim 简化")
    set_boolean_argument(parser, "fp16", "fp16",
                         help_true="导出后转 fp16 (需 onnxconverter-common)",
                         help_false="保持 fp32 (默认)")
    set_boolean_argument(parser, "verify", "verify",
                         help_true="导出后用 onnxruntime 对比 PyTorch 输出 (默认)",
                         help_false="跳过数值验证")

    parser.add_argument("--output-dir", "-o", type=str, default=None,
                        help="输出目录 (默认 runs/export/default)")
    parser.add_argument("--tolerance", type=float, default=None,
                        help="数值验证容差 (默认 1e-5; fp16 时自动放宽到 1e-2)")

    return parser.parse_args()


def args_to_config(args: argparse.Namespace) -> Dict[str, Any]:
    """将命令行参数转换为嵌套配置字典。"""
    config: Dict[str, Any] = {}

    model_cfg = config_from_args(
        args, plain=("model_path", "resolution", "finetune_ckpt"),
        rename={"model_path": "path"},
    )
    if model_cfg:
        config["model"] = model_cfg

    export_cfg = config_from_args(
        args,
        plain=("components", "opset"),
        boolean=("dynamic", "simplify", "fp16"),
    )
    if export_cfg:
        config["export"] = export_cfg

    output_cfg = config_from_args(
        args, plain=("output_dir",), rename={"output_dir": "dir"}
    )
    if output_cfg:
        config["output"] = output_cfg

    verify_cfg = config_from_args(
        args, plain=("tolerance",), boolean=("verify",), rename={"verify": "enabled"}
    )
    if verify_cfg:
        config["verify"] = verify_cfg

    return config


def main():
    args = parse_args()
    try:
        config = {}
        if args.config:
            config = load_yaml_config(args.config)
        cli_config = args_to_config(args)
        config = merge_configs(config, cli_config)

        from core.export import run_export

        run_export(config)
    except KeyboardInterrupt:
        print("\n导出被用户中断。")
        sys.exit(130)
    except Exception as e:
        print(f"\n{'='*60}\n错误: {e}\n{'='*60}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
