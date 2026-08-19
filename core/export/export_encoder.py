"""子模型 A: 图像编码器 (ViT + Sam3TriViTDetNeck)

来源: multiplex predictor 的 detector.backbone.vision_backbone (运行时实际
使用的就是 detector 侧的 tri neck; tracker 自带的 backbone 在 predictor 构建
时被删除)。

输入 [1,3,H,W] → 3 头 (sam3/interactive/propagation) × 3 级 FPN
(stride 3.5/7/14) 的特征 + 位置编码, 共 18 个输出 tensor。
"""

from typing import Dict, List, Tuple

import torch

from .utils import ExportOptions, export_component
from .wrappers import ImageEncoderWrapper

NAME = "image_encoder"

_HEADS = ("sam3", "interactive", "propagation")
_LEVELS = 3

INPUT_NAMES: List[str] = ["image"]
OUTPUT_NAMES: List[str] = [
    f"{head}_{kind}_{i}"
    for head in _HEADS
    for i in range(_LEVELS)
    for kind in ("fpn", "pos")
]


def build_wrapper(predictor, resolution: int) -> ImageEncoderWrapper:
    neck = predictor.model.detector.backbone.vision_backbone
    return ImageEncoderWrapper(neck)


def dummy_inputs(resolution: int, device) -> Tuple[torch.Tensor, ...]:
    return (torch.randn(1, 3, resolution, resolution, device=device),)


def dynamic_axes() -> Dict:
    axes = {"image": {0: "batch"}}
    for name in OUTPUT_NAMES:
        axes[name] = {0: "batch"}
    return axes


def export(predictor, options: ExportOptions, output_dir):
    wrapper = build_wrapper(predictor, options.resolution)
    device = next(wrapper.parameters()).device
    inputs = dummy_inputs(options.resolution, device)
    return export_component(
        NAME, wrapper, inputs, INPUT_NAMES, OUTPUT_NAMES, options, output_dir,
        dynamic_axes=dynamic_axes(),
    )
