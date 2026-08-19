"""子模型 C: 提示编码器 (SAM 风格 PromptEncoder)

来源: tracker.interactive_sam_prompt_encoder。固定 boxes=None (与原调用点
_forward_sam_heads 的交互路径一致, 点提示总是 pad 一个 dummy 点)。

默认导出 points-only 变体 (with_mask=False, dense 输出为 no_mask_embed
广播); 需要 mask 提示路径时把 build_wrapper 的 with_mask 设为 True,
dummy_inputs/INPUT_NAMES 相应带上 mask_input。
"""

from typing import Dict, List, Tuple

import torch

from .utils import ExportOptions, export_component
from .wrappers import PromptEncoderWrapper

NAME = "prompt_encoder"

_NUM_POINTS = 5  # dummy 输入的点数 (仅影响 trace 示例, 不影响图结构)

# 是否与 mask 提示一起导出 (True 时输入额外带 mask_input [B,1,4F,4F])
WITH_MASK = False


def _input_names() -> List[str]:
    names = ["point_coords", "point_labels"]
    if WITH_MASK:
        names.append("mask_input")
    return names


INPUT_NAMES: List[str] = _input_names()
OUTPUT_NAMES: List[str] = ["sparse_embeddings", "dense_embeddings"]


def build_wrapper(predictor, resolution: int) -> PromptEncoderWrapper:
    tracker = predictor.model.tracker.model
    return PromptEncoderWrapper(tracker.interactive_sam_prompt_encoder, with_mask=WITH_MASK)


def dummy_inputs(resolution: int, device) -> Tuple[torch.Tensor, ...]:
    # 坐标取图内均匀随机点; 标签取 0/1 (前景点/背景点)
    point_coords = torch.rand(1, _NUM_POINTS, 2, device=device) * resolution
    point_labels = torch.randint(0, 2, (1, _NUM_POINTS), dtype=torch.int64, device=device)
    inputs = [point_coords, point_labels]
    if WITH_MASK:
        feat = resolution // 14
        inputs.append(torch.randn(1, 1, 4 * feat, 4 * feat, device=device))
    return tuple(inputs)


def dynamic_axes() -> Dict:
    return {
        "point_coords": {0: "batch"},
        "point_labels": {0: "batch"},
        "sparse_embeddings": {0: "batch"},
        "dense_embeddings": {0: "batch"},
    }


def export(predictor, options: ExportOptions, output_dir):
    wrapper = build_wrapper(predictor, options.resolution)
    device = next(wrapper.parameters()).device
    inputs = dummy_inputs(options.resolution, device)
    return export_component(
        NAME, wrapper, inputs, INPUT_NAMES, OUTPUT_NAMES, options, output_dir,
        dynamic_axes=dynamic_axes(),
    )
