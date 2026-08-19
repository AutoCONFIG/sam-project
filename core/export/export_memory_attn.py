"""子模型 D: 记忆注意力 (TransformerEncoderDecoupledCrossAttention)

来源: tracker.transformer.encoder。dummy 输入按「单个条件帧 + 一帧记忆 +
multiplex_count 个 obj ptr token」构造 (num_obj_ptr_tokens 固化为
multiplex_count=16); 记忆帧数变化属于形状变化, 静态导出下需要按帧数各导
一份或自行用 dynamic 轴扩展。

依赖 export_patches 里的 sdpa_kernel 补丁 (fp32 下 flash 后端不可用)。
"""

from typing import Dict, List, Tuple

import torch

from .utils import ExportOptions, export_component, get_tracker_model
from .wrappers import MemoryAttentionWrapper

NAME = "memory_attn"

INPUT_NAMES: List[str] = [
    "image",
    "src",
    "memory_image",
    "memory",
    "image_pos",
    "src_pos",
    "memory_image_pos",
    "memory_pos",
]
OUTPUT_NAMES: List[str] = ["fused_memory", "pos_embed"]


def build_wrapper(predictor, resolution: int) -> MemoryAttentionWrapper:
    tracker = get_tracker_model(predictor)
    # 单个条件帧时 obj ptr token 数 = multiplex_count (每 bucket 一个)
    return MemoryAttentionWrapper(
        tracker.transformer.encoder, num_obj_ptr_tokens=tracker.multiplex_count
    )


def dummy_inputs(predictor, resolution: int, device) -> Tuple[torch.Tensor, ...]:
    # hidden_dim=256; 序列均为 seq-first [S, 1, C]
    tracker = get_tracker_model(predictor)
    C = tracker.hidden_dim
    feat = resolution // 14
    hw = feat * feat
    num_obj_ptr = tracker.multiplex_count
    mem = hw + num_obj_ptr  # 1 帧记忆图像特征 + obj ptr tokens

    def randn(*shape):
        return torch.randn(*shape, device=device)

    return (
        randn(hw, 1, C),  # image
        randn(hw, 1, C),  # src
        randn(hw, 1, C),  # memory_image
        randn(mem, 1, C),  # memory (含 obj ptr)
        randn(hw, 1, C),  # image_pos
        randn(hw, 1, C),  # src_pos
        randn(hw, 1, C),  # memory_image_pos
        randn(mem, 1, C),  # memory_pos
    )


def dynamic_axes() -> Dict:
    # seq-first 布局, batch 在 dim 1
    axes = {name: {1: "batch"} for name in INPUT_NAMES}
    axes["fused_memory"] = {1: "batch"}
    axes["pos_embed"] = {1: "batch"}
    return axes


def export(predictor, options: ExportOptions, output_dir):
    wrapper = build_wrapper(predictor, options.resolution)
    device = next(wrapper.parameters()).device
    inputs = dummy_inputs(predictor, options.resolution, device)
    return export_component(
        NAME, wrapper, inputs, INPUT_NAMES, OUTPUT_NAMES, options, output_dir,
        dynamic_axes=dynamic_axes(),
    )
