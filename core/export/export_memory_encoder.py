"""子模型 F: 记忆编码器 (SimpleMaskEncoder, multiplex maskmem_backbone)

来源: tracker.maskmem_backbone。skip_mask_sigmoid 固化为 True (与
_encode_new_memory 调用点一致: sigmoid 及 scale/bias 已在喂入前完成);
mask 输入直接给 interpol_size (16×feat) 分辨率, 跳过 SimpleMaskDownSampler
里 ONNX 不支持的 antialias 双线性插值 — 端侧需自行把 4×feat 分辨率的 mask
双线性缩放到 16×feat。
"""

from typing import Dict, List, Tuple

import torch

from .utils import ExportOptions, export_component, get_tracker_model
from .wrappers import MemoryEncoderWrapper

NAME = "memory_encoder"

INPUT_NAMES: List[str] = ["pix_feat", "masks"]
OUTPUT_NAMES: List[str] = ["memory_features", "memory_pos_enc"]


def build_wrapper(predictor, resolution: int) -> MemoryEncoderWrapper:
    tracker = get_tracker_model(predictor)
    return MemoryEncoderWrapper(tracker.maskmem_backbone)


def dummy_inputs(predictor, resolution: int, device) -> Tuple[torch.Tensor, ...]:
    tracker = get_tracker_model(predictor)
    maskmem = tracker.maskmem_backbone
    C = maskmem.pix_feat_proj.in_channels  # 256
    feat = resolution // 14
    # multiplex_count × input_channel_multiplier (16 对象 mask 通道 + 16 条件通道)
    mask_channels = maskmem.mask_downsampler.multiplex_count
    # 直接给 downsampler 的 interpol_size 输入分辨率, 跳过 antialias 插值
    mask_size = maskmem.mask_downsampler.interpol_size[0]
    return (
        torch.randn(1, C, feat, feat, device=device),  # pix_feat
        torch.randn(1, mask_channels, mask_size, mask_size, device=device),  # masks
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
