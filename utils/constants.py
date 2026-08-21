"""Shared constants for the SAM 3 project.

Centralizes default values to avoid scattering hard-coded literals across
the codebase.
"""

from typing import Final

# ═══════════════════════════════════════════════════════════════════════════════
#  Path defaults
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_PREDICT_OUTPUT: Final[str] = "runs/predict"

# Image extensions accepted by both input scanning (core/io_dispatch.py) and
# frame loading (core/engine.py) — single source of truth to keep them aligned.
IMAGE_EXT: Final[frozenset] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
)

# ═══════════════════════════════════════════════════════════════════════════════
#  Model defaults
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_IMAGE_SIZE: Final[int] = 1008
# image_size must be a multiple of 336 (14×24) for windowed attention.
# 1008 = default (requires ≥24GB VRAM); 672 works on 16GB GPUs.
IMAGE_SIZE_STEP: Final[int] = 336

# Model version: "sam3.1" (multiplex, recommended) or "sam3" (base).
# sam3.1 supports image_size parameterization; sam3 is fixed at 1008.
DEFAULT_MODEL_VERSION: Final[str] = "sam3.1"

DEFAULT_USE_FA3: Final[bool] = False
DEFAULT_USE_ROPE_REAL: Final[bool] = True
DEFAULT_COMPILE: Final[bool] = False

# sam3.1 (multiplex) 专用: 一个 session 最多同时跟踪的对象数 (超出丢弃并告警
# "hitting tracking_obj.max_num_objects"); 运行时上限, 可自由调大 (显存随之涨)
DEFAULT_MAX_NUM_OBJECTS: Final[int] = 16
# sam3.1 专用: 每个 multiplex 桶的对象容量 — 结构参数! 已固化进 checkpoint
# 权重形状 (预训练=16), 改它会与预训练权重 size mismatch 崩溃; 仅自训模型可改
DEFAULT_MULTIPLEX_COUNT: Final[int] = 16

# ═══════════════════════════════════════════════════════════════════════════════
#  Prompt defaults
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_FRAME_INDEX: Final[int] = 0

# ═══════════════════════════════════════════════════════════════════════════════
#  Output defaults
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_SAVE_VIS: Final[bool] = True
DEFAULT_SAVE_MASKS: Final[bool] = True
DEFAULT_SAVE_VIDEO: Final[bool] = True

# ═══════════════════════════════════════════════════════════════════════════════
#  Label export defaults
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_SAVE_LABELS: Final[bool] = True
# supported: "coco", "yolo"
DEFAULT_LABEL_FORMATS: Final[tuple] = ("coco", "yolo")
