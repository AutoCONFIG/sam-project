"""Shared constants for the SAM 3 project.

Centralizes default values to avoid scattering hard-coded literals across
the codebase.
"""

from typing import Final

# ═══════════════════════════════════════════════════════════════════════════════
#  Path defaults
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_PREDICT_OUTPUT: Final[str] = "runs/predict"

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

# ═══════════════════════════════════════════════════════════════════════════════
#  Prompt defaults
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_FRAME_INDEX: Final[int] = 0

# ═══════════════════════════════════════════════════════════════════════════════
#  Output defaults
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_SAVE_VIS: Final[bool] = True
DEFAULT_SAVE_MASKS: Final[bool] = True
DEFAULT_SAVE_VIDEO: Final[bool] = False

# ═══════════════════════════════════════════════════════════════════════════════
#  Label export defaults
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_SAVE_LABELS: Final[bool] = True
# supported: "coco", "yolo"
DEFAULT_LABEL_FORMATS: Final[tuple] = ("coco", "yolo")
