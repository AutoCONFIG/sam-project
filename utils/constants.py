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
