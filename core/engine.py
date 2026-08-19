"""SAM 3 video inference engine.

Wraps the SAM 3 video predictor (sam3 or sam3.1) into a clean session-based
API: build model → start_session → add_prompt → propagate → close_session.

The underlying model uses a global bf16 autocast context (enabled in the
predictor ``__init__``), so weights stay fp32 and forward passes automatically
use bf16. Do NOT manually convert weights to bf16 — the decoder FFN contains
``autocast(enabled=False)`` which conflicts with bf16 weights.

Both model versions share the same ``handle_request`` / ``handle_stream_request``
API via the unified ``build_sam3_predictor`` entry point:

- ``version="sam3.1"`` — Object Multiplex, supports ``image_size`` parameterization
  (16GB GPU can use 672). This is the recommended default.
- ``version="sam3"``   — base dense tracking, ``image_size`` is fixed at 1008
  (requires ≥24GB VRAM).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Generator

import cv2
import numpy as np

# Reduce CUDA memory fragmentation (must be set before torch import).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from utils.constants import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_USE_FA3,
    DEFAULT_USE_ROPE_REAL,
    DEFAULT_COMPILE,
    DEFAULT_MODEL_VERSION,
    IMAGE_EXT,
)


class Sam3VideoPredictor:
    """Session-based wrapper around the SAM 3 video predictor.

    Parameters
    ----------
    checkpoint : str
        Path to the model checkpoint (``sam3.1_multiplex.pt`` or ``sam3.pt``).
    version : str
        Model version — ``"sam3.1"`` (multiplex, recommended) or ``"sam3"``
        (base dense tracking). Determines which backend builder is used.
    image_size : int
        Inference resolution. Must be a multiple of 336 (14×24) for windowed
        attention. Default 1008 (≥24GB VRAM). Use 672 for 16GB GPUs.
        NOTE: only ``sam3.1`` honors this; ``sam3`` is fixed at 1008.
    use_fa3 : bool
        Enable Flash Attention 3 (requires flash-attn installed). Default False.
    use_rope_real : bool
        Use real-valued RoPE (avoids complex64 position buffers). Default True.
    compile : bool
        Enable torch.compile (slow first run). Default False.
    """

    def __init__(
        self,
        checkpoint: str,
        version: str = DEFAULT_MODEL_VERSION,
        image_size: int = DEFAULT_IMAGE_SIZE,
        use_fa3: bool = DEFAULT_USE_FA3,
        use_rope_real: bool = DEFAULT_USE_ROPE_REAL,
        compile: bool = DEFAULT_COMPILE,
    ):
        from sam3.model_builder import build_sam3_predictor

        build_kwargs = dict(
            checkpoint_path=checkpoint,
            version=version,
            use_fa3=use_fa3,
            use_rope_real=use_rope_real,
            compile=compile,
            warm_up=compile,
        )
        if version == "sam3.1":
            # image_size 参数化仅 sam3.1 (multiplex) 支持; sam3 原版的
            # Sam3VideoPredictor.__init__ 不接受该参数, 传入会 TypeError
            build_kwargs["image_size"] = image_size
        self._predictor = build_sam3_predictor(**build_kwargs)

        import torch as _torch
        _torch.cuda.empty_cache()

    def _request(self, request: dict) -> dict:
        """Send a synchronous request to the predictor."""
        return self._predictor.handle_request(request=request)

    def _stream(self, request: dict) -> Generator[dict, None, None]:
        """Send a streaming request, yielding responses."""
        for r in self._predictor.handle_stream_request(request=request):
            yield r

    def start_session(
        self,
        resource_path: str,
        offload_video_to_cpu: bool = True,
    ) -> str:
        """Start an inference session for a video/frames directory.

        Parameters
        ----------
        resource_path : str
            Path to a video file or a directory of JPEG frames on disk.
        offload_video_to_cpu : bool
            Keep video frames on CPU to save GPU VRAM. Default True.

        Returns
        -------
        str
            Session ID.
        """
        resp = self._request({
            "type": "start_session",
            "resource_path": resource_path,
            "offload_video_to_cpu": offload_video_to_cpu,
        })
        return resp["session_id"]

    def add_text_prompt(
        self,
        session_id: str,
        frame_index: int,
        text: str,
    ) -> dict:
        """Add a text prompt on a specific frame.

        Returns
        -------
        dict
            Outputs dict with keys like ``out_obj_ids``, ``out_binary_masks``,
            ``out_probs``, ``out_boxes_xywh``.
        """
        resp = self._request({
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": frame_index,
            "text": text,
        })
        return resp.get("outputs", {})

    def propagate(
        self,
        session_id: str,
    ) -> Generator[tuple[int, dict], None, None]:
        """Propagate masks across all frames in the video.

        Yields
        ------
        tuple[int, dict]
            (frame_index, outputs) for each frame.
        """
        for response in self._stream({
            "type": "propagate_in_video",
            "session_id": session_id,
        }):
            yield response["frame_index"], response.get("outputs", {})

    def reset_session(self, session_id: str) -> None:
        """Reset inference state (clear prompt/tracker) but keep frames loaded.

        Used for multi-class inference: reuse one session across classes so
        video frames are loaded only once, instead of re-writing temp jpgs and
        re-loading per class. The backbone feature cache is still recomputed
        per class (SAM3 clears it on reset), but frame disk I/O is saved.
        """
        self._request({"type": "reset_session", "session_id": session_id})

    def close_session(self, session_id: str) -> None:
        """Close an inference session and release resources."""
        try:
            self._request({"type": "close_session", "session_id": session_id})
        except Exception:
            pass


# ─── Frame I/O helpers ──────────────────────────────────────────────────


def get_frames(input_path: str):
    """Load frames from a video file or image directory.

    Returns
    -------
    tuple
        (frames: list[np.ndarray] in RGB HWC format, fps: float | None, count: int)
    """
    p = Path(input_path)
    if p.is_dir():
        files = sorted(
            [f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXT],
            key=lambda x: x.name,
        )
        if not files:
            raise ValueError(f"目录 {input_path} 下没有图片")
        frames = [cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB) for f in files]
        return frames, None, len(frames)
    elif p.is_file():
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            raise ValueError(f"无法打开视频 {input_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames, fps, len(frames)
    else:
        raise ValueError(f"输入路径不存在: {input_path}")


def write_frames_to_temp_dir(frames: list[np.ndarray]) -> str:
    """Write RGB frames to a temporary JPEG directory (SAM3 needs disk paths).

    Returns
    -------
    str
        Path to the temporary directory. Caller is responsible for cleanup.
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="sam3_frames_"))
    for i, fr in enumerate(frames):
        cv2.imwrite(str(tmp_dir / f"{i:08d}.jpg"), cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    return str(tmp_dir)
