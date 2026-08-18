"""Visualization utilities for SAM 3 inference results.

Draws mask overlays, bounding boxes, and object ID labels on video frames.
"""

from __future__ import annotations

import cv2
import numpy as np

# Stable color palette (BGR for OpenCV), 256 entries.
_PALETTE = np.random.default_rng(42).integers(64, 255, size=(256, 3), dtype=np.uint8)


def draw_mask_overlay(
    frame_rgb: np.ndarray,
    obj_ids: list,
    masks: list,
    probs: list | None = None,
    boxes: list | None = None,
    alpha: float = 0.5,
) -> np.ndarray:
    """Draw semi-transparent mask overlays, boxes, and ID labels on a frame.

    Args:
        frame_rgb: Input frame in RGB format (H, W, 3).
        obj_ids: List of object IDs (one per detected object).
        masks: List of binary masks, each (H, W) or resizable to frame shape.
        probs: Optional list of confidence scores.
        boxes: Optional list of [x, y, w, h] bounding boxes.
        alpha: Blending factor for mask overlay (0=invisible, 1=opaque).

    Returns:
        Visualization frame in BGR format (ready for cv2.imwrite).
    """
    vis = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR).copy()
    h, w = vis.shape[:2]
    probs = list(probs) if probs is not None and len(probs) > 0 else []
    boxes = list(boxes) if boxes is not None and len(boxes) > 0 else []

    for i, (oid, m) in enumerate(zip(obj_ids, masks)):
        m = np.asarray(m).astype(bool)
        if m.shape != (h, w):
            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

        color = _PALETTE[int(oid) % 256] if oid is not None else _PALETTE[i % 256]
        vis[m] = (vis[m].astype(np.float32) * (1 - alpha) + color.astype(np.float32) * alpha).astype(np.uint8)

        if i < len(boxes):
            x, y, bw, bh = boxes[i]
            x, y, bw, bh = float(x), float(y), float(bw), float(bh)
            cv2.rectangle(vis, (int(x), int(y)), (int(x + bw), int(y + bh)), color.tolist(), 2)
            label = f"id:{int(oid) if oid is not None else i}"
            if i < len(probs):
                label += f" {float(probs[i]):.2f}"
            cv2.putText(vis, label, (int(x) + 2, int(y) + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return vis
