"""Visualization utilities for SAM 3 inference results.

Draws segmentation mask overlays on video frames: per-class colors,
semi-transparent fill + solid contour, with a corner legend
(no bounding boxes — masks only).
"""

from __future__ import annotations

import cv2
import numpy as np

# 人工挑选的高区分度调色板 (BGR), 按类别顺序循环取用
_DISTINCT_COLORS = [
    (60, 76, 231),    # 红
    (134, 184, 36),   # 绿
    (240, 176, 0),    # 蓝
    (203, 90, 191),   # 品红
    (0, 214, 234),    # 黄
    (113, 122, 245),  # 橙
    (195, 133, 23),   # 深蓝
    (48, 154, 84),    # 深绿
    (153, 72, 169),   # 紫
    (26, 188, 156),   # 青
    (92, 92, 205),    # 砖红
    (200, 200, 60),   # 浅青
]


def build_class_colors(classes: list) -> dict:
    """为本次运行的类别列表生成稳定的 类名→BGR颜色 映射 (按传入顺序)。"""
    return {c: _DISTINCT_COLORS[i % len(_DISTINCT_COLORS)] for i, c in enumerate(classes)}


def draw_mask_overlay(
    frame_rgb: np.ndarray,
    masks: list,
    class_names: list,
    class_colors: dict | None = None,
    alpha: float = 0.7,
) -> np.ndarray:
    """Draw segmentation overlays on a frame: 每类一个颜色, 半透明填充 +
    实心轮廓线, 左上角绘制图例 (只列出当前帧实际出现的类别)。

    Args:
        frame_rgb: Input frame in RGB format (H, W, 3).
        masks: List of binary masks, each (H, W) or resizable to frame shape
            (None 会被跳过).
        class_names: 与 masks 一一对应的类别名.
        class_colors: 类名→BGR 颜色映射; 不传则按 class_names 出现顺序现场分配.
        alpha: 掩码填充的不透明度 (0=不可见, 1=不透明).

    Returns:
        Visualization frame in BGR format (ready for cv2.imwrite).
    """
    vis = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR).copy()
    h, w = vis.shape[:2]

    if class_colors is None:
        class_colors = build_class_colors(list(dict.fromkeys(class_names)))

    kernel = np.ones((3, 3), np.uint8)
    for m, cn in zip(masks, class_names):
        if m is None:
            continue
        m = np.asarray(m).astype(bool)
        if m.shape != (h, w):
            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        if not m.any():
            continue

        color = np.array(class_colors.get(cn, (200, 200, 200)), dtype=np.float32)
        # 半透明填充
        vis[m] = (vis[m].astype(np.float32) * (1 - alpha) + color * alpha).astype(np.uint8)
        # 实心轮廓线 (掩码边缘 = mask & ~erode(mask))
        edge = m & ~cv2.erode(m.astype(np.uint8), kernel).astype(bool)
        vis[edge] = color.astype(np.uint8)

    _draw_legend(vis, class_names, class_colors)
    return vis


def _draw_legend(vis: np.ndarray, class_names: list, class_colors: dict) -> None:
    """左上角图例: 色块 + 类名 (白字描黑边, 任何背景都可读)。"""
    present = [c for c in class_colors if c in set(class_names)]
    if not present:
        return
    y = 12
    for cn in present:
        color = tuple(int(v) for v in class_colors[cn])
        cv2.rectangle(vis, (12, y), (44, y + 20), color, -1)
        cv2.rectangle(vis, (12, y), (44, y + 20), (255, 255, 255), 1)
        cv2.putText(vis, cn, (52, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, cn, (52, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        y += 28
