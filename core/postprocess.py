"""跨类别 mask NMS 后处理.

SAM3 的开放词表架构按每个文本类别独立推理 (每次 prompt reset session),
三类结果直接拼接, 跨类别之间没有竞争。对互斥区域分割任务 (如高速公路
上行/下行/隔离带), 同一区域可能被多个类别的 prompt 同时高分命中。

本模块提供 class-agnostic mask-IoU NMS: 把一帧内所有类别的检测放在一起,
按置信度排序, 用 mask IoU (而非 box IoU) 做贪心抑制。算法与 YOLO-seg
的 NMS 一致, 只是把 box IoU 换成 mask IoU —— 对相邻大面积区域更准确,
不会因 box 必然重叠而误杀真实相邻区域。

不写死任何类别对关系, 全类两两竞争 (等价 YOLO agnostic=True);
不需要互斥的任务关掉开关即可, 不影响后续迁移。
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

# frame_results 里每条检测的元组结构:
#   (cls_name: str, obj_id: int, mask: np.ndarray | None, box: list | None, prob: float)
Detection = Tuple[str, int, "np.ndarray | None", "list | None", float]


def cross_class_mask_nms(
    detections: List[Detection],
    frame_h: int,
    frame_w: int,
    iou_threshold: float = 0.5,
    prob_threshold: float = 0.0,
) -> List[Detection]:
    """对一帧的全部检测做 class-agnostic mask-IoU NMS.

    Args:
        detections: 该帧所有检测的列表, 每条为
            (cls_name, obj_id, mask, box, prob)。
        frame_h, frame_w: 帧尺寸, 用于把各 mask resize 到统一尺寸算 IoU。
        iou_threshold: mask IoU 阈值, >= 该值则抑制低分检测 (默认 0.5)。
        prob_threshold: 预过滤, prob 低于此值的直接丢弃 (默认 0 = 不过滤)。

    Returns:
        抑制后保留的检测列表 (按原始顺序排列, 非按分数排序)。
    """
    if len(detections) <= 1:
        return list(detections)

    # ── 1. 预过滤: 丢弃无 mask 或 prob 过低的检测 ──
    valid_idx: List[int] = []
    masks: List[np.ndarray] = []
    probs: List[float] = []
    for i, (_, _, m, _, pr) in enumerate(detections):
        if m is None:
            continue
        if pr < prob_threshold:
            continue
        m_arr = np.asarray(m).astype(bool)
        if m_arr.shape != (frame_h, frame_w):
            m_arr = cv2.resize(
                m_arr.astype(np.uint8), (frame_w, frame_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        if not m_arr.any():
            continue
        valid_idx.append(i)
        masks.append(m_arr)
        probs.append(float(pr))

    if len(valid_idx) <= 1:
        return [detections[i] for i in valid_idx]

    # ── 2. 按置信度降序排序 (NMS 标准做法: 高分优先保留) ──
    order = np.argsort(probs)[::-1]  # 高 → 低
    masks_stacked = np.stack(masks, axis=0)  # (N, H, W) bool

    # ── 3. 贪心 NMS: 逐个保留最高分, 抑制与之 IoU >= 阈值的低分检测 ──
    suppressed = np.zeros(len(order), dtype=bool)
    keep_pos: List[int] = []
    for rank, pos in enumerate(order):
        if suppressed[rank]:
            continue
        keep_pos.append(pos)
        m_keep = masks_stacked[pos]
        area_keep = m_keep.sum()
        if area_keep == 0:
            continue
        # 与后续 (更低分) 检测算 mask IoU
        for rank2 in range(rank + 1, len(order)):
            if suppressed[rank2]:
                continue
            pos2 = order[rank2]
            m_other = masks_stacked[pos2]
            inter = np.logical_and(m_keep, m_other).sum()
            if inter == 0:
                continue
            area_other = m_other.sum()
            union = area_keep + area_other - inter
            iou = inter / union if union > 0 else 0.0
            if iou >= iou_threshold:
                suppressed[rank2] = True

    # ── 4. 返回保留的检测, 保持原始输入顺序 ──
    kept_indices = sorted(valid_idx[pos] for pos in keep_pos)
    return [detections[i] for i in kept_indices]
