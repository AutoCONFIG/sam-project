"""Label export for SAM 3 inference results.

Converts per-frame SAM 3 outputs (object ids, binary masks, boxes, scores)
into COCO and YOLO annotation formats. Both detection (bbox) and instance
segmentation (polygon) annotations are produced.

Class names come from the text prompt (open-vocabulary); class ids are
assigned in order of first appearance, or from an explicit class list.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}


@dataclass
class FrameAnnotations:
    """Collected annotations for a single frame."""

    frame_idx: int
    image_name: str  # filename used in COCO images[] / YOLO txt name
    width: int
    height: int
    # each entry: (obj_id, class_id, polygon_xy normalized, bbox_xywh abs, score)
    objects: List[Tuple[int, int, List[Tuple[float, float]], List[float], float]] = field(
        default_factory=list
    )


class LabelExporter:
    """Accumulate per-frame detections and write COCO + YOLO label files.

    Supports both single-class and multi-class workflows. Each ``add_frame``
    call registers the objects detected by one prompt under a class name;
    class ids are assigned in order of first appearance, or from an explicit
    class list.

    Parameters
    ----------
    class_name : str
        Default class name used when ``add_frame`` is called without an
        explicit ``class_name`` (single-prompt workflow).
    predefined_classes : list[str], optional
        If given, class ids are looked up by name; unknown names raise
        ValueError. If None, classes are auto-numbered on first appearance.
    """

    def __init__(
        self,
        class_name: str,
        predefined_classes: Optional[List[str]] = None,
    ):
        self.class_name = class_name
        self._predefined = predefined_classes
        # class name -> id mapping (auto-numbered or from predefined list)
        self._class_to_id: Dict[str, int] = {}
        if predefined_classes:
            for i, name in enumerate(predefined_classes):
                self._class_to_id[name] = i
        else:
            # seed with the active class
            self._class_to_id[class_name] = 0

        self._frames: List[FrameAnnotations] = []

    @property
    def class_id(self) -> int:
        """Id of the default (prompt) class."""
        return self._class_to_id[self.class_name]

    @property
    def categories(self) -> List[Dict[str, Any]]:
        """COCO categories list, sorted by id."""
        names = sorted(self._class_to_id.items(), key=lambda kv: kv[1])
        return [{"id": cid, "name": name} for name, cid in names]

    def _resolve_class_id(self, name: str) -> int:
        if name not in self._class_to_id:
            if self._predefined is not None:
                raise ValueError(
                    f"类别 '{name}' 不在预定义类别列表 {self._predefined} 中"
                )
            self._class_to_id[name] = len(self._class_to_id)
        return self._class_to_id[name]

    def add_frame(
        self,
        frame_idx: int,
        image_name: str,
        height: int,
        width: int,
        obj_ids: List,
        masks: List,
        boxes: List,
        probs: List,
        class_name: Optional[str] = None,
    ) -> None:
        """Register one frame's detections.

        Parameters
        ----------
        frame_idx : int
            Index of the frame in the sequence (0-based).
        image_name : str
            Filename for this frame in COCO/YOLO (e.g. "000012.jpg").
        height, width : int
            Frame dimensions in pixels.
        obj_ids : list
            Per-object ids from SAM 3 (``out_obj_ids``).
        masks : list
            Per-object binary masks (``out_binary_masks``).
        boxes : list
            Per-object boxes [x, y, w, h] in absolute pixels.
        probs : list
            Per-object confidence scores (``out_probs``).
        class_name : str, optional
            Class name to assign these objects to. Defaults to the exporter's
            ``class_name`` (single-prompt workflow). In multi-class mode each
            prompt's results are registered under its own class name.
        """
        cname = class_name if class_name is not None else self.class_name
        # If this frame was already added by a previous prompt (multi-class
        # per-class mode), append to the existing FrameAnnotations; otherwise
        # create a new one.
        fa = None
        for existing in self._frames:
            if existing.frame_idx == frame_idx:
                fa = existing
                break
        if fa is None:
            fa = FrameAnnotations(
                frame_idx=frame_idx,
                image_name=image_name,
                width=width,
                height=height,
            )
            self._frames.append(fa)
        cid = self._resolve_class_id(cname)

        for i, (oid, m) in enumerate(zip(obj_ids, masks)):
            m_arr = np.asarray(m).astype(bool)
            if m_arr.shape != (height, width):
                m_arr = cv2.resize(
                    m_arr.astype(np.uint8), (width, height),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)

            box = boxes[i] if i < len(boxes) else None
            if box is not None:
                # SAM3 returns out_boxes_xywh as normalized [0,1] coordinates;
                # convert to absolute pixels for consistent internal storage.
                bx, by, bw, bh = (float(v) for v in box)
                if 0.0 <= bx <= 1.0 and 0.0 <= by <= 1.0 and bw <= 1.0 and bh <= 1.0:
                    box = [bx * width, by * height, bw * width, bh * height]
                else:
                    box = [bx, by, bw, bh]
            else:
                # derive bbox from mask if SAM3 didn't supply one (absolute px)
                ys, xs = np.where(m_arr)
                if len(xs) == 0:
                    continue
                box = [float(xs.min()), float(ys.min()),
                       float(xs.max() - xs.min()), float(ys.max() - ys.min())]

            score = float(probs[i]) if i < len(probs) else 0.0
            polygon = _mask_to_polygon(m_arr, height, width)
            fa.objects.append((int(oid) if oid is not None else i, cid,
                               polygon, [float(v) for v in box], score))

    # ─── COCO ────────────────────────────────────────────────────────────

    def write_coco(self, path: str | Path) -> None:
        """Write a COCO instance-segmentation JSON.

        Produces ``images[]``, ``annotations[]`` (each with segmentation
        polygon + bbox + area), and ``categories[]``.
        """
        path = Path(path)
        images: List[Dict[str, Any]] = []
        annotations: List[Dict[str, Any]] = []
        ann_id = 1

        for fa in self._frames:
            image_id = fa.frame_idx + 1
            images.append({
                "id": image_id,
                "file_name": fa.image_name,
                "width": fa.width,
                "height": fa.height,
            })
            for oid, cid, polygon, box, score in fa.objects:
                seg_poly = [[float(x), float(y)] for x, y in polygon]
                if not seg_poly:
                    continue
                x, y, w, h = box
                area = float(_polygon_area(polygon, fa.width, fa.height))
                annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cid,
                    "segmentation": [seg_poly],
                    "bbox": [float(x), float(y), float(w), float(h)],
                    "area": area,
                    "iscrowd": 0,
                    "score": score,
                    "attributes": {"object_id": oid},
                })
                ann_id += 1

        coco = {
            "images": images,
            "annotations": annotations,
            "categories": self.categories,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(coco, f, ensure_ascii=False, indent=2)

    # ─── YOLO ────────────────────────────────────────────────────────────

    def write_yolo(self, out_dir: str | Path) -> None:
        """Write YOLO detection + segmentation txt files.

        - ``det/<image_stem>.txt`` — ``cls cx cy w h`` (normalized)
        - ``seg/<image_stem>.txt`` — ``cls x1 y1 x2 y2 ...`` (normalized polygon)
        """
        out_dir = Path(out_dir)
        det_dir = out_dir / "det"
        seg_dir = out_dir / "seg"
        det_dir.mkdir(parents=True, exist_ok=True)
        seg_dir.mkdir(parents=True, exist_ok=True)

        for fa in self._frames:
            stem = Path(fa.image_name).stem
            det_lines: List[str] = []
            seg_lines: List[str] = []
            for oid, cid, polygon, box, score in fa.objects:
                x, y, w, h = box
                cx = (x + w / 2) / fa.width
                cy = (y + h / 2) / fa.height
                det_lines.append(
                    f"{cid} {cx:.6f} {cy:.6f} {w / fa.width:.6f} {h / fa.height:.6f}"
                )
                if polygon:
                    coords = " ".join(
                        f"{px / fa.width:.6f} {py / fa.height:.6f}"
                        for px, py in polygon
                    )
                    seg_lines.append(f"{cid} {coords}")

            (det_dir / f"{stem}.txt").write_text("\n".join(det_lines))
            (seg_dir / f"{stem}.txt").write_text("\n".join(seg_lines))

        # also write the classes file for convenience
        (out_dir / "classes.txt").write_text(
            "\n".join(name for name, _ in
                      sorted(self._class_to_id.items(), key=lambda kv: kv[1]))
        )

    # ─── class names file ────────────────────────────────────────────────

    def write_classes(self, path: str | Path) -> None:
        """Write the class-name list (one name per line, id = line index)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        names = [name for name, _ in
                 sorted(self._class_to_id.items(), key=lambda kv: kv[1])]
        path.write_text("\n".join(names))


# ─── Helpers ─────────────────────────────────────────────────────────────


def _mask_to_polygon(mask: np.ndarray, height: int, width: int) -> List[Tuple[float, float]]:
    """Extract the largest polygon contour from a binary mask.

    Returns a list of (x, y) pixel coordinates. Empty if no contour found.
    """
    mask_u8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    # keep the largest contour by area (drop tiny specks)
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 1.0:
        return []
    # simplify slightly to avoid huge point lists
    eps = 0.005 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, eps, True)
    return [(float(p[0][0]), float(p[0][1])) for p in approx]


def _polygon_area(polygon: List[Tuple[float, float]], width: int, height: int) -> float:
    """Approximate polygon area via the shoelace formula (pixels²)."""
    if len(polygon) < 3:
        return 0.0
    pts = np.array(polygon, dtype=np.float64)
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
