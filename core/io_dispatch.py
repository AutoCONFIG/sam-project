"""Input discovery and output tree mirroring for prediction.

Scans an input path into processing units (each a video file or an image
sequence), then maps each unit to a mirrored output directory tree so that
the output structure mirrors the input: a video in yields a video out, an
image directory yields a parallel image directory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from core.labels import VIDEO_EXT
from utils.constants import IMAGE_EXT


@dataclass
class InputUnit:
    """One processable input: a single video or an image sequence."""

    kind: str  # "video" | "image_seq"
    # for video: the source file; for image_seq: the directory of images
    source: Path
    # relative path under the input root (for mirroring output structure)
    rel_dir: str  # "" means top level
    # only for video: original filename stem (e.g. "bedroom")
    stem: str = ""
    # only for image_seq: ordered list of image paths
    frames: List[Path] = field(default_factory=list)


@dataclass
class OutputTree:
    """Mirrored output directories for one input unit."""

    unit: InputUnit
    base: Path  # <output_root>/<rel_dir>
    vis: Path  # <base>/vis
    masks: Path  # <base>/masks
    npz: Path  # <base>/masks_npz
    labels: Path  # <base>/labels
    # output filename stem (video name or image-seq folder name)
    stem: str


def discover_inputs(input_path: str) -> List[InputUnit]:
    """Scan an input path into one or more processing units.

    - A single video file → one ``video`` unit.
    - A directory → recursive scan: videos become ``video`` units, and each
      subdirectory containing images becomes one ``image_seq`` unit. A
      directory may yield a mix of both.
    """
    p = Path(input_path)
    if not p.exists():
        raise ValueError(f"输入路径不存在: {input_path}")

    if p.is_file():
        if p.suffix.lower() in VIDEO_EXT:
            return [InputUnit(kind="video", source=p, rel_dir="", stem=p.stem)]
        if p.suffix.lower() in IMAGE_EXT:
            # single image → treat as a one-frame image sequence
            return [InputUnit(kind="image_seq", source=p.parent, rel_dir="",
                              frames=[p])]
        raise ValueError(f"不支持的输入文件类型: {p.suffix}")

    # directory → recursive scan
    return _scan_directory(p, rel_root=p)


def _scan_directory(dir_path: Path, rel_root: Path) -> List[InputUnit]:
    units: List[InputUnit] = []

    # videos directly in this directory
    for entry in sorted(dir_path.iterdir()):
        if entry.is_file() and entry.suffix.lower() in VIDEO_EXT:
            rel = _rel_dir(entry.parent, rel_root)
            units.append(InputUnit(kind="video", source=entry, rel_dir=rel,
                                   stem=entry.stem))

    # images directly in this directory → one image_seq unit
    imgs = sorted(
        [f for f in dir_path.iterdir()
         if f.is_file() and f.suffix.lower() in IMAGE_EXT],
        key=lambda x: x.name,
    )
    if imgs:
        rel = _rel_dir(dir_path, rel_root)
        units.append(InputUnit(kind="image_seq", source=dir_path, rel_dir=rel,
                               frames=imgs))

    # recurse into subdirectories
    for sub in sorted(dir_path.iterdir()):
        if sub.is_dir():
            units.extend(_scan_directory(sub, rel_root))

    return units


def _rel_dir(path: Path, root: Path) -> str:
    """Relative directory string from root to path ("" if same)."""
    try:
        rel = path.relative_to(root)
    except ValueError:
        return ""
    return rel.as_posix()


def resolve_output_tree(output_root: str, unit: InputUnit) -> OutputTree:
    """Build the mirrored output directory tree for one input unit.

    Layout under ``<output_root>/<rel_dir>/``:
        vis/       — visualization (mp4 for video, jpg for image_seq)
        masks/     — mask images (mp4 for video, png for image_seq)
        masks_npz/ — raw mask data (npz per frame, always)
        labels/    — coco.json + yolo/ for this unit
    """
    root = Path(output_root)
    base = root / unit.rel_dir if unit.rel_dir else root
    stem = unit.stem if unit.kind == "video" else unit.source.name

    tree = OutputTree(
        unit=unit,
        base=base,
        vis=base / "vis",
        masks=base / "masks",
        npz=base / "masks_npz",
        labels=base / "labels",
        stem=stem,
    )
    for d in (tree.vis, tree.masks, tree.npz, tree.labels):
        d.mkdir(parents=True, exist_ok=True)
    return tree
