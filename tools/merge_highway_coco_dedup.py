#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多子集 COCO 合并 + 内容级去重/去泄漏脚本
============================================================
背景: configs/datasets/highway_road_multi.yaml 的 ConcatSam3Datasets 拼接链路
有两个结构性问题:
  1. 各子集 COCO image id 都从 1 开始, 拼接后 id 冲突; 而评测 GT (gt_path)
     只覆盖第一个子集 → val mAP 是把别的子集预测对到错误 GT 上算出的噪声,
     早停 (coco_eval_bbox_AP) 不可信;
  2. 子集间本是同批源数据的不同划分, 拼接后存在内容级 train/valid 泄漏
     (md5 全等图 127 张) 与 train 内部重复 (305 组)。

本脚本把所有子集的 train 合并为一个 COCO、valid 合并为另一个 COCO:
  - image/annotation id 全部重映射为全局唯一 (从 1 连续编号);
  - 图片按 md5 内容去重: valid 内部去重 → train 剔除命中 valid 的图 (去泄漏)
    → train 内部去重 (同内容保留首次出现, 含其标注);
  - 退化标注过滤: 多边形栅格化面积为 0 的标注直接丢弃 (细碎标注,
    否则训练时 DecodeRle 反复告警 "empty mask found");
  - 图片不落盘复制, 在输出目录建软链接 (Linux), 文件名加子集前缀保证唯一;
  - 输出即 Roboflow COCO 布局, 配单数据集 yaml (path/train/val/ann_file) 直接训练,
    评测 GT 与 val 数据集同源同 id, 指标/早停恢复可信。

用法: python tools/merge_highway_coco_dedup.py
"""

import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from pycocotools import mask as mask_utils

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ============================================================
# 配置 (批处理模式: 在开头配置输入/输出路径)
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# 输入: 多数据集配置 (读取其 datasets: 列表, 与训练配置保持同源, 不重复维护路径)
INPUT_CONFIG = PROJECT_ROOT / "configs/datasets/highway_road_multi.yaml"
# 输出: 合并后的数据集根目录 (下建 {train,valid}/_annotations.coco.json + 软链接图片)
OUTPUT_ROOT = Path("/data2/kaiyun/datasets_seg/highway_road_merged_coco")

# 类别规范 (所有子集必须一致, 不一致直接报错; id 从 1 开始与源数据一致)
CANONICAL_CATEGORIES = [
    {"id": 1, "name": "highway up direction", "supercategory": "road"},
    {"id": 2, "name": "highway down direction", "supercategory": "road"},
    {"id": 3, "name": "road isolation zone", "supercategory": "road"},
]

COPY_FILES = False  # False=软链接 (Linux 推荐, 不占额外空间); True=复制文件


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sanitize(name: str) -> str:
    """子集名 → 文件名安全前缀 (保留中文, 只去掉路径分隔符与空格)。"""
    return name.replace("/", "_").replace(" ", "_")


def link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if COPY_FILES:
        import shutil

        shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)


def iter_bar(iterable, desc):
    if tqdm is not None:
        return tqdm(iterable, desc=desc, ncols=100)
    print(desc)
    return iterable


def collect_split(entries, split_key: str):
    """读取所有子集某划分 (train/valid) 的 (图片, 标注)。

    返回 items: [{ds, img_id, file_name, width, height, abs_path, anns}], 并校验类别一致性。
    """
    items = []
    canon_names = {c["id"]: c["name"] for c in CANONICAL_CATEGORIES}
    for ds_idx, e in enumerate(entries):
        root = Path(e["path"])
        sub = e[split_key]
        ann_path = root / sub / e.get("ann_file", "_annotations.coco.json")
        if not ann_path.exists():
            print(f"[警告] 标注不存在, 跳过: {ann_path}")
            continue
        with open(ann_path, encoding="utf-8") as f:
            coco = json.load(f)
        ds_names = {c["id"]: c["name"] for c in coco["categories"]}
        if ds_names != canon_names:
            raise ValueError(
                f"类别不一致: {e['path']} 的 {ds_names} != 规范 {canon_names}"
            )
        anns_by_img = defaultdict(list)
        for a in coco["annotations"]:
            anns_by_img[a["image_id"]].append(a)
        img_dir = root / sub
        for img in coco["images"]:
            items.append(
                {
                    "ds": f"{ds_idx:02d}_{sanitize(Path(e['path']).name)}",
                    "img_id": img["id"],
                    "file_name": img["file_name"],
                    "width": img["width"],
                    "height": img["height"],
                    "abs_path": img_dir / img["file_name"],
                    "anns": anns_by_img.get(img["id"], []),
                }
            )
    return items


def dedup_items(items, banned_md5: set, stats: Counter, split: str):
    """按 md5 去重; banned_md5 命中即丢弃 (去泄漏)。返回 (kept, kept_md5)。"""
    kept, seen = [], set()
    for it in iter_bar(items, f"[{split}] 计算 md5 并去重"):
        if not it["abs_path"].exists():
            stats[f"{split}_missing"] += 1
            continue
        h = md5_of(it["abs_path"])
        if h in banned_md5:
            stats[f"{split}_leak_dropped"] += 1
            continue
        if h in seen:
            stats[f"{split}_dup_dropped"] += 1
            continue
        seen.add(h)
        it["md5"] = h
        kept.append(it)
    return kept, seen


def _raster_area(ann, height: int, width: int) -> float:
    """标注的栅格化面积 (polygon/RLE 均可)。"""
    seg = ann.get("segmentation")
    if isinstance(seg, dict):
        return float(np.asarray(mask_utils.area(seg), dtype=float).sum())
    polys = seg if seg and isinstance(seg[0], list) else [seg]
    if not polys or max(len(p) for p in polys) < 6:
        return 0.0
    rles = mask_utils.frPyObjects(polys, height, width)
    rle = mask_utils.merge(rles) if len(rles) > 1 else rles
    return float(np.asarray(mask_utils.area(rle), dtype=float).sum())


def drop_empty_mask_anns(items, stats: Counter, split: str):
    """丢弃栅格化面积为 0 的退化标注 (细碎到 1~2 像素, 栅格化后为空)。"""
    n = 0
    for it in iter_bar(items, f"[{split}] 过滤空 mask 标注"):
        kept = [a for a in it["anns"] if _raster_area(a, it["height"], it["width"]) > 0]
        n += len(it["anns"]) - len(kept)
        it["anns"] = kept
    stats[f"{split}_empty_mask_dropped"] = n


def write_coco(kept, split: str, out_dir: Path):
    """重映射 id + 建软链接 + 写 _annotations.coco.json。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    ann_id = 1
    per_ds = Counter()
    for new_id, it in enumerate(iter_bar(kept, f"[{split}] 写软链接与标注"), start=1):
        link_name = f"{it['ds']}__{os.path.basename(it['file_name'])}"
        link_or_copy(it["abs_path"], out_dir / link_name)
        images.append(
            {
                "id": new_id,
                "file_name": link_name,
                "width": it["width"],
                "height": it["height"],
            }
        )
        for a in it["anns"]:
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": new_id,
                    "category_id": a["category_id"],
                    "bbox": a["bbox"],
                    "area": a["area"],
                    "segmentation": a["segmentation"],
                    "iscrowd": a.get("iscrowd", 0),
                }
            )
            ann_id += 1
        per_ds[it["ds"]] += 1
    coco = {
        "info": {
            "description": "highway_road merged COCO (去重/去泄漏), 由 "
            "tools/merge_highway_coco_dedup.py 生成",
        },
        "categories": CANONICAL_CATEGORIES,
        "images": images,
        "annotations": annotations,
    }
    ann_out = out_dir / "_annotations.coco.json"
    with open(ann_out, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False)
    return ann_out, per_ds


def main():
    with open(INPUT_CONFIG, encoding="utf-8") as f:
        entries = (yaml.safe_load(f) or {}).get("datasets") or []
    if not entries:
        raise SystemExit(f"未从 {INPUT_CONFIG} 读到 datasets: 列表")
    for e in entries:
        if float(e.get("limit_ratio", 1.0)) < 1.0:
            print(f"[警告] {e['path']} limit_ratio<1, 合并脚本忽略该字段 (按全量合并)")
    print(f"共 {len(entries)} 个子数据集, 输出到 {OUTPUT_ROOT}")

    stats = Counter()
    # 1) valid 先合并去重, 得到 valid 内容指纹集合
    val_items = collect_split(entries, "val")
    stats["val_raw"] = len(val_items)
    val_kept, val_md5 = dedup_items(val_items, set(), stats, "valid")
    drop_empty_mask_anns(val_kept, stats, "valid")
    # 2) train 剔除命中 valid 的内容 (去泄漏), 再内部去重
    train_items = collect_split(entries, "train")
    stats["train_raw"] = len(train_items)
    train_kept, _ = dedup_items(train_items, val_md5, stats, "train")
    drop_empty_mask_anns(train_kept, stats, "train")

    # 3) 写输出
    train_ann, train_per_ds = write_coco(train_kept, "train", OUTPUT_ROOT / "train")
    val_ann, val_per_ds = write_coco(val_kept, "valid", OUTPUT_ROOT / "valid")

    # 4) 报告
    report = {
        "input_config": str(INPUT_CONFIG),
        "output_root": str(OUTPUT_ROOT),
        "stats": dict(stats),
        "train_kept": len(train_kept),
        "val_kept": len(val_kept),
        "train_per_subset": dict(train_per_ds),
        "val_per_subset": dict(val_per_ds),
    }
    with open(OUTPUT_ROOT / "merge_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n================ 合并完成 ================")
    print(f"train: {stats['train_raw']} → {len(train_kept)} 张 "
          f"(去泄漏 {stats['train_leak_dropped']}, 去重 {stats['train_dup_dropped']}, "
          f"缺失 {stats['train_missing']}, 空 mask 标注 {stats['train_empty_mask_dropped']}) → {train_ann}")
    print(f"valid: {stats['val_raw']} → {len(val_kept)} 张 "
          f"(去重 {stats['valid_dup_dropped']}, 缺失 {stats['valid_missing']}, "
          f"空 mask 标注 {stats['valid_empty_mask_dropped']}) → {val_ann}")
    print(f"报告: {OUTPUT_ROOT / 'merge_report.json'}")


if __name__ == "__main__":
    main()
