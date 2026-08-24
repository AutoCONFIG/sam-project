#!/usr/bin/env python3
# YOLO 分割数据集 → COCO 格式转换工具 (SAM 3 训练用)
# ==================================================
# 把 YOLO 分割标注 (class_id + 归一化多边形) 转为本项目训练所需的
# Roboflow 风格 COCO 布局:
#
#   <out>/                         # 输出根目录
#     <子数据集名>/                # 每个含 images/ 的子目录自成一组
#       train/_annotations.coco.json + 图片
#       valid/_annotations.coco.json + 图片
#     batch_summary.json           # 各子数据集统计汇总
#
# 默认行为: 改下面 INPUT_ROOT / OUTPUT_ROOT 两个常量, 然后
#   python tools/yolo_seg_to_coco.py
# 即可自动发现根目录下所有含 images/ 的子数据集并逐一转换。
#
#   - 有 train/images + val/images 的子集 -> 沿用现有划分
#   - 只有 images/ 的子集 -> 按 VAL_SPLIT (默认 0.1) 自动划分
#   - classes.txt 取 INPUT_ROOT/classes.txt, 全局共享
#
# 命令行参数 (可选, 传了就覆盖开头常量):
#   --input-root / --output-root / --val-split / --link
#   --root / --out / --train-list / --val-list / --images-dir / --classes
#       (老的单 split 用法, 传 --root 走原单数据集流程)
#
# 标签文件查找顺序: <图片同名>_seg.txt -> <图片同名>.txt (标准 YOLO)
# 找不到标签的图片按无标注背景图处理 (计入统计, 作为负样本保留)。
#
# 依赖: 仅 stdlib + Pillow。

import argparse
import hashlib
import json
import os
import random
import shutil
import sys

from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ============================================================
# 批处理配置 (无需命令行参数时直接改这里)
#   INPUT_ROOT  : 数据集根目录 (含 classes.txt 和各子数据集)
#   OUTPUT_ROOT : COCO 输出根目录 (其下按子数据集名建 train/ valid/)
#   VAL_SPLIT   : 仅对无 train/val 划分的数据集生效，默认 0.1
#   LINK_MODE   : 硬链接(默认, 跨文件系统自动退化为复制) / 软链接 / 复制
# ============================================================
INPUT_ROOT = "/data2/kaiyun/datasets_seg/yolo_format_merged"
OUTPUT_ROOT = "/data2/kaiyun/datasets_seg/yolo_format_coco"
VAL_SPLIT = 0.1
LINK_MODE = "hard"


# ---------------------------------------------------------------------------
# 标签/类别 解析
# ---------------------------------------------------------------------------

def load_classes_from_file(path):
    """读类别名 (0 号 = 第一行)，返回 1-based COCO categories 列表。"""
    if not os.path.isfile(path):
        sys.exit(f"[错误] 类别文件不存在: {path} (用 --classes 指定)")
    with open(path, encoding="utf-8") as f:
        names = [l.strip() for l in f if l.strip()]
    if not names:
        sys.exit(f"[错误] 类别文件为空: {path}")
    print(f"共 {len(names)} 个类别 (来自 {path}):")
    for i, name in enumerate(names):
        print(f"  [{i}] {name!r}")
    return [{"id": i + 1, "name": n, "supercategory": "none"} for i, n in enumerate(names)]


def find_label(img_path):
    """<同名>_seg.txt 优先, 回退标准 YOLO 的 <同名>.txt。"""
    stem = os.path.splitext(img_path)[0]
    for suffix in ("_seg.txt", ".txt"):
        cand = stem + suffix
        if os.path.isfile(cand):
            return cand
    return None


def parse_label(label_path, width, height):
    """解析 YOLO 分割标注 → (annotations 草稿, 统计计数)。

    每行: class_id x1 y1 x2 y2 ... (归一化多边形)
    """
    anns = []
    stats = {"bad_lines": 0, "clipped": 0, "skipped_class": 0}
    with open(label_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                cls_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]
            except (ValueError, IndexError):
                stats["bad_lines"] += 1
                continue
            if len(coords) < 6 or len(coords) % 2 != 0:
                stats["bad_lines"] += 1
                continue
            if any(c < 0.0 or c > 1.0 for c in coords):
                stats["clipped"] += 1
                coords = [min(1.0, max(0.0, c)) for c in coords]
            xs = [round(coords[i] * width, 2) for i in range(0, len(coords), 2)]
            ys = [round(coords[i] * height, 2) for i in range(1, len(coords), 2)]
            poly = [v for pair in zip(xs, ys) for v in pair]
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
            # 多边形面积 (shoelace); 退化多边形面积为 0, 后端加载会按 bbox 重算
            area = abs(sum(xs[i] * ys[(i + 1) % len(ys)]
                           - xs[(i + 1) % len(xs)] * ys[i]
                           for i in range(len(xs)))) / 2.0
            anns.append({
                "cls_id": cls_id,  # 0-based, 转 category_id 时 +1
                "bbox": [round(x0, 2), round(y0, 2),
                         round(x1 - x0, 2), round(y1 - y0, 2)],
                "area": round(area, 2),
                "segmentation": [poly],
            })
    return anns, stats


# ---------------------------------------------------------------------------
# 文件落盘
# ---------------------------------------------------------------------------

def link_or_copy(src, dst, mode):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return
    if mode == "symlink":
        os.symlink(os.path.abspath(src), dst)
    elif mode == "copy" or mode == "hard":
        try:
            if mode == "hard":
                os.link(src, dst)
            else:
                raise OSError("force copy")
        except OSError:
            shutil.copy2(src, dst)  # 跨文件系统时硬链接退化为复制


# ---------------------------------------------------------------------------
# 核心: 把一组图片路径转成一个 split 的 COCO (不依赖 argparse)
# ---------------------------------------------------------------------------

def convert_split(split, img_paths, root, out, categories, link_mode, dry_run=False, limit=None):
    """处理一个 split: 解析标注、落盘图片、生成 COCO dict。返回 (coco, stats)。

    root       : 该子数据集的基准目录 (用于计算 file_name 相对路径)
    out        : 该 split 的输出目录 (图片落此, JSON 落此)
    link_mode  : hard / symlink / copy
    """
    coco = {"images": [], "annotations": [], "categories": categories}
    stats = {"no_label": 0, "not_found": 0, "bad_lines": 0,
             "clipped": 0, "skipped_class": 0, "linked": 0, "outside_root": 0}
    ann_id = 1
    n_classes = len(categories)
    paths = img_paths[:limit] if limit else img_paths

    for img_id, src in enumerate(paths, start=1):
        if not os.path.isfile(src):
            stats["not_found"] += 1
            print(f"[警告] 图片找不到, 跳过: {src}", file=sys.stderr)
            continue
        rel = os.path.relpath(src, root)
        if rel.startswith(".." + os.sep) or rel == "..":
            # 源图在 root 之外: 用 路径哈希+原名 扁平化, 防止写出输出目录
            digest = hashlib.md5(os.path.abspath(src).encode()).hexdigest()[:8]
            rel = f"{digest}_{os.path.basename(src)}"
            stats["outside_root"] += 1
        with Image.open(src) as im:
            width, height = im.size

        label_path = find_label(src)
        if label_path is None:
            stats["no_label"] += 1  # 无标注 = 背景图, 作为负样本保留
            anns = []
        else:
            anns, lst = parse_label(label_path, width, height)
            for k in ("bad_lines", "clipped", "skipped_class"):
                stats[k] += lst[k]

        if not dry_run:
            dst = os.path.join(out, rel)
            link_or_copy(src, dst, link_mode)
            stats["linked"] += 1

        coco["images"].append({
            "id": img_id, "file_name": rel, "width": width, "height": height,
        })
        for a in anns:
            if not (0 <= a["cls_id"] < n_classes):
                stats["skipped_class"] += 1
                continue
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": a["cls_id"] + 1,
                "bbox": a["bbox"],
                "area": a["area"],
                "segmentation": a["segmentation"],
                "iscrowd": 0,
            })
            ann_id += 1

    if not dry_run:
        ann_path = os.path.join(out, "_annotations.coco.json")
        os.makedirs(os.path.dirname(ann_path) or ".", exist_ok=True)
        with open(ann_path, "w", encoding="utf-8") as f:
            json.dump(coco, f, ensure_ascii=False)

    print(f"  [{split}] 图片 {len(coco['images'])} 张, 标注 {len(coco['annotations'])} 条"
          f" | 无标注背景图 {stats['no_label']}, 找不到图片 {stats['not_found']},"
          f" 坏行 {stats['bad_lines']}, 越界已截断 {stats['clipped']},"
          f" 类别越界丢弃 {stats['skipped_class']}, root外扁平化 {stats['outside_root']}")
    return coco, stats


# ---------------------------------------------------------------------------
# 子数据集发现 + 划分
# ---------------------------------------------------------------------------

def list_images_with_labels(images_dir):
    """递归扫描目录下带标签的图片 (绝对路径), 用于无 train/val 划分时自动切分。"""
    imgs = []
    for dirpath, _, files in os.walk(images_dir):
        for fn in sorted(files):
            if os.path.splitext(fn)[1].lower() in IMG_EXTS:
                full = os.path.join(dirpath, fn)
                if find_label(full):
                    imgs.append(full)
    return imgs


def discover_subdatasets(input_root):
    """扫描 input_root 下一级含 images/ 的子目录, 返回 [(子数据集名, 划分描述)]。

    划分描述为 ('split', train_dir, val_dir) 或 ('auto', images_dir, None)。
    同时把根目录本身若含 images/ 也算一个 (名为 input_root 的 basename)。
    """
    results = []
    candidates = []
    # 一级子目录
    for name in sorted(os.listdir(input_root)):
        sub = os.path.join(input_root, name)
        if os.path.isdir(sub):
            candidates.append((name, sub))

    for name, sub in candidates:
        # 优先: 有 train/images + val/images -> 沿用现有划分
        train_dir = os.path.join(sub, "train", "images")
        val_dir = os.path.join(sub, "val", "images")
        if os.path.isdir(train_dir) and os.path.isdir(val_dir):
            results.append((name, "split", sub, train_dir, val_dir))
            continue
        # 仅有 images/ -> 自动划分
        images_dir = os.path.join(sub, "images")
        if os.path.isdir(images_dir):
            results.append((name, "auto", sub, images_dir, None))
            continue
        # 淄博历史布局: all_汇总/{train,val}/images
        all_dir = os.path.join(sub, "all_汇总")
        if os.path.isdir(all_dir):
            t2 = os.path.join(all_dir, "train", "images")
            v2 = os.path.join(all_dir, "val", "images")
            if os.path.isdir(t2) and os.path.isdir(v2):
                results.append((name, "split", sub, t2, v2))
                continue
    return results


def collect_images_from_dir(images_dir):
    """递归收集 images_dir 下所有图片 (带不带标签都收, 不带标签的转成背景图)。"""
    imgs = []
    for dirpath, _, files in os.walk(images_dir):
        for fn in sorted(files):
            if os.path.splitext(fn)[1].lower() in IMG_EXTS:
                imgs.append(os.path.join(dirpath, fn))
    return imgs


def auto_split(imgs, val_split, seed=42):
    """按 val_split 随机划分; val_split=0 时全部进 train。"""
    rng = random.Random(seed)
    imgs = list(imgs)
    rng.shuffle(imgs)
    n_val = int(round(len(imgs) * val_split))
    return imgs[n_val:], imgs[:n_val]


# ---------------------------------------------------------------------------
# 批处理主流程
# ---------------------------------------------------------------------------

def batch_convert(input_root, output_root, classes_file, val_split, link_mode,
                  dry_run=False, limit=None):
    """自动发现子数据集并逐一转换为 COCO, 写 batch_summary.json。"""
    input_root = os.path.abspath(input_root)
    output_root = os.path.abspath(output_root)
    if output_root == input_root or output_root.startswith(input_root + os.sep):
        sys.exit("[错误] 输出目录不能是 INPUT_ROOT 或其子目录 (避免污染源数据集)")

    categories = load_classes_from_file(classes_file)
    subdatasets = discover_subdatasets(input_root)
    if not subdatasets:
        sys.exit(f"[错误] {input_root} 下未找到含 images/ 的子数据集")

    print("=" * 60)
    print(f"批处理模式: 发现 {len(subdatasets)} 个子数据集")
    print(f"输入根: {input_root}")
    print(f"输出根: {output_root}")
    print(f"无划分数据集 val_split={val_split}, link={link_mode}"
          f"{', DRY RUN' if dry_run else ''}")
    print("=" * 60)

    summary = {"datasets": [], "total_train_imgs": 0, "total_val_imgs": 0,
               "total_anns": 0}

    for idx, (name, kind, sub_root, train_src, val_src) in enumerate(subdatasets, 1):
        print(f"\n[{idx}/{len(subdatasets)}] {name} ({'已有划分' if kind == 'split' else '自动划分'})")
        out_dir = os.path.join(output_root, name)
        train_out = os.path.join(out_dir, "train")
        valid_out = os.path.join(out_dir, "valid")

        if kind == "split":
            train_imgs = collect_images_from_dir(train_src)
            val_imgs = collect_images_from_dir(val_src)
            # relpath 基准取 images/ 的父目录(train/ 或 val/), 这样 file_name = images/xxx.jpg,
            # 落盘到 out/<name>/train|valid/images/xxx.jpg, 与输出目录结构一致
            train_root = os.path.dirname(train_src)  # .../<name>/train
            val_root = os.path.dirname(val_src)      # .../<name>/val
        else:
            all_imgs = collect_images_from_dir(train_src)  # train_src 这里是 images_dir
            train_imgs, val_imgs = auto_split(all_imgs, val_split)
            # auto 模式基准取 images/ 的父目录(子数据集根), file_name = images/xxx.jpg
            train_root = os.path.dirname(train_src)  # .../<name>
            val_root = os.path.dirname(train_src)

        ds_stats = {"name": name, "kind": kind, "train_images": 0, "val_images": 0,
                    "train_anns": 0, "val_anns": 0}

        if train_imgs:
            coco_t, st_t = convert_split("train", train_imgs, train_root, train_out,
                                         categories, link_mode, dry_run, limit)
            ds_stats["train_images"] = len(coco_t["images"])
            ds_stats["train_anns"] = len(coco_t["annotations"])
            summary["total_train_imgs"] += ds_stats["train_images"]
            summary["total_anns"] += ds_stats["train_anns"]
        else:
            print("  [train] 无图片, 跳过")

        if val_imgs:
            coco_v, st_v = convert_split("valid", val_imgs, val_root, valid_out,
                                         categories, link_mode, dry_run, limit)
            ds_stats["val_images"] = len(coco_v["images"])
            ds_stats["val_anns"] = len(coco_v["annotations"])
            summary["total_val_imgs"] += ds_stats["val_images"]
            summary["total_anns"] += ds_stats["val_anns"]
        else:
            print("  [valid] 无图片, 跳过")

        summary["datasets"].append(ds_stats)

    if not dry_run:
        summary_path = os.path.join(output_root, "batch_summary.json")
        os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"批处理完成: 训练 {summary['total_train_imgs']} 张, "
          f"验证 {summary['total_val_imgs']} 张, 标注 {summary['total_anns']} 条")
    print(f"汇总: {os.path.join(output_root, 'batch_summary.json')}")
    print("\n各子数据集训练配置片段 (path 填 OUTPUT_ROOT, train/val 填 子集名/train|valid):")
    for ds in summary["datasets"]:
        print(f"  {ds['name']}: path={output_root}/{ds['name']} "
              f"train=train val=valid ann_file=_annotations.coco.json "
              f"(train {ds['train_images']} / val {ds['val_images']} 张)")
    print("=" * 60)


# ---------------------------------------------------------------------------
# 老的列表/扫描单数据集模式 (兼容 --root 用法)
# ---------------------------------------------------------------------------

def relocate(path, root):
    """把列表里的图片路径落实为 root 下真实存在的文件。"""
    parts = os.path.normpath(path).split(os.sep)
    for i in range(len(parts)):
        cand = os.path.join(root, *parts[i:])
        if os.path.isfile(cand):
            return cand
    if os.path.isfile(path):
        return path
    return None


def collect_list_mode(root, train_list, val_list):
    """按 split 列表收集 (split, 图片路径) 对。"""
    entries = {"train": [], "valid": []}
    for split, list_arg in (("train", train_list), ("valid", val_list)):
        if not list_arg:
            continue
        list_path = list_arg if os.path.isabs(list_arg) else os.path.join(root, list_arg)
        if not os.path.isfile(list_path):
            sys.exit(f"[错误] split 列表不存在: {list_path}")
        with open(list_path, encoding="utf-8") as f:
            entries[split] = [l.strip() for l in f if l.strip()]
    if not entries["train"] and not entries["valid"]:
        sys.exit("[错误] 列表模式至少需要 --train-list 或 --val-list 之一")
    return entries


def collect_scan_mode(images_dir, val_split, seed=42):
    """递归扫描有标签的图片, 按 val_split 随机划分。"""
    imgs = []
    for dirpath, _, files in os.walk(images_dir):
        for fn in sorted(files):
            if os.path.splitext(fn)[1].lower() in IMG_EXTS:
                full = os.path.join(dirpath, fn)
                if find_label(full):
                    imgs.append(full)
    if not imgs:
        sys.exit(f"[错误] {images_dir} 下未找到任何带标签的图片")
    return auto_split(imgs, val_split, seed)


def run_single_root(args, categories):
    """老的单数据集流程: --root + (--train-list/--val-list 或 --images-dir)。"""
    args.root = os.path.abspath(args.root)
    args.out = os.path.abspath(args.out)
    if args.out == args.root or args.out.startswith(args.root + os.sep):
        sys.exit("[错误] 输出目录不能是 --root 或其子目录 (避免污染源数据集)")

    if args.images_dir:
        if args.train_list or args.val_list:
            sys.exit("[错误] --images-dir 与 --train-list/--val-list 只能二选一")
        train_imgs, val_imgs = collect_scan_mode(args.images_dir, args.split, args.seed)
        entries = {"train": train_imgs, "valid": val_imgs}
        rel_root = args.root
    else:
        entries = collect_list_mode(args.root, args.train_list, args.val_list)
        # 列表里的路径先重定位到 root 下, 再统一用 root 作 relpath 基准
        for split in ("train", "valid"):
            entries[split] = [relocate(p, args.root) or p for p in entries[split]]
        rel_root = args.root

    overlap = set(os.path.abspath(p) for p in entries["train"]) & \
              set(os.path.abspath(p) for p in entries["valid"])
    if overlap:
        print(f"[警告] 训练/验证列表有 {len(overlap)} 张重复图片, 请检查 split 划分",
              file=sys.stderr)

    # 应用 --rename
    cats = [dict(c) for c in categories]
    renames = {}
    for item in args.rename or []:
        if "=" not in item:
            sys.exit(f"[错误] --rename 格式应为 旧名=新提示词, 收到: {item!r}")
        old, new = item.split("=", 1)
        renames[old.strip()] = new.strip()
    if renames:
        for c in cats:
            if c["name"] in renames:
                c["name"] = renames[c["name"]]
        print("应用 --rename 后的类别名:")
        for c in cats:
            print(f"  [{c['id']}] {c['name']!r}")

    for split in ("train", "valid"):
        paths = entries[split]
        if args.limit:
            paths = paths[:args.limit]
        if not paths:
            print(f"[{split}] 无条目, 跳过")
            continue
        out_dir = os.path.join(args.out, split)
        convert_split(split, paths, rel_root, out_dir, cats, args.link, args.dry_run)

    print("\n转换完成。数据集配置片段 (存为 configs/datasets/xxx.yaml 即可训练):")
    print(f"  path: {args.out}")
    print("  train: train")
    print("  val: valid")
    print("  ann_file: _annotations.coco.json")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="YOLO 分割数据集 → COCO (SAM 3 Roboflow 风格) 转换。"
                    "无参运行走批处理(用文件开头的 INPUT_ROOT/OUTPUT_ROOT);"
                    "传 --root 走单数据集流程。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 批处理覆盖参数
    p.add_argument("--input-root", default=None,
                   help="批处理输入根目录 (覆盖 INPUT_ROOT 常量)")
    p.add_argument("--output-root", default=None,
                   help="批处理输出根目录 (覆盖 OUTPUT_ROOT 常量)")
    p.add_argument("--val-split", type=float, default=None,
                   help="无划分数据集的验证集比例 (覆盖 VAL_SPLIT 常量, 默认 0.1)")
    p.add_argument("--link", choices=["hard", "symlink", "copy"], default=LINK_MODE,
                   help="图片落盘方式: 硬链接 / 软链接 / 复制")
    p.add_argument("--classes", default=None,
                   help="类别名文件 (批处理默认取 INPUT_ROOT/classes.txt)")
    p.add_argument("--limit", type=int, default=None, help="每个 split 最多处理 N 张 (debug)")
    p.add_argument("--dry-run", action="store_true", help="只解析和统计, 不写任何文件")
    # 老的单数据集参数 (传 --root 即进入单数据集流程)
    p.add_argument("--root", default=None,
                   help="单数据集模式根目录 (传此参数则不走批处理)")
    p.add_argument("--out", default=None, help="单数据集模式输出目录")
    p.add_argument("--train-list", default=None, help="单数据集: 训练集图片列表 txt")
    p.add_argument("--val-list", default=None, help="单数据集: 验证集图片列表 txt")
    p.add_argument("--images-dir", default=None,
                   help="单数据集: 递归扫描图片的目录 (替代列表模式)")
    p.add_argument("--split", type=float, default=0.1,
                   help="单数据集 --images-dir 模式的验证集比例")
    p.add_argument("--seed", type=int, default=42, help="单数据集扫描模式随机种子")
    p.add_argument("--rename", action="append", default=[], metavar="旧名=新提示词",
                   help="类别改名 (类别名即 SAM3 文本 prompt), 可多次指定")
    return p.parse_args()


def main():
    args = parse_args()

    if args.dry_run:
        print("=== DRY RUN: 只解析统计, 不写文件 ===")
    if args.limit:
        print(f"=== --limit {args.limit}: 每个 split 只处理前 {args.limit} 张 ===")

    # 单数据集模式: 传了 --root
    if args.root:
        if not args.out:
            sys.exit("[错误] --root 模式需要同时指定 --out")
        classes_path = args.classes or os.path.join(args.root, "classes.txt")
        categories = load_classes_from_file(classes_path)
        run_single_root(args, categories)
        return

    # 批处理模式 (默认)
    input_root = args.input_root or INPUT_ROOT
    output_root = args.output_root or OUTPUT_ROOT
    val_split = args.val_split if args.val_split is not None else VAL_SPLIT
    classes_path = args.classes or os.path.join(input_root, "classes.txt")
    if not os.path.isdir(input_root):
        sys.exit(f"[错误] INPUT_ROOT 不是目录: {input_root}"
                 f" (改文件开头 INPUT_ROOT 或用 --input-root)")
    batch_convert(input_root, output_root, classes_path, val_split,
                  args.link, args.dry_run, args.limit)


if __name__ == "__main__":
    main()
