#!/usr/bin/env python3
# YOLO 分割数据集 → COCO 格式转换工具 (SAM 3 训练用)
# ==================================================
# 把 YOLO 分割标注 (class_id + 归一化多边形) 转为本项目训练所需的
# Roboflow 风格 COCO 布局:
#
#   <out>/
#     train/                       # 图片 (保留源目录相对结构, 避免同名冲突)
#       _annotations.coco.json
#     valid/
#       _annotations.coco.json
#
# 配合 configs/datasets/ 下的数据集配置使用 (转换结束会打印配置片段):
#   path: <out> / train: train / val: valid / ann_file: _annotations.coco.json
#
# 输入支持两种方式:
#   1. split 列表模式 (--train-list/--val-list): 每行一个图片路径的 txt
#      (列表里的过期绝对路径会自动按后缀在 --root 下重新定位)
#   2. 目录扫描模式 (--images-dir): 递归找有标签的图片, 按 --split 自动划分
#
# 标签文件查找顺序: <图片同名>_seg.txt → <图片同名>.txt (标准 YOLO)
# 找不到标签的图片按无标注背景图处理 (计入统计, 作为负样本保留)。
#
# 类别名即 SAM 3 训练的文本 prompt, 可用 --rename "旧名=新提示词" 在转换时改名
# (参考: 先零样本测出最佳提示词, 再把标签名映射成它)。
#
# 用法示例:
#   python tools/yolo_seg_to_coco.py \
#       --root /path/to/highway_road \
#       --train-list train_seg_det.txt --val-list val_seg_det.txt \
#       --classes classes.txt \
#       --out /path/to/highway_road_coco
#
#   # 先验证再执行:
#   python tools/yolo_seg_to_coco.py ... --dry-run
#
# 依赖: 仅 stdlib + Pillow。

import argparse
import json
import os
import random
import sys

from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(
        description="YOLO 分割数据集 → COCO (SAM 3 Roboflow 风格) 转换",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--root", required=True,
                   help="数据集根目录 (列表路径重定位 / 相对结构的基准)")
    p.add_argument("--train-list", help="训练集图片列表 txt (每行一个图片路径)")
    p.add_argument("--val-list", help="验证集图片列表 txt")
    p.add_argument("--images-dir", help="(替代列表模式) 递归扫描图片的目录")
    p.add_argument("--split", type=float, default=0.1,
                   help="扫描模式下验证集比例 (0 = 全部进训练集)")
    p.add_argument("--seed", type=int, default=42, help="扫描模式自动划分的随机种子")
    p.add_argument("--classes", help="类别名文件 (默认 <root>/classes.txt, 每行一个)")
    p.add_argument("--rename", action="append", default=[], metavar="旧名=新提示词",
                   help="类别改名 (类别名即 SAM3 文本 prompt), 可多次指定")
    p.add_argument("--out", required=True, help="输出目录 (不得与 --root 相同)")
    p.add_argument("--link", choices=["hard", "symlink", "copy"], default="hard",
                   help="图片落盘方式: 硬链接 / 软链接 / 复制")
    p.add_argument("--limit", type=int, default=None, help="每个 split 最多处理 N 张 (debug)")
    p.add_argument("--dry-run", action="store_true", help="只解析和统计, 不写任何文件")
    return p.parse_args()


def load_classes(args):
    """读类别名 (0 号 = 第一行), 应用 --rename 映射, 返回 1-based COCO categories。"""
    path = args.classes or os.path.join(args.root, "classes.txt")
    if not os.path.isfile(path):
        sys.exit(f"[错误] 类别文件不存在: {path} (用 --classes 指定)")
    with open(path, encoding="utf-8") as f:
        names = [l.strip() for l in f if l.strip()]
    if not names:
        sys.exit(f"[错误] 类别文件为空: {path}")

    renames = {}
    for item in args.rename:
        if "=" not in item:
            sys.exit(f"[错误] --rename 格式应为 旧名=新提示词, 收到: {item!r}")
        old, new = item.split("=", 1)
        renames[old.strip()] = new.strip()
    for old in renames:
        if old not in names:
            print(f"[警告] --rename 的旧名 {old!r} 不在类别文件中", file=sys.stderr)

    mapped = [renames.get(n, n) for n in names]
    print(f"共 {len(names)} 个类别 (类别名 = SAM3 文本 prompt):")
    for i, (orig, name) in enumerate(zip(names, mapped)):
        mark = f"  <- 原名 {orig!r}" if name != orig else ""
        print(f"  [{i}] {name!r}{mark}")
    # COCO category id 用 1-based (0 留给背景的习惯; 后端按 id->name 建映射, 不强制)
    return [{"id": i + 1, "name": n, "supercategory": "none"} for i, n in enumerate(mapped)]


def relocate(path, root):
    """把列表里的图片路径落实为 root 下真实存在的文件。

    优先在 root 下按目录后缀定位 (从最长后缀开始逐个尝试)——列表里是迁移前
    的过期绝对路径也能正确重定位, 且保证返回路径一定在 root 之内;
    root 下实在找不到才回退原路径 (由调用方对 root 之外的路径做扁平化处理)。
    """
    parts = os.path.normpath(path).split(os.sep)
    for i in range(len(parts)):
        cand = os.path.join(root, *parts[i:])
        if os.path.isfile(cand):
            return cand
    if os.path.isfile(path):
        return path
    return None


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


def collect_list_mode(args):
    """按 split 列表收集 (split, 图片路径) 对。"""
    entries = {"train": [], "valid": []}
    for split, list_arg in (("train", args.train_list), ("valid", args.val_list)):
        if not list_arg:
            continue
        list_path = list_arg if os.path.isabs(list_arg) else os.path.join(args.root, list_arg)
        if not os.path.isfile(list_path):
            sys.exit(f"[错误] split 列表不存在: {list_path}")
        with open(list_path, encoding="utf-8") as f:
            entries[split] = [l.strip() for l in f if l.strip()]
    if not entries["train"] and not entries["valid"]:
        sys.exit("[错误] 列表模式至少需要 --train-list 或 --val-list 之一")
    return entries


def collect_scan_mode(args):
    """递归扫描有标签的图片, 按 --split 随机划分。"""
    imgs = []
    for dirpath, _, files in os.walk(args.images_dir):
        for fn in sorted(files):
            if os.path.splitext(fn)[1].lower() in IMG_EXTS:
                full = os.path.join(dirpath, fn)
                if find_label(full):
                    imgs.append(full)
    if not imgs:
        sys.exit(f"[错误] {args.images_dir} 下未找到任何带标签的图片")
    rng = random.Random(args.seed)
    rng.shuffle(imgs)
    n_val = int(round(len(imgs) * args.split))
    return {"train": imgs[n_val:], "valid": imgs[:n_val]}


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
            import shutil
            shutil.copy2(src, dst)  # 跨文件系统时硬链接退化为复制


def convert_split(split, img_paths, args, categories):
    """处理一个 split: 解析标注、落盘图片、生成 COCO dict。返回 (coco, stats)。"""
    coco = {"images": [], "annotations": [], "categories": categories}
    stats = {"no_label": 0, "not_found": 0, "bad_lines": 0,
             "clipped": 0, "skipped_class": 0, "linked": 0, "outside_root": 0}
    ann_id = 1
    n_classes = len(categories)

    for img_id, raw_path in enumerate(img_paths, start=1):
        src = relocate(raw_path, args.root)
        if src is None:
            stats["not_found"] += 1
            print(f"[警告] 图片找不到, 跳过: {raw_path}", file=sys.stderr)
            continue
        rel = os.path.relpath(src, args.root)
        if rel.startswith(".." + os.sep) or rel == "..":
            # 源图在 root 之外: 用 路径哈希+原名 扁平化, 防止写出输出目录
            import hashlib
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

        if not args.dry_run:
            dst = os.path.join(args.out, split, rel)
            link_or_copy(src, dst, args.link)
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

    if not args.dry_run:
        ann_path = os.path.join(args.out, split, "_annotations.coco.json")
        os.makedirs(os.path.dirname(ann_path), exist_ok=True)
        with open(ann_path, "w", encoding="utf-8") as f:
            json.dump(coco, f, ensure_ascii=False)

    print(f"[{split}] 图片 {len(coco['images'])} 张, 标注 {len(coco['annotations'])} 条"
          f" | 无标注背景图 {stats['no_label']}, 找不到图片 {stats['not_found']},"
          f" 坏行 {stats['bad_lines']}, 越界已截断 {stats['clipped']},"
          f" 类别越界丢弃 {stats['skipped_class']}, root 外扁平化 {stats['outside_root']}")
    return coco, stats


def main():
    args = parse_args()
    args.root = os.path.abspath(args.root)
    args.out = os.path.abspath(args.out)
    if args.out == args.root or args.out.startswith(args.root + os.sep):
        sys.exit("[错误] 输出目录不能是 --root 或其子目录 (避免污染源数据集)")
    if args.dry_run:
        print("=== DRY RUN: 只解析统计, 不写文件 ===")
    if args.limit:
        print(f"=== --limit {args.limit}: 每个 split 只处理前 {args.limit} 张 ===")

    categories = load_classes(args)

    if args.images_dir:
        if args.train_list or args.val_list:
            sys.exit("[错误] --images-dir 与 --train-list/--val-list 只能二选一")
        entries = collect_scan_mode(args)
    else:
        entries = collect_list_mode(args)

    overlap = set(os.path.abspath(p) for p in entries["train"]) & \
              set(os.path.abspath(p) for p in entries["valid"])
    if overlap:
        print(f"[警告] 训练/验证列表有 {len(overlap)} 张重复图片, 请检查 split 划分",
              file=sys.stderr)

    for split in ("train", "valid"):
        paths = entries[split]
        if args.limit:
            paths = paths[:args.limit]
        if not paths:
            print(f"[{split}] 无条目, 跳过")
            continue
        convert_split(split, paths, args, categories)

    print("\n转换完成。数据集配置片段 (存为 configs/datasets/xxx.yaml 即可训练):")
    print(f"  path: {args.out}")
    print("  train: train")
    print("  val: valid")
    print("  ann_file: _annotations.coco.json")
    print("  num_images: null")


if __name__ == "__main__":
    main()
