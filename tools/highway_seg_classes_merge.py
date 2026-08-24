#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标注合并脚本 - 将8分类合并为3分类，并融合同类别相接的分割区域
融合只针对真正相接(间隙 <=TOUCH_TOLERANCE)的同类别多边形，离得远的同类区域保持独立；
跨类别重叠只做截断(重叠像素判给距边界最远者)，不允许删除任何区域，
保证同一图像输出的多边形两两互不重叠。
支持Linux(软链接)和Windows(复制)
"""

import argparse
import json
import math
import os
import sys
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# Windows 终端编码修复
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ============================================================
# 数据集配置 (与训练配置 tasks/highway_seg 中未注释的数据集条目一致)
# 格式: (标注文件路径, 图片根目录, 数据集名称, dataset_flag)
#   - 有子目录的数据集: img_prefix 指向 .../train/images 或 .../val/images
#   - 无子目录的数据集: img_prefix 指向数据集根目录 (file_name 含 images/ 前缀)
#   - dataset_name 用于输出目录命名，避免不同数据集混到同一目录
#   - dataset_flag 对应训练配置中的类别映射标识，写入输出JSON的info字段
# 注意: 训练配置中"场景2"的train条目路径误写为场景1(flag=6却指向场景1)，
#       此处按真实路径修正为场景2；test_day/test_night 在服务器上有 train/val 子目录
# ============================================================
DATA_BASE = "/data/kaiyun/ding/datasets"
DATA_BASE_KAIYUN = "/data2/kaiyun/datasets_seg"

DATASET_CONFIGS = [
    # ==================== train ====================
    # --- 有子目录的数据集 (img_prefix 以 /images 结尾) ---
    (f"{DATA_BASE}/2501234348_1/train/trainset.json",
     f"{DATA_BASE}/2501234348_1/train/images",
     "2501234348_1", 2),
    (f"{DATA_BASE}/task1_test_train/task-1/train/trainset.json",
     f"{DATA_BASE}/task1_test_train/task-1/train/images",
     "task1_task-1", 3),
    (f"{DATA_BASE}/task1_test_train/train/train/trainset.json",
     f"{DATA_BASE}/task1_test_train/train/train/images",
     "task1_train", 4),
    (f"{DATA_BASE}/task1_test_train/场景1/train/trainset.json",
     f"{DATA_BASE}/task1_test_train/场景1/train/images",
     "task1_场景1", 5),
    (f"{DATA_BASE}/task1_test_train/场景2/train/trainset.json",
     f"{DATA_BASE}/task1_test_train/场景2/train/images",
     "task1_场景2", 6),
    (f"{DATA_BASE}/task1_test_train/test/train_day/train/trainset.json",
     f"{DATA_BASE}/task1_test_train/test/train_day/train/images",
     "task1_train_day", 7),
    (f"{DATA_BASE}/task1_test_train/test/train_night/train/trainset.json",
     f"{DATA_BASE}/task1_test_train/test/train_night/train/images",
     "task1_train_night", 8),
    (f"{DATA_BASE}/湖南/train/trainset.json",
     f"{DATA_BASE}/湖南/train/images",
     "湖南", 9),
    (f"{DATA_BASE}/淄博/all_汇总/train/trainset.json",
     f"{DATA_BASE}/淄博/all_汇总/train/images",
     "淄博", 9),
    (f"{DATA_BASE}/zibo_2/train/trainset.json",
     f"{DATA_BASE}/zibo_2/train/images",
     "zibo_2", 9),
    (f"{DATA_BASE}/103路采样+96路采样/train/trainset.json",
     f"{DATA_BASE}/103路采样+96路采样/train/images",
     "103路采样+96路采样", 9),
    (f"{DATA_BASE}/桂三20260204/train/trainset.json",
     f"{DATA_BASE}/桂三20260204/train/images",
     "桂三20260204", 9),
    (f"{DATA_BASE}/违法停车/train/trainset.json",
     f"{DATA_BASE}/违法停车/train/images",
     "违法停车", 9),
    # --- 无子目录的数据集 (img_prefix 指向数据集根目录，file_name 含 images/ 前缀) ---
    (f"{DATA_BASE_KAIYUN}/重庆20260226/trainset.json",
     f"{DATA_BASE_KAIYUN}/重庆20260226",
     "重庆20260226", 9),
    (f"{DATA_BASE_KAIYUN}/沪武/trainset.json",
     f"{DATA_BASE_KAIYUN}/沪武",
     "沪武", 9),
    (f"{DATA_BASE_KAIYUN}/宁沪-无锡高速大脑/trainset.json",
     f"{DATA_BASE_KAIYUN}/宁沪-无锡高速大脑",
     "宁沪-无锡高速大脑", 9),
    (f"{DATA_BASE_KAIYUN}/snap_picture/trainset.json",
     f"{DATA_BASE_KAIYUN}/snap_picture",
     "snap_picture", 9),

    # ==================== val ====================
    # --- 有子目录的数据集 ---
    (f"{DATA_BASE}/2501234348_1/val/valset.json",
     f"{DATA_BASE}/2501234348_1/val/images",
     "2501234348_1", 2),
    (f"{DATA_BASE}/task1_test_train/task-1/val/valset.json",
     f"{DATA_BASE}/task1_test_train/task-1/val/images",
     "task1_task-1", 3),
    (f"{DATA_BASE}/task1_test_train/train/val/valset.json",
     f"{DATA_BASE}/task1_test_train/train/val/images",
     "task1_train", 4),
    (f"{DATA_BASE}/task1_test_train/场景1/val/valset.json",
     f"{DATA_BASE}/task1_test_train/场景1/val/images",
     "task1_场景1", 5),
    (f"{DATA_BASE}/task1_test_train/场景2/val/valset.json",
     f"{DATA_BASE}/task1_test_train/场景2/val/images",
     "task1_场景2", 6),
    (f"{DATA_BASE}/task1_test_train/test/train_day/val/valset.json",
     f"{DATA_BASE}/task1_test_train/test/train_day/val/images",
     "task1_train_day", 7),
    (f"{DATA_BASE}/task1_test_train/test/train_night/val/valset.json",
     f"{DATA_BASE}/task1_test_train/test/train_night/val/images",
     "task1_train_night", 8),
    (f"{DATA_BASE}/湖南/val/valset.json",
     f"{DATA_BASE}/湖南/val/images",
     "湖南", 9),
    (f"{DATA_BASE}/淄博/all_汇总/val/valset.json",
     f"{DATA_BASE}/淄博/all_汇总/val/images",
     "淄博", 9),
    (f"{DATA_BASE}/zibo_2/val/valset.json",
     f"{DATA_BASE}/zibo_2/val/images",
     "zibo_2", 9),
    (f"{DATA_BASE}/103路采样+96路采样/val/valset.json",
     f"{DATA_BASE}/103路采样+96路采样/val/images",
     "103路采样+96路采样", 9),
    (f"{DATA_BASE}/桂三20260204/val/valset.json",
     f"{DATA_BASE}/桂三20260204/val/images",
     "桂三20260204", 9),
    (f"{DATA_BASE}/违法停车/val/valset.json",
     f"{DATA_BASE}/违法停车/val/images",
     "违法停车", 9),
    # --- 无子目录的数据集 ---
    (f"{DATA_BASE_KAIYUN}/重庆20260226/valset.json",
     f"{DATA_BASE_KAIYUN}/重庆20260226",
     "重庆20260226", 9),
    (f"{DATA_BASE_KAIYUN}/沪武/valset.json",
     f"{DATA_BASE_KAIYUN}/沪武",
     "沪武", 9),
    (f"{DATA_BASE_KAIYUN}/宁沪-无锡高速大脑/valset.json",
     f"{DATA_BASE_KAIYUN}/宁沪-无锡高速大脑",
     "宁沪-无锡高速大脑", 9),
    (f"{DATA_BASE_KAIYUN}/snap_picture/valset.json",
     f"{DATA_BASE_KAIYUN}/snap_picture",
     "snap_picture", 9),
]

# 默认输出目录(相对当前工作目录)，可用命令行参数 -o/--output 覆盖，
# 指定后在其下创建 certain/weak/uncertain 三个子目录
CERTAIN_OUTPUT_BASE = "highway_seg_merged/certain"
WEAK_OUTPUT_BASE = "highway_seg_merged/weak"
UNCERTAIN_OUTPUT_BASE = "highway_seg_merged/uncertain"

# 设置为True使用复制(Windows)，False使用软链接(Linux)
USE_COPY = False  # 输入路径为Linux服务器路径, 默认软链接; Windows本地运行时改为True

# 多边形相接容忍度(像素)：标注不够精确时容许小缝隙
TOUCH_TOLERANCE = 5.0

# 同类别区域融合参数
# 闭运算核大小: 桥接 <=TOUCH_TOLERANCE 的标注缝隙(tolerance=5 时为 7x7, 桥接约6px)
FUSE_CLOSING_KERNEL = 2 * int(math.ceil(TOUCH_TOLERANCE / 2)) + 1
# 闭运算对边界的最大外扩距离(两侧合计): 核半径 (K-1)/2 px, 跨类别间距小于该值就可能被闭运算压重叠
FUSE_BRIDGE_REACH = FUSE_CLOSING_KERNEL - 1
FUSE_SIMPLIFY_EPS = 1.5   # 融合轮廓 approxPolyDP 简化容差(像素)
# 融合/仲裁只做截断不做删除: 不设面积阈值, 所有连通区域都保留

# 类别映射
ORIGINAL_CATEGORIES = {
    1: "上行主线车道",
    2: "下行主线车道",
    3: "应急车道",
    4: "匝道（行车区）",
    5: "紧急停车带",
    6: "匝道导流区",
    7: "隔离带",
    8: "护栏"
}

ORIGINAL_COCO_CATEGORIES = [
    {"id": category_id, "name": category_name}
    for category_id, category_name in ORIGINAL_CATEGORIES.items()
]

NEW_CATEGORIES = [
    {"id": 1, "name": "上行"},
    {"id": 2, "name": "下行"},
    {"id": 3, "name": "隔离带"}
]

# ============================================================
# 类别映射: 数据集原始类别ID -> 模型统一类别ID
# 模型统一类别空间 (num_classes_value=9, 0=背景):
#   0=background, 1=highway_up(上行车道), 2=highway_down(下行车道),
#   3=emergency_lane(应急车道), 4=ramp(匝道), 5=escape_lane(避险车道),
#   6=ramp_guiding_area(匝道导流区), 7=isolation_zone(隔离带), 8=guardrail(护栏)
# ============================================================

# 合并后3分类(上行/下行/隔离带) -> 模型9分类统一空间
MERGED_CATEGORY_MAPPING = {
    1: 1,   # 上行   -> highway_up
    2: 2,   # 下行   -> highway_down
    3: 7,   # 隔离带 -> isolation_zone
}

# 原始8分类 -> 模型9分类统一空间 (identity, 默认回退)
ORIGINAL_CATEGORY_MAPPING = {
    1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8
}

# dataset_flag -> 对应的原始类别映射 (与 HighwaySegDataset/LoadHighwaySegLabel 中的 id_mapping 一致)
DATASET_FLAG_CATEGORY_MAPPING = {
    0: {5: 1, 6: 2, 9: 3, 10: 5, 11: 4},                                              # 车辆分割 612
    1: {5: 1, 6: 2, 9: 3, 10: 5, 11: 4},                                              # 827
    2: {1: 2, 2: 1, 3: 6, 4: 7, 5: 0, 6: 8, 7: 3, 8: 4, 9: 5},                       # task-2 (2501234348_1)
    3: {1: 2, 2: 1, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 0, 10: 0},                # task-1
    4: {1: 2, 2: 1, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 0, 10: 0},                # train
    5: {1: 2, 2: 1, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 0, 10: 0},                # 场景1
    6: {1: 2, 2: 1, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 0, 10: 0},                # 场景2
    7: {1: 2, 2: 1, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 0, 10: 0},                # test_day
    8: {1: 2, 2: 1, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 0, 10: 0},                # test_night
    9: {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8},                              # 湖南/淄博/zibo_2/103路/桂三/违法停车/重庆/沪武/宁沪/snap_picture
}


def get_annotation_bbox(ann):
    """返回标注 bbox: (x1, y1, x2, y2)，优先聚合所有 segmentation 点。"""
    xs = []
    ys = []
    for points in ann.get('segmentation') or []:
        if len(points) >= 6:
            xs.extend(points[0::2])
            ys.extend(points[1::2])
    if xs and ys:
        return min(xs), min(ys), max(xs), max(ys)

    if 'bbox' in ann:
        x, y, w, h = ann['bbox']
        return x, y, x + w, y + h
    return None


def get_annotation_center(ann):
    bbox = get_annotation_bbox(ann)
    if bbox is None:
        return None, None
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def bbox_area(bbox):
    if bbox is None:
        return 0.0
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_gap(a, b):
    """两个 bbox 的欧氏间距；相交或相贴时为 0。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    dx = max(bx1 - ax2, ax1 - bx2, 0)
    dy = max(by1 - ay2, ay1 - by2, 0)
    return math.hypot(dx, dy)


def iter_polygons(ann):
    """从 COCO segmentation 中提取多边形点列表。"""
    polygons = []
    for points in ann.get('segmentation') or []:
        if len(points) >= 6:
            polygon = [(points[i], points[i + 1]) for i in range(0, len(points), 2)]
            if len(polygon) >= 3:
                polygons.append(polygon)
    return polygons


def point_in_polygon(point, polygon):
    """射线法判断点是否在多边形内。"""
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def segment_distance(a, b, c, d):
    """两条线段的最短距离；相交时为 0。"""
    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if abs(val) < 1e-9:
            return 0
        return 1 if val > 0 else 2

    def on_segment(p, q, r):
        return (min(p[0], r[0]) - 1e-9 <= q[0] <= max(p[0], r[0]) + 1e-9 and
                min(p[1], r[1]) - 1e-9 <= q[1] <= max(p[1], r[1]) + 1e-9)

    def intersects(p1, q1, p2, q2):
        o1 = orientation(p1, q1, p2)
        o2 = orientation(p1, q1, q2)
        o3 = orientation(p2, q2, p1)
        o4 = orientation(p2, q2, q1)
        if o1 != o2 and o3 != o4:
            return True
        return ((o1 == 0 and on_segment(p1, p2, q1)) or
                (o2 == 0 and on_segment(p1, q2, q1)) or
                (o3 == 0 and on_segment(p2, p1, q2)) or
                (o4 == 0 and on_segment(p2, q1, q2)))

    def point_segment_distance(p, u, v):
        ux, uy = u
        vx, vy = v
        px, py = p
        dx = vx - ux
        dy = vy - uy
        if dx == 0 and dy == 0:
            return math.hypot(px - ux, py - uy)
        t = max(0.0, min(1.0, ((px - ux) * dx + (py - uy) * dy) / (dx * dx + dy * dy)))
        proj_x = ux + t * dx
        proj_y = uy + t * dy
        return math.hypot(px - proj_x, py - proj_y)

    if intersects(a, b, c, d):
        return 0.0
    return min(
        point_segment_distance(a, c, d),
        point_segment_distance(b, c, d),
        point_segment_distance(c, a, b),
        point_segment_distance(d, a, b)
    )


def polygon_distance(poly_a, poly_b):
    """两个多边形的最短边界距离；包含/相交时为 0。"""
    if not poly_a or not poly_b:
        return None
    if any(point_in_polygon(p, poly_b) for p in poly_a) or any(point_in_polygon(p, poly_a) for p in poly_b):
        return 0.0

    min_dist = None
    for i in range(len(poly_a)):
        a1 = poly_a[i]
        a2 = poly_a[(i + 1) % len(poly_a)]
        for j in range(len(poly_b)):
            b1 = poly_b[j]
            b2 = poly_b[(j + 1) % len(poly_b)]
            dist = segment_distance(a1, a2, b1, b2)
            if min_dist is None or dist < min_dist:
                min_dist = dist
                if min_dist == 0:
                    return 0.0
    return min_dist


def min_polygon_gap(target_polygons, seeds):
    """目标多边形到一组种子多边形的最短真实边界距离。"""
    if not target_polygons or not seeds:
        return None
    min_dist = None
    for target_poly in target_polygons:
        for seed in seeds:
            for seed_poly in seed.get('polygons') or []:
                dist = polygon_distance(target_poly, seed_poly)
                if dist is None:
                    continue
                if min_dist is None or dist < min_dist:
                    min_dist = dist
                    if min_dist == 0:
                        return 0.0
    return min_dist


def make_seed(ann):
    bbox = get_annotation_bbox(ann)
    if bbox is None:
        return None
    cx, cy = get_annotation_center(ann)
    return {'center': (cx, cy), 'bbox': bbox, 'polygons': iter_polygons(ann), 'category_id': ann['category_id']}


def median(values):
    """返回中位数。"""
    values = sorted(values)
    if not values:
        return None
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2


def nearest_seed(center, seeds):
    """返回最近种子中心和距离。"""
    if center is None or not seeds:
        return None, None
    cx, cy = center
    nearest = min(seeds, key=lambda seed: (cx - seed['center'][0]) ** 2 + (cy - seed['center'][1]) ** 2)
    nearest_center = nearest['center']
    distance = math.hypot(cx - nearest_center[0], cy - nearest_center[1])
    return nearest, distance


def nearest_bbox_gap(target_bbox, seeds):
    if target_bbox is None or not seeds:
        return None, None
    nearest = min(seeds, key=lambda seed: bbox_gap(target_bbox, seed['bbox']))
    distance = bbox_gap(target_bbox, nearest['bbox'])
    return nearest, distance


def build_direction_seeds(annotations):
    """提取每张图中可信的上/下行主线和隔离带几何种子。"""
    up_seeds = []
    down_seeds = []
    divider_seeds = []
    for ann in annotations:
        seed = make_seed(ann)
        if seed is None:
            continue
        if ann['category_id'] == 1:
            up_seeds.append(seed)
        elif ann['category_id'] == 2:
            down_seeds.append(seed)
        elif ann['category_id'] == 7:
            divider_seeds.append(seed)
    return up_seeds, down_seeds, divider_seeds


def infer_divider_sides(up_seeds, down_seeds, divider_seeds):
    """用隔离带确定左右两侧分别属于上行还是下行。"""
    divider_x = median([seed['center'][0] for seed in divider_seeds])
    if divider_x is None:
        return None, None, None

    left_votes = []
    right_votes = []
    for seed in up_seeds:
        center = seed['center']
        (left_votes if center[0] < divider_x else right_votes).append(1)
    for seed in down_seeds:
        center = seed['center']
        (left_votes if center[0] < divider_x else right_votes).append(2)

    def majority(votes):
        up_count = votes.count(1)
        down_count = votes.count(2)
        if up_count > down_count:
            return 1
        if down_count > up_count:
            return 2
        return None

    left_class = majority(left_votes)
    right_class = majority(right_votes)

    if left_class is None and right_class in (1, 2):
        left_class = 2 if right_class == 1 else 1
    if right_class is None and left_class in (1, 2):
        right_class = 2 if left_class == 1 else 1
    if left_class == right_class:
        return divider_x, None, None

    return divider_x, left_class, right_class


def detect_divider_direction_issue(up_seeds, down_seeds, divider_seeds):
    """检测隔离带两侧原始主线方向是否自相矛盾。

    如果图中已经有隔离带，且隔离带两侧都存在 1/2 主线标注，
    但两侧方向相同，说明原始标注很可能有问题。
    若只有一侧存在主线标注则视为正常：另一侧可能太暗/不可见。
    若同一侧同时出现上/下行，通常是斜向隔离带、多路面、匝道或透视导致的
    全局 divider_x 粗分失效，不应图级拦截，交给后续 polygon/bbox 邻近规则处理。
    这种图不能强行按距离兜底合并，应整体进入 uncertain 供人工修复。
    """
    divider_x = median([seed['center'][0] for seed in divider_seeds])
    if divider_x is None:
        return None

    left_labels = set()
    right_labels = set()
    for seed in up_seeds:
        (left_labels if seed['center'][0] < divider_x else right_labels).add(1)
    for seed in down_seeds:
        (left_labels if seed['center'][0] < divider_x else right_labels).add(2)

    if not left_labels or not right_labels:
        return None

    if len(left_labels) > 1 or len(right_labels) > 1:
        return None

    left_class = next(iter(left_labels))
    right_class = next(iter(right_labels))
    if left_class == right_class:
        return f"隔离带两侧主线均为{class_name(left_class)}，无法分清上下行，疑似原始标注错误(divider_x={divider_x:.2f})"

    return None


def class_name(class_id):
    return "上行" if class_id == 1 else "下行"


def distance_confidence(dist_a, dist_b):
    """两个方向距离的相对差异，越大越可信。"""
    if dist_a is None or dist_b is None:
        return 1.0
    total = dist_a + dist_b
    if total == 0:
        return 0.0
    return abs(dist_a - dist_b) / total


def make_decision(new_id, status, reason, meta):
    return new_id, status, reason, meta


def classify_annotation(ann, up_seeds, down_seeds, divider_info, img_width, img_height):
    """将8分类标注融合为3分类。

    直接映射：1->上行，2->下行，7->隔离带，8删除。
    对 3/4/5/6：用多边形相接/重叠判断与哪条主线连接，
    不按距离划分（上下行可能在画面同一侧）。
    """
    cat_id = ann['category_id']
    if cat_id == 8:
        return None, 'deleted', "护栏删除", {}
    if cat_id == 7:
        return 3, 'strong', "隔离带保留", {}
    if cat_id == 1:
        return 1, 'strong', "上行主线车道", {}
    if cat_id == 2:
        return 2, 'strong', "下行主线车道", {}

    # cat 3/4/5/6: 检查多边形与哪条主线相接
    old_name = ORIGINAL_CATEGORIES.get(cat_id, f"未知类别{cat_id}")
    polygons = iter_polygons(ann)
    bbox = get_annotation_bbox(ann)
    area_ratio = bbox_area(bbox) / max(img_width * img_height, 1) if bbox else 0.0

    poly_gap_up = min_polygon_gap(polygons, up_seeds)
    poly_gap_down = min_polygon_gap(polygons, down_seeds)

    touches_up = poly_gap_up is not None and poly_gap_up <= TOUCH_TOLERANCE
    touches_down = poly_gap_down is not None and poly_gap_down <= TOUCH_TOLERANCE

    if touches_up and not touches_down:
        return make_decision(1, 'strong', f"{old_name}与上行主线相接", {
            'method': 'polygon_touch',
            'gap_up': round(poly_gap_up, 2),
            'gap_down': round(poly_gap_down, 2) if poly_gap_down is not None else None,
            'area_ratio': round(area_ratio, 4),
            'confidence': 1.0
        })
    if touches_down and not touches_up:
        return make_decision(2, 'strong', f"{old_name}与下行主线相接", {
            'method': 'polygon_touch',
            'gap_up': round(poly_gap_up, 2) if poly_gap_up is not None else None,
            'gap_down': round(poly_gap_down, 2),
            'area_ratio': round(area_ratio, 4),
            'confidence': 1.0
        })
    if touches_up and touches_down:
        return None, 'review_required', f"{old_name}同时与上下行相接，无法确定归属", {
            'method': 'polygon_touch_both',
            'gap_up': round(poly_gap_up, 2),
            'gap_down': round(poly_gap_down, 2),
            'area_ratio': round(area_ratio, 4)
        }
    # 未与任何主线相接
    gap_up_str = round(poly_gap_up, 2) if poly_gap_up is not None else None
    gap_down_str = round(poly_gap_down, 2) if poly_gap_down is not None else None
    return None, 'review_required', f"{old_name}未与任何主线相接(距上行={gap_up_str}, 距下行={gap_down_str})", {
        'method': 'polygon_no_touch',
        'gap_up': gap_up_str,
        'gap_down': gap_down_str,
        'area_ratio': round(area_ratio, 4)
    }


def rasterize_annotation(ann, width, height):
    """把 annotation 的全部分割多边形栅格化为 uint8 mask(0/1)。"""
    mask = np.zeros((height, width), dtype=np.uint8)
    for points in ann.get('segmentation') or []:
        if len(points) < 6:
            continue
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
        cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
    return mask


def mask_to_fused_annotations(mask, template_ann, eps=FUSE_SIMPLIFY_EPS):
    """把单个类别的 mask 转回 annotation 列表，每个连通区域一个。

    COCO 多边形格式无法表达孔洞，与标准 COCO 转换器一致使用
    RETR_EXTERNAL(孔洞随外轮廓一起填满)。eps=0 时不做简化，轮廓逐像素精确。
    只做截断不做删除: 不设面积阈值，所有连通区域都保留；
    退化轮廓(1-2px碎屑)用其包围盒矩形兜底，保证区域不丢失。
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fused = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, eps, True) if eps > 0 else contour
        if len(approx) >= 3:
            points = approx.reshape(-1, 2).astype(float)
            area = float(cv2.contourArea(approx))
        else:
            rx, ry, rw, rh = cv2.boundingRect(contour)
            rw = max(rw, 1)
            rh = max(rh, 1)
            points = np.array([[rx, ry], [rx + rw, ry], [rx + rw, ry + rh], [rx, ry + rh]], dtype=float)
            area = float(rw * rh)
        polygon = points.reshape(-1).tolist()
        x1, y1 = points.min(axis=0)
        x2, y2 = points.max(axis=0)
        new_ann = template_ann.copy()
        new_ann['segmentation'] = [polygon]
        new_ann['bbox'] = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
        new_ann['area'] = area
        new_ann['iscrowd'] = 0
        fused.append(new_ann)
    return fused


def annotations_overlap(annotations, width, height):
    """栅格化验证输出标注两两互不重叠。"""
    union = np.zeros((height, width), dtype=np.uint8)
    for ann in annotations:
        mask = rasterize_annotation(ann, width, height)
        if cv2.bitwise_and(union, mask).any():
            return True
        union = cv2.bitwise_or(union, mask)
    return False


def fuse_image_annotations(annotations, width, height):
    """融合同一图像内同类别且相接的分割区域，保证输出互不重叠且只做截断不做删除。

    流程: 各类别栅格化 -> 闭运算桥接同类小缝隙(<=TOUCH_TOLERANCE) ->
    跨类别仲裁(重叠像素判给距自身边界最远的类别, 即只截断重叠部分) ->
    区域守恒检查(被完全夺走的区域夺回其核心盘, 不允许整片删除) ->
    各类别连通域轮廓生成 annotation(所有区域保留, 无面积阈值过滤)。
    无同类融合需求且各类别间距超出闭运算影响范围时原样透传，避免栅格化精度损失。
    轮廓简化若引入像素级重叠则以不简化轮廓重试，保证最终输出两两不重叠。
    返回 (标注列表, {'fused': 融合消除的冗余标注数, 'overlap_px': 仲裁的跨类重叠像素数,
                     'restored': 守恒恢复的区域数})。
    """
    stats = {'fused': 0, 'overlap_px': 0, 'restored': 0}
    if len(annotations) <= 1:
        return annotations, stats

    by_category = defaultdict(list)
    for ann in annotations:
        by_category[ann['category_id']].append(ann)

    need_pipeline = any(len(group) > 1 for group in by_category.values())
    if not need_pipeline:
        # 各类别均单标注: 跨类别贴近到闭运算影响范围内(含重叠)才需要进 mask 流水线
        singletons = [group[0] for group in by_category.values()]
        for i in range(len(singletons)):
            polys_i = iter_polygons(singletons[i])
            for j in range(i + 1, len(singletons)):
                gap = None
                for poly_i in polys_i:
                    for poly_j in iter_polygons(singletons[j]):
                        dist = polygon_distance(poly_i, poly_j)
                        if dist is not None and (gap is None or dist < gap):
                            gap = dist
                if gap is not None and gap <= FUSE_BRIDGE_REACH:
                    need_pipeline = True
                    break
            if need_pipeline:
                break

    if not need_pipeline:
        return annotations, stats

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (FUSE_CLOSING_KERNEL, FUSE_CLOSING_KERNEL))

    # 1) 各类别栅格化 + 闭运算桥接同类小缝隙
    cat_masks = {}
    passthrough = []
    for cat_id, group in by_category.items():
        mask = None
        for ann in group:
            ann_mask = rasterize_annotation(ann, width, height)
            mask = ann_mask if mask is None else cv2.bitwise_or(mask, ann_mask)
        if mask is None or not mask.any():
            print(f"  ⚠ 类别{cat_id}栅格化为空，保留原始 {len(group)} 个标注")
            passthrough.extend(group)
            continue
        cat_masks[cat_id] = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 2) 跨类别仲裁: 重叠像素判给距自身边界最远者(只截断重叠部分), 保证类别间互不重叠
    cats = sorted(cat_masks)
    if len(cats) > 1:
        stack = np.stack([cat_masks[c] for c in cats])
        contested = stack.sum(axis=0) > 1
        stats['overlap_px'] = int(contested.sum())
        if contested.any():
            dists = np.stack([cv2.distanceTransform(m, cv2.DIST_L2, 3) for m in stack])
            winner = dists.argmax(axis=0)
            for i, cat_id in enumerate(cats):
                cat_masks[cat_id] = np.where((winner == i) & (stack[i] > 0), 1, 0).astype(np.uint8)

            # 区域守恒: 仲裁前的每个连通区域若被完全夺走，夺回其距边界最远的
            # 核心盘(半径2px)，并从其他类别截掉对应像素以维持互不重叠。
            # 只截断不删除：任何区域都不允许因仲裁整片消失。
            # 恢复的核心落在别类区域内部时会形成孔洞，而 COCO 多边形无法表达孔洞
            # (外轮廓会把孔洞重新填上)，故从核心朝最近的外部方向切一条 1px 狭缝，
            # 使包住它的类别单连通，外轮廓即可绕开核心；取最近方向避免把
            # 包住的类别拦腰切断。
            for i, cat_id in enumerate(cats):
                n_comp, comp_labels = cv2.connectedComponents(stack[i])
                restored = np.zeros_like(stack[i])
                slits = np.zeros_like(stack[i])
                others_union = np.zeros_like(stack[i])
                for j in range(len(cats)):
                    if j != i:
                        others_union = cv2.bitwise_or(others_union, cat_masks[cats[j]])
                for comp_id in range(1, n_comp):
                    comp = comp_labels == comp_id
                    if cat_masks[cat_id][comp].any():
                        continue
                    dist = cv2.distanceTransform(comp.astype(np.uint8), cv2.DIST_L2, 3)
                    _, _, _, peak = cv2.minMaxLoc(dist)
                    cv2.circle(restored, peak, 2, 1, -1)
                    px, py = peak
                    best_end = (0, py)
                    best_steps = None
                    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        cx, cy = px, py
                        steps = 0
                        while 0 <= cx < width and 0 <= cy < height and others_union[cy, cx]:
                            cx += dx
                            cy += dy
                            steps += 1
                        end = (min(max(cx, 0), width - 1), min(max(cy, 0), height - 1))
                        if best_steps is None or steps < best_steps:
                            best_steps = steps
                            best_end = end
                    cv2.line(slits, peak, best_end, 1, 1)
                restored = cv2.bitwise_and(restored, stack[i])
                if restored.any():
                    stats['restored'] += 1
                    slits[restored > 0] = 0  # 狭缝不切除恢复的核心自身
                    cat_masks[cat_id][restored > 0] = 1
                    for j in range(len(cats)):
                        if j != i:
                            cat_masks[cats[j]][restored > 0] = 0
                            cat_masks[cats[j]][slits > 0] = 0

    # 3) 连通域轮廓 -> annotation; 简化引入重叠时以 eps=0 重试保证互不重叠
    fused = []
    for eps in (FUSE_SIMPLIFY_EPS, 0.0):
        fused = []
        for cat_id in cats:
            fused.extend(mask_to_fused_annotations(cat_masks[cat_id], by_category[cat_id][0], eps))
        if eps == 0.0 or not annotations_overlap(fused, width, height):
            break

    n_input = len(annotations) - len(passthrough)
    stats['fused'] = max(n_input - len(fused), 0)
    return passthrough + fused, stats


def create_symlink(src_path, dst_path):
    """创建软链接；目标已存在时视为成功，保证重复图片仍写入JSON。"""
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists():
        return True
    try:
        os.symlink(src_path, dst_path)
        return True
    except Exception as e:
        print(f"  ⚠ 软链接创建失败: {e}")
        return False


def copy_file_safe(src_path, dst_path):
    """复制文件；目标已存在时视为成功，保证重复图片仍写入JSON。"""
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists():
        return True
    try:
        shutil.copy2(src_path, dst_path)
        return True
    except Exception as e:
        print(f"  ⚠ 复制失败: {e}")
        return False


def progress_iter(iterable, desc, total):
    """有 tqdm 时显示进度条，否则每完成约 10% 打印一行进度。"""
    if tqdm is not None:
        yield from tqdm(iterable, desc=desc, total=total, ncols=100)
        return
    step = max(total // 10, 1)
    count = 0
    for item in iterable:
        yield item
        count += 1
        if count % step == 0 or count == total:
            print(f"    {desc}: {count}/{total} ({count * 100 // total}%)")


def process_dataset(ann_file, img_prefix, dataset_name, certain_base, weak_base, uncertain_base, dataset_flag=9):
    """处理单个标注文件

    Args:
        ann_file: 标注JSON文件路径
        img_prefix: 图片根目录路径
        dataset_name: 数据集名称（用于输出目录命名）
        certain_base: 确定分类输出根目录
        weak_base: 弱确定分类输出根目录
        uncertain_base: 不确定分类输出根目录
        dataset_flag: 数据集类别映射标识，写入输出JSON的info字段
    """
    ann_file = Path(ann_file)
    img_prefix = Path(img_prefix)

    if not ann_file.exists():
        print(f"⚠ 跳过（文件不存在）: {ann_file}")
        return None

    # ★ 修复：通过 img_prefix 是否以 /images 结尾来判断是否有子目录
    # 有子目录: img_prefix = ".../2501234348_1/train/images" → has_subdir=True
    # 无子目录: img_prefix = ".../重庆20260226" → has_subdir=False
    has_subdir = img_prefix.name == "images"

    # 输出目录
    output_certain_dataset_dir = certain_base / dataset_name
    output_weak_dataset_dir = weak_base / dataset_name
    output_uncertain_dataset_dir = uncertain_base / dataset_name

    # ★ 修复：JSON输出路径保持 train/val 结构，放在 split_dir 根目录
    # 图片放到 split_dir/images/ 子目录，避免和 JSON 混在一起
    # 有子目录: output/数据集名/train/trainset.json + output/数据集名/train/images/xxx.jpg
    # 无子目录: output/数据集名/trainset.json + output/数据集名/images/xxx.jpg
    if has_subdir:
        # 从 ann_file 中提取 train 或 val 部分
        # ann_file = ".../2501234348_1/train/trainset.json" → split_dir = "train"
        # ann_file = ".../2501234348_1/val/valset.json" → split_dir = "val"
        split_dir = ann_file.parent.name  # "train" 或 "val"
        output_certain_json_path = output_certain_dataset_dir / split_dir / ann_file.name
        output_weak_json_path = output_weak_dataset_dir / split_dir / ann_file.name
        output_uncertain_json_path = output_uncertain_dataset_dir / split_dir / ann_file.name
    else:
        split_dir = None
        output_certain_json_path = output_certain_dataset_dir / ann_file.name
        output_weak_json_path = output_weak_dataset_dir / ann_file.name
        output_uncertain_json_path = output_uncertain_dataset_dir / ann_file.name

    # 读取原始JSON
    with open(ann_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # ---------- 标签处理 ----------
    # 按 dataset_flag 将原始 category_id 映射到模型统一类别空间
    # 模型统一空间: 0=背景(跳过), 1=上行, 2=下行, 3=应急车道, 4=匝道,
    #              5=避险车道, 6=匝道导流区, 7=隔离带, 8=护栏(删除)
    id_mapping = DATASET_FLAG_CATEGORY_MAPPING.get(dataset_flag, ORIGINAL_CATEGORY_MAPPING)
    mapped_annotations = []
    skipped_count = 0
    for ann in data['annotations']:
        mapped_cat = id_mapping.get(ann['category_id'])
        if mapped_cat is None or mapped_cat == 0:
            skipped_count += 1
            continue
        mapped_ann = ann.copy()
        mapped_ann['category_id'] = mapped_cat
        mapped_annotations.append(mapped_ann)

    images_dict = {img['id']: img for img in data['images']}
    annotations_by_image = defaultdict(list)
    for ann in mapped_annotations:
        annotations_by_image[ann['image_id']].append(ann)

    certain_annotations = []
    weak_annotations = []
    uncertain_annotations = []
    uncertain_records = []
    weak_assignment_records = []
    certain_image_ids = set()
    weak_image_ids = set()
    uncertain_image_ids = set()
    stats = {'deleted': 0, 'skipped': skipped_count, 'certain': 0, 'weak': 0, 'uncertain': 0, 'fused': 0, 'overlap_px': 0, 'restored': 0}

    for img_id, img_annotations in progress_iter(annotations_by_image.items(), "分类+融合", len(annotations_by_image)):
        img_info = images_dict.get(img_id, {})
        img_width = img_info.get('width', 1280)
        img_height = img_info.get('height', 720)
        up_seeds, down_seeds, divider_seeds = build_direction_seeds(img_annotations)
        divider_info = infer_divider_sides(up_seeds, down_seeds, divider_seeds)
        seed_method = (
            f"up_seeds={len(up_seeds)},down_seeds={len(down_seeds)},"
            f"divider_seeds={len(divider_seeds)},divider_x={divider_info[0]}"
        )

        divider_issue = detect_divider_direction_issue(up_seeds, down_seeds, divider_seeds)
        if divider_issue:
            img_has_uncertain = True
            stats['uncertain'] += len([ann for ann in img_annotations if ann['category_id'] != 8])
            uncertain_records.append({
                'image_id': img_id,
                'old_category_id': None,
                'old_category_name': '图级检查',
                'new_category_id': None,
                'new_category_name': None,
                'status': 'review_required',
                'reason': divider_issue,
                'annotation_center': None,
                'target_bbox': None,
                'seed_method': seed_method,
                'method': 'divider_direction_consistency'
            })
        else:
            img_has_uncertain = False

        img_merged_anns = []
        img_has_weak = False

        for ann in [] if divider_issue else img_annotations:
            new_cat_id, status, reason, meta = classify_annotation(
                ann, up_seeds, down_seeds, divider_info, img_width, img_height
            )
            if new_cat_id is None:
                stats['deleted'] += 1
                continue

            certain_ann = ann.copy()
            certain_ann['category_id'] = new_cat_id

            if status in ('strong', 'weak_nonblocking'):
                img_merged_anns.append(certain_ann)
                stats['certain' if status == 'strong' else 'weak'] += 1
                if status == 'weak_nonblocking':
                    img_has_weak = True
                    record = {
                        'image_id': img_id,
                        'old_category_id': ann['category_id'],
                        'old_category_name': ORIGINAL_CATEGORIES.get(ann['category_id'], '未知'),
                        'new_category_id': new_cat_id,
                        'new_category_name': '上行' if new_cat_id == 1 else '下行',
                        'status': status,
                        'reason': reason,
                        'annotation_center': list(get_annotation_center(ann)),
                        'target_bbox': list(get_annotation_bbox(ann) or []),
                        'seed_method': seed_method
                    }
                    record.update(meta)
                    weak_assignment_records.append(record)
            else:
                img_has_uncertain = True
                stats['uncertain'] += 1
                record = {
                    'image_id': img_id,
                    'old_category_id': ann['category_id'],
                    'old_category_name': ORIGINAL_CATEGORIES.get(ann['category_id'], '未知'),
                    'new_category_id': new_cat_id,
                    'new_category_name': '上行' if new_cat_id == 1 else '下行',
                    'status': status,
                    'reason': reason,
                    'annotation_center': list(get_annotation_center(ann)),
                    'target_bbox': list(get_annotation_bbox(ann) or []),
                    'seed_method': seed_method
                }
                record.update(meta)
                uncertain_records.append(record)

        # 融合同类别相接的分割区域：分类完成后再做几何融合，
        # 使主线+相接匝道/应急车道等同类别区域成为一个 annotation；
        # 跨类别重叠只做截断(重叠像素判给距边界最远者)，不允许删除任何区域
        # (uncertain 图保留原始标注供人工修补，不融合)
        if img_merged_anns and not img_has_uncertain:
            img_merged_anns, fuse_info = fuse_image_annotations(img_merged_anns, img_width, img_height)
            stats['fused'] += fuse_info['fused']
            stats['overlap_px'] += fuse_info['overlap_px']
            stats['restored'] += fuse_info['restored']

        if img_has_uncertain:
            # uncertain 输出保留该图【除护栏外的全部原始标注】(含1/2上下行主线、7隔离带作为参考上下文)
            # 且保持原始 category_id，便于后期在 visual 里参照上下文手动修补
            for ann in img_annotations:
                if ann['category_id'] == 8:  # 护栏删除，与 certain 策略一致
                    continue
                original_ann = ann.copy()
                original_ann['image_id'] = img_id
                uncertain_annotations.append(original_ann)
            uncertain_image_ids.add(img_id)
        elif img_has_weak:
            for ann in img_merged_anns:
                ann['image_id'] = img_id
                weak_annotations.append(ann)
            weak_image_ids.add(img_id)
        else:
            for ann in img_merged_anns:
                ann['image_id'] = img_id
                certain_annotations.append(ann)
            if img_merged_anns:
                certain_image_ids.add(img_id)

    # ★ 修复：复制/软链接图片
    # 路径结构（图片和 JSON 分离）：
    # - 有子目录(train/val): output/数据集名/train/images/xxx.jpg (file_name 扁平如 "1.1.jpg")
    # - 无子目录: output/数据集名/images/隧道/xxx.jpg (file_name 含子目录如 "images/隧道/xxx.jpg")
    certain_images = []
    weak_images = []
    uncertain_images = []
    copied_count = 0

    for img in progress_iter(data['images'], "复制/链接图片", len(data['images'])):
        img_id = img['id']
        file_name = '/'.join(p.strip() for p in img['file_name'].split('/'))

        # 源图片路径: img_prefix + file_name
        src_img = img_prefix / file_name

        if not src_img.exists():
            print(f"  ⚠ 图片不存在: {src_img}")
            continue

        is_certain_image = img_id in certain_image_ids
        is_weak_image = img_id in weak_image_ids
        is_uncertain_image = img_id in uncertain_image_ids

        def output_image_path(output_dataset_dir):
            # ★ 修复：目标路径 - 图片放到 images/ 子目录，和 JSON 分离
            if has_subdir:
                # 有子目录: output/数据集名/train/images/xxx.jpg
                # file_name 是扁平的（如 "1.1.jpg"），放在 split_dir/images/ 下
                return output_dataset_dir / split_dir / "images" / file_name
            # 无子目录: output/数据集名/images/隧道/xxx.jpg
            # file_name 本身含子目录（如 "images/隧道/xxx.jpg"），放在数据集根下
            return output_dataset_dir / file_name

        def copy_to_bucket(output_dataset_dir, image_list):
            dst = output_image_path(output_dataset_dir)
            if USE_COPY:
                ok = copy_file_safe(src_img, dst)
            else:
                ok = create_symlink(src_img, dst)
            if ok:
                img_out = img.copy()
                img_out['file_name'] = file_name
                image_list.append(img_out)
                return 1
            return 0

        if is_certain_image:
            copied_count += copy_to_bucket(output_certain_dataset_dir, certain_images)
        elif is_weak_image:
            copied_count += copy_to_bucket(output_weak_dataset_dir, weak_images)

        # 不确定分类目录
        elif is_uncertain_image:
            copied_count += copy_to_bucket(output_uncertain_dataset_dir, uncertain_images)

    # ---------- 输出JSON ----------
    # ★ 空内容跳过: images 和 annotations 均为空时不输出该 JSON 文件
    certain_data = {
        'info': {
            'dataset_flag': dataset_flag,
            'category_mapping': MERGED_CATEGORY_MAPPING,
            'description': '3分类合并结果: 1=上行, 2=下行, 3=隔离带'
        },
        'images': certain_images,
        'annotations': certain_annotations,
        'categories': NEW_CATEGORIES
    }
    if certain_images or certain_annotations:
        output_certain_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_certain_json_path, 'w', encoding='utf-8') as f:
            json.dump(certain_data, f, ensure_ascii=False, indent=2)

    weak_data = {
        'info': {
            'dataset_flag': dataset_flag,
            'category_mapping': MERGED_CATEGORY_MAPPING,
            'description': '3分类合并结果(弱合并): 1=上行, 2=下行, 3=隔离带'
        },
        'images': weak_images,
        'annotations': weak_annotations,
        'categories': NEW_CATEGORIES
    }
    if weak_images or weak_annotations:
        output_weak_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_weak_json_path, 'w', encoding='utf-8') as f:
            json.dump(weak_data, f, ensure_ascii=False, indent=2)

    uncertain_data = {
        'info': {
            'dataset_flag': dataset_flag,
            'category_mapping': DATASET_FLAG_CATEGORY_MAPPING.get(dataset_flag, ORIGINAL_CATEGORY_MAPPING),
            'description': '8分类标注(存疑未合并), category_id已按dataset_flag映射到模型统一类别空间'
        },
        'images': uncertain_images,
        'annotations': uncertain_annotations,
        'categories': ORIGINAL_COCO_CATEGORIES
    }
    if uncertain_images or uncertain_annotations:
        output_uncertain_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_uncertain_json_path, 'w', encoding='utf-8') as f:
            json.dump(uncertain_data, f, ensure_ascii=False, indent=2)

    if weak_assignment_records:
        weak_records_path = output_weak_json_path.parent / 'weak_assignment_records.json'
        weak_records_path.parent.mkdir(parents=True, exist_ok=True)
        with open(weak_records_path, 'w', encoding='utf-8') as f:
            json.dump({'total': len(weak_assignment_records), 'records': weak_assignment_records}, f, ensure_ascii=False, indent=2)

    return {
        'dataset': dataset_name,
        'dataset_flag': dataset_flag,
        'has_subdir': has_subdir,
        'split': split_dir if has_subdir else ann_file.name,
        'ann_file': str(ann_file),
        'img_prefix': str(img_prefix),
        'certain_annotations': len(certain_annotations),
        'weak_annotations': len(weak_annotations),
        'uncertain_annotations': len(uncertain_annotations),
        'deleted': stats['deleted'],
        'skipped': stats['skipped'],
        'fused': stats['fused'],
        'overlap_resolved_px': stats['overlap_px'],
        'restored_regions': stats['restored'],
        'weak_assignments': stats['weak'],
        'certain_images': len(certain_images),
        'weak_images': len(weak_images),
        'uncertain_images': len(uncertain_images),
        'uncertain_count': len(uncertain_records),
        'uncertain_records': uncertain_records,
        'weak_assignment_records': weak_assignment_records,
        'copied_count': copied_count
    }


def main():
    parser = argparse.ArgumentParser(
        description="8分类合并为3分类，并融合同类别相接的分割区域(只截断不删除，输出互不重叠)")
    parser.add_argument('-o', '--output', default=None,
                        help='输出根目录，其下创建 certain/weak/uncertain 子目录 '
                             '(默认: highway_seg_merged，即脚本顶部 *_OUTPUT_BASE 常量)')
    args = parser.parse_args()

    if args.output:
        output_base = Path(args.output)
        certain_base = output_base / 'certain'
        weak_base = output_base / 'weak'
        uncertain_base = output_base / 'uncertain'
    else:
        certain_base = Path(CERTAIN_OUTPUT_BASE)
        weak_base = Path(WEAK_OUTPUT_BASE)
        uncertain_base = Path(UNCERTAIN_OUTPUT_BASE)

    # 输入路径只读保护: 输出目录不允许落在输入数据目录内，
    # 脚本对输入只做读取(标注JSON只读打开, 图片仅作复制/软链接源)
    data_bases_resolved = [Path(DATA_BASE).resolve(), Path(DATA_BASE_KAIYUN).resolve()]
    for base in (certain_base, weak_base, uncertain_base):
        resolved = base.resolve()
        for data_base_resolved in data_bases_resolved:
            if resolved == data_base_resolved or data_base_resolved in resolved.parents:
                raise ValueError(f"输出目录 {base} 落在输入数据目录 {data_base_resolved} 内，违反输入只读约束")

    certain_base.mkdir(parents=True, exist_ok=True)
    weak_base.mkdir(parents=True, exist_ok=True)
    uncertain_base.mkdir(parents=True, exist_ok=True)

    all_reports = []
    all_uncertain = []
    all_weak = []

    print("=" * 60)
    print("开始处理数据集...")
    print(f"模式: {'复制' if USE_COPY else '软链接'}")
    print(f"确定分类输出: {certain_base}")
    print(f"弱确定分类输出: {weak_base}")
    print(f"不确定分类输出: {uncertain_base}")
    print("=" * 60)

    for config_idx, config in enumerate(DATASET_CONFIGS, 1):
        if len(config) == 4:
            ann_file, img_prefix, dataset_name, dataset_flag = config
        elif len(config) == 3:
            ann_file, img_prefix, dataset_name = config
            dataset_flag = 9  # 默认flag
        else:
            # 兼容旧格式（二元组），自动推断 dataset_name
            ann_file, img_prefix = config
            img_prefix_path = Path(img_prefix)
            if img_prefix_path.name == "images":
                # 有子目录: 取 images 的父目录名作为数据集名
                dataset_name = img_prefix_path.parent.name
            else:
                dataset_name = img_prefix_path.name
            dataset_flag = 9  # 默认flag

        print(f"\n[{config_idx}/{len(DATASET_CONFIGS)}] 处理: {ann_file}")
        print(f"  图片路径: {img_prefix}")
        print(f"  数据集名: {dataset_name}")
        print(f"  dataset_flag: {dataset_flag}")
        report = process_dataset(ann_file, img_prefix, dataset_name, certain_base, weak_base, uncertain_base, dataset_flag)
        if report:
            all_reports.append(report)
            all_uncertain.extend(report.get('uncertain_records', []))
            all_weak.extend(report.get('weak_assignment_records', []))
            print(f"  ✓ 确定: {report['certain_annotations']} 标注, {report['certain_images']} 图片")
            print(f"  ✓ 弱确定: {report['weak_annotations']} 标注, {report['weak_images']} 图片")
            print(f"  ✓ 存疑: {report['uncertain_annotations']} 标注, {report['uncertain_images']} 图片")
            print(f"  ✓ 删除: {report['deleted']} (护栏)")
            print(f"  ✓ 跳过: {report['skipped']} (背景/未映射)")
            print(f"  ✓ 融合: {report['fused']} (同类别相接区域合并消除的冗余标注)")
            print(f"  ✓ 重叠仲裁: {report['overlap_resolved_px']} px 截断, 守恒恢复 {report['restored_regions']} 区域")
            print(f"  ✓ 图片: {report['copied_count']} 张")

    # 汇总
    summary = {
        'total_files': len(all_reports),
        'certain_annotations': sum(r['certain_annotations'] for r in all_reports),
        'weak_annotations': sum(r['weak_annotations'] for r in all_reports),
        'uncertain_annotations': sum(r['uncertain_annotations'] for r in all_reports),
        'weak_assignments': sum(r['weak_assignments'] for r in all_reports),
        'deleted': sum(r['deleted'] for r in all_reports),
        'skipped': sum(r['skipped'] for r in all_reports),
        'fused': sum(r['fused'] for r in all_reports),
        'overlap_resolved_px': sum(r['overlap_resolved_px'] for r in all_reports),
        'restored_regions': sum(r['restored_regions'] for r in all_reports),
        'certain_images': sum(r['certain_images'] for r in all_reports),
        'weak_images': sum(r['weak_images'] for r in all_reports),
        'uncertain_images': sum(r['uncertain_images'] for r in all_reports),
        'details': all_reports
    }

    with open(certain_base / 'processing_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if all_uncertain:
        with open(uncertain_base / 'uncertain_records.json', 'w', encoding='utf-8') as f:
            json.dump({'total': len(all_uncertain), 'records': all_uncertain}, f, ensure_ascii=False, indent=2)

    if all_weak:
        with open(weak_base / 'weak_assignment_records.json', 'w', encoding='utf-8') as f:
            json.dump({'total': len(all_weak), 'records': all_weak}, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("处理完成!")
    print(f"  确定分类: {summary['certain_annotations']} 标注, {summary['certain_images']} 图片")
    print(f"  弱确定分类: {summary['weak_annotations']} 标注, {summary['weak_images']} 图片")
    print(f"  不确定分类: {summary['uncertain_annotations']} 标注, {summary['uncertain_images']} 图片")
    print(f"  删除(护栏): {summary['deleted']}")
    print(f"  跳过(背景/未映射): {summary['skipped']}")
    print(f"  融合(同类别相接区域): {summary['fused']}")
    print(f"  重叠仲裁截断: {summary['overlap_resolved_px']} px, 守恒恢复: {summary['restored_regions']} 区域")
    print("=" * 60)


if __name__ == '__main__':
    main()
