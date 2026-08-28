"""
SAM 3 Prediction / Inference Module
=====================================
Video / image-sequence inference dispatched by the unified YAML entry point.

Supports text prompts for open-vocabulary segmentation and tracking. Output
mirrors the input form: a video in yields a video out (+ mask video + labels),
an image directory in yields a parallel image directory (+ mask images +
labels). COCO and YOLO (det + seg) labels are exported per input.

Usage::

    python sam.py configs/predict/video_text.yaml
"""

import argparse
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from core.engine import (
    Sam3VideoPredictor,
    get_frames,
    write_frames_to_temp_dir,
)
from core.io_dispatch import (
    InputUnit,
    OutputTree,
    discover_inputs,
    resolve_output_tree,
)
from core.labels import LabelExporter
from core.visualization import build_class_colors, draw_mask_overlay
from utils.config import (
    config_from_args,
    get_nested_value,
    load_yaml_config,
    merge_configs,
    set_boolean_argument,
    setup_sam3_path,
    to_bool,
)
from utils.constants import (
    DEFAULT_COMPILE,
    DEFAULT_FRAME_INDEX,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_LABEL_FORMATS,
    DEFAULT_MAX_NUM_OBJECTS,
    DEFAULT_MODEL_VERSION,
    DEFAULT_MULTIPLEX_COUNT,
    DEFAULT_PREDICT_OUTPUT,
    DEFAULT_SAVE_LABELS,
    DEFAULT_SAVE_MASKS,
    DEFAULT_SAVE_VIDEO,
    DEFAULT_SAVE_VIS,
    DEFAULT_USE_FA3,
    DEFAULT_USE_ROPE_REAL,
    IMAGE_SIZE_STEP,
)

setup_sam3_path()


# ─── Argument parser ────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SAM 3 / 3.1 视频推理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python sam.py configs/predict/video_text.yaml
    python -m commands.predict --config configs/predict/video_text.yaml --text "person"
    python -m commands.predict -c /path/to/sam3.1_multiplex.pt --version sam3.1 -i video.mp4 -t "person" --image-size 672
    python -m commands.predict -c /path/to/sam3.pt --version sam3 -i video.mp4 -t "person"
        """,
    )

    parser.add_argument("--config", type=str, default=None, help="YAML 配置文件路径")
    parser.add_argument("--version", type=str, default=None,
                        choices=["sam3", "sam3.1"],
                        help=f"模型版本 (默认 {DEFAULT_MODEL_VERSION}; sam3 固定 image_size=1008)")
    parser.add_argument("--checkpoint", "-c", type=str, default=None,
                        help="模型权重路径 (sam3.1_multiplex.pt 或 sam3.pt)")
    parser.add_argument("--finetune-ckpt", type=str, default=None,
                        help="微调 checkpoint (训练链产出的 image model 权重, 仅 sam3.1; 加载进 detector, 基础权重仍由 checkpoint 提供)")
    parser.add_argument("--input", "-i", type=str, default=None,
                        help="输入视频文件或图片帧目录")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="输出目录")
    parser.add_argument("--text", "-t", type=str, default=None,
                        help="文本提示, 如 'person'")
    parser.add_argument("--frame-index", type=int, default=None,
                        help="添加文本提示的帧索引 (默认 0)")
    parser.add_argument("--image-size", type=int, default=None,
                        help=f"推理分辨率, 须为 {IMAGE_SIZE_STEP} 的倍数; 16G 显存用 672 (默认 {DEFAULT_IMAGE_SIZE})")
    parser.add_argument("--max-num-objects", type=int, default=None,
                        help=f"单 session 最多同时跟踪对象数 (仅 sam3.1, 默认 {DEFAULT_MAX_NUM_OBJECTS}; 多目标场景可调大, 显存随之涨)")
    parser.add_argument("--multiplex-count", type=int, default=None,
                        help=f"每个 multiplex 桶的对象容量 (仅 sam3.1, 默认 {DEFAULT_MULTIPLEX_COUNT}; 结构参数, 与预训练权重绑定, 勿改)")

    # ── 检测后处理阈值 (原后端硬编码, 现可调) ──
    parser.add_argument("--score-threshold", type=float, default=None,
                        help="检测置信度阈值 (sam3.1 默认 0.4, sam3 默认 0.5; 调低→更多检测, 调高→更精准)")
    parser.add_argument("--nms-thresh", type=float, default=None,
                        help="检测 NMS IoU 阈值 (默认 0.1; 调高→保留更多重叠框, 调低→去重更激进)")
    parser.add_argument("--new-det-thresh", type=float, default=None,
                        help="新对象确认阈值 (sam3.1 默认 0.65, sam3 默认 0.7; 跟踪中新增对象需达此分数)")

    set_boolean_argument(parser, "use_fa3", "use-fa3",
                         help_true="使用 Flash Attention 3 (需安装 flash-attn)",
                         help_false="不使用 Flash Attention (默认)")
    set_boolean_argument(parser, "use_rope_real", "use-rope-real",
                         help_true="使用实数 RoPE (默认)",
                         help_false="使用复数 RoPE")
    set_boolean_argument(parser, "compile", "compile",
                         help_true="启用 torch.compile (首次慢)",
                         help_false="不启用 torch.compile (默认)")

    set_boolean_argument(parser, "save_vis", "save-vis",
                         help_true="保存可视化叠加图 (默认)",
                         help_false="不保存可视化")
    set_boolean_argument(parser, "save_masks", "save-masks",
                         help_true="保存 mask (默认)",
                         help_false="不保存 mask")
    set_boolean_argument(parser, "save_video", "save-video",
                         help_true="视频输入时合成 mp4 (默认)",
                         help_false="不合成 mp4")
    set_boolean_argument(parser, "save_labels", "save-labels",
                         help_true="导出 COCO/YOLO 标签 (默认)",
                         help_false="不导出标签")
    parser.add_argument("--label-format", type=str, default=None, nargs="+",
                        choices=["coco", "yolo"],
                        help=f"标签格式, 可多选 (默认 {'/'.join(DEFAULT_LABEL_FORMATS)})")
    parser.add_argument("--device", type=str, default=None,
                        help="GPU 选卡, 如 '0' / '0,1' / '0,2,3'; 空或 'auto'=用 CUDA_VISIBLE_DEVICES/默认 GPU 0。"
                             "通过设置 CUDA_VISIBLE_DEVICES 生效, 须在模型构建前指定")
    parser.add_argument("--image-mode", type=str, default=None,
                        choices=["independent", "sequence"],
                        help="图片目录处理方式: independent=每张独立检测 (默认, 适合各自不同场景的图片集); "
                             "sequence=第 0 帧检测后跨帧传播跟踪 (适合视频抽帧序列); 视频文件输入忽略此字段")

    return parser.parse_args()


def args_to_config(args: argparse.Namespace) -> Dict[str, Any]:
    """将命令行参数转换为嵌套配置字典。"""
    config: Dict[str, Any] = {}

    model_cfg = config_from_args(
        args,
        plain=("version", "checkpoint", "image_size", "frame_index", "finetune_ckpt",
               "max_num_objects", "multiplex_count", "device"),
        boolean=("use_fa3", "use_rope_real", "compile"),
    )
    if model_cfg:
        config["model"] = model_cfg

    io_cfg = config_from_args(
        args,
        plain=("input", "output", "label_format", "image_mode"),
        boolean=("save_vis", "save_masks", "save_video", "save_labels"),
    )
    if io_cfg:
        config["io"] = io_cfg

    prompt_cfg = config_from_args(args, plain=("text",))
    if prompt_cfg:
        config["prompt"] = prompt_cfg

    predict_cfg = config_from_args(
        args,
        plain=("score_threshold", "nms_thresh", "new_det_thresh"),
    )
    if predict_cfg:
        # CLI 参数名 → 配置键名映射
        key_map = {
            "score_threshold": "score_threshold_detection",
            "nms_thresh": "det_nms_thresh",
            "new_det_thresh": "new_det_thresh",
        }
        config["predict"] = {key_map[k]: v for k, v in predict_cfg.items()}

    return config


# ─── Main prediction orchestration ──────────────────────────────────────────


def predict(config: Dict) -> None:
    """运行 SAM 3 / 3.1 视频/图像序列推理。

    输入形态决定输出形态: 视频进→视频出, 图片目录进→镜像结构图片目录出。
    每路输入额外输出 mask 和标签 (COCO + YOLO)。模型只构建一次, 顺序处理
    多个输入单元 (session 复用)。
    """
    # ── Extract config ──────────────────────────────────────────────────
    version = get_nested_value(config, "model", "version", default=DEFAULT_MODEL_VERSION)
    checkpoint = get_nested_value(config, "model", "checkpoint")
    finetune_ckpt = get_nested_value(config, "model", "finetune_ckpt")
    image_size = get_nested_value(config, "model", "image_size", default=DEFAULT_IMAGE_SIZE)
    use_fa3 = get_nested_value(config, "model", "use_fa3", default=DEFAULT_USE_FA3)
    use_rope_real = get_nested_value(config, "model", "use_rope_real", default=DEFAULT_USE_ROPE_REAL)
    compile_model = get_nested_value(config, "model", "compile", default=DEFAULT_COMPILE)
    max_num_objects = get_nested_value(config, "model", "max_num_objects", default=DEFAULT_MAX_NUM_OBJECTS)
    multiplex_count = get_nested_value(config, "model", "multiplex_count", default=DEFAULT_MULTIPLEX_COUNT)
    device = get_nested_value(config, "model", "device")

    # ── GPU 选卡 (在模型构建 / CUDA 初始化前设 CUDA_VISIBLE_DEVICES) ──
    # 后端 Sam3VideoPredictorMultiGPU 通过 torch.cuda.current_device() 选卡,
    # 受 CUDA_VISIBLE_DEVICES 控制。默认走 GPU 0; 多卡训练占满 0 号时, 用此
    # 字段指定空闲卡 (如 "3" / "0,1")。须在任何 CUDA 操作前设置才生效。
    import os as _os
    if device and str(device).strip().lower() not in ("", "auto", "cpu"):
        gpus = ",".join(g.strip() for g in str(device).split(",") if g.strip())
        _os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        print(f"GPU 选卡: CUDA_VISIBLE_DEVICES={gpus}")
    elif _os.environ.get("CUDA_VISIBLE_DEVICES"):
        print(f"GPU 选卡 (继承环境): CUDA_VISIBLE_DEVICES={_os.environ['CUDA_VISIBLE_DEVICES']}")


    # ── 检测后处理阈值 (原后端硬编码, 现可调) ──
    # sam3.1 默认 0.4/0.1/0.65, sam3 默认 0.5/0.1/0.7; 前端配置按版本给不同默认值
    version_default_thresh = {
        "sam3.1": (0.4, 0.1, 0.65),
        "sam3":   (0.5, 0.1, 0.7),
    }
    d_score, d_nms, d_newdet = version_default_thresh.get(version, (0.4, 0.1, 0.65))
    score_threshold_detection = get_nested_value(
        config, "predict", "score_threshold_detection", default=d_score)
    det_nms_thresh = get_nested_value(
        config, "predict", "det_nms_thresh", default=d_nms)
    new_det_thresh = get_nested_value(
        config, "predict", "new_det_thresh", default=d_newdet)

    input_path = get_nested_value(config, "io", "input")
    output_path = get_nested_value(config, "io", "output", default=DEFAULT_PREDICT_OUTPUT)
    save_vis = get_nested_value(config, "io", "save_vis", default=DEFAULT_SAVE_VIS)
    save_masks = get_nested_value(config, "io", "save_masks", default=DEFAULT_SAVE_MASKS)
    save_video = get_nested_value(config, "io", "save_video", default=DEFAULT_SAVE_VIDEO)
    save_labels = get_nested_value(config, "io", "save_labels", default=DEFAULT_SAVE_LABELS)
    label_formats = get_nested_value(
        config, "io", "label_format", default=list(DEFAULT_LABEL_FORMATS),
    )
    image_mode = get_nested_value(config, "io", "image_mode", default="independent")

    text = get_nested_value(config, "prompt", "text")
    frame_index = get_nested_value(
        config, "model", "frame_index",
        default=get_nested_value(config, "prompt", "frame_index", default=DEFAULT_FRAME_INDEX),
    )

    # ── Parse class prompts ─────────────────────────────────────────────
    # prompt.text 统一作为类别列表处理。单类别写 ["person"], 多类别写
    # ["person", "car", ...]。多类别是逐类别独立推理 (SAM3 架构不支持一次
    # session 检测多个不同类别, 每次 text prompt 都 reset session)。
    # 为兼容旧的字符串写法, 字符串会被自动包成单元素列表。
    if isinstance(text, str):
        text = [text]
    classes: List[str] = [t.strip() for t in (text or []) if t and t.strip()]

    # ── Validate ────────────────────────────────────────────────────────
    if not checkpoint:
        raise ValueError("--checkpoint 或配置 model.checkpoint 是必需的")
    if not input_path:
        raise ValueError("--input 或配置 io.input 是必需的")
    if not classes or not classes[0]:
        raise ValueError("--text 或配置 prompt.text 是必需的")
    if image_size % IMAGE_SIZE_STEP != 0:
        raise ValueError(
            f"image_size 必须是 {IMAGE_SIZE_STEP} 的倍数 (得到 {image_size})"
        )
    # sam3 (base) 不支持 image_size 参数化, 后端固定 1008
    if version == "sam3" and image_size != 1008:
        raise ValueError(
            f"sam3 原版不支持自定义 image_size (后端固定 1008), 得到 {image_size}; "
            f"如需低分辨率请用 sam3.1"
        )

    # ── Scan inputs ─────────────────────────────────────────────────────
    units = discover_inputs(input_path)
    if not units:
        raise ValueError(f"输入路径下没有可处理的视频或图片: {input_path}")

    print(f"\n{'='*60}")
    print(f"SAM {version} 推理")
    print(f"{'='*60}")
    print(f"模型权重: {checkpoint}")
    if finetune_ckpt:
        print(f"微调权重: {finetune_ckpt} (加载进 detector)")
    if version == "sam3.1":
        print(f"推理分辨率: {image_size}")
    else:
        print(f"推理分辨率: 1008 (sam3 原版固定)")
    print(f"Flash Attention 3: {use_fa3} | 实数 RoPE: {use_rope_real} | torch.compile: {compile_model}")
    print(f"检测阈值: score={score_threshold_detection} nms_iou={det_nms_thresh} new_det={new_det_thresh}")
    print(f"类别: {classes} @ 帧 {frame_index}")
    print(f"输入单元: {len(units)} 个")
    for i, u in enumerate(units):
        kind_label = "视频" if u.kind == "video" else f"图片序列({len(u.frames)}张)"
        loc = str(u.source) if u.kind == "video" else str(u.source)
        print(f"  [{i+1}] {kind_label}: {loc}")
    print(f"{'='*60}\n")

    # ── Build model once ────────────────────────────────────────────────
    print(f"构建模型 (version={version}, use_fa3={use_fa3})...")
    engine = Sam3VideoPredictor(
        checkpoint=checkpoint,
        version=version,
        image_size=image_size,
        use_fa3=use_fa3,
        use_rope_real=use_rope_real,
        compile=compile_model,
        finetune_ckpt=finetune_ckpt,
        max_num_objects=max_num_objects,
        multiplex_count=multiplex_count,
        score_threshold_detection=score_threshold_detection,
        det_nms_thresh=det_nms_thresh,
        new_det_thresh=new_det_thresh,
    )
    print(f"模型构建完成\n")

    # ── Process each unit ───────────────────────────────────────────────
    for idx, unit in enumerate(units):
        tree = resolve_output_tree(output_path, unit)
        print(f"[{idx+1}/{len(units)}] 处理 {unit.kind}: {unit.source}")
        process_unit(
            engine=engine,
            unit=unit,
            tree=tree,
            classes=classes,
            frame_index=frame_index,
            image_mode=image_mode,
            save_vis=save_vis,
            save_masks=save_masks,
            save_video=save_video,
            save_labels=save_labels,
            label_formats=label_formats,
        )
        print()


def process_unit(
    engine: Sam3VideoPredictor,
    unit: InputUnit,
    tree: OutputTree,
    classes: List[str],
    frame_index: int,
    image_mode: str = "independent",
    save_vis: bool = False,
    save_masks: bool = False,
    save_video: bool = False,
    save_labels: bool = False,
    label_formats: List[str] = None,
) -> None:
    """Process a single input unit: load frames → session → per-class inference.

    Output mirrors the input kind:
      - video  → vis/<stem>.mp4, masks/<stem>.mp4, masks_npz/*.npz, labels/
      - image_seq → vis/*.jpg, masks/*.png, masks_npz/*.npz, labels/

    image_mode (仅 image_seq, 视频文件忽略):
      - independent: 每张图独立检测 (add_text_prompt 逐帧调用), 不做跨帧传播。
        适合各自不同场景的图片集 (如高速公路截图)。
      - sequence: 第 0 帧检测后 propagate 跨帧传播跟踪, 适合视频抽帧序列。

    Multi-class: SAM3 cannot detect multiple distinct classes in one session
    (each text prompt resets the session). So each class runs its own
    inference, reusing ONE session so frames are loaded only once; between
    classes we reset_session (clears prompt/tracker, keeps frames) to avoid
    re-writing temp jpgs and re-loading frames. Backbone features are still
    recomputed per class (reset clears the feature cache) — that's inherent
    to the architecture. Object ids are offset per class to avoid collisions.
    Single class is just the N=1 case of this loop.
    """
    # ── Load frames ─────────────────────────────────────────────────────
    if unit.kind == "video":
        frames, fps, n = get_frames(str(unit.source))
        if not fps:
            fps = 25.0
    else:  # image_seq
        frames, _fps, n = get_frames(str(unit.source))
        fps = None

    if n == 0:
        print("  跳过: 没有帧")
        return
    print(f"  共 {n} 帧, fps={fps}")

    h, w = frames[0].shape[:2]

    # ── Label exporter for this unit (handles multiple classes) ─────────
    # Use the first class as default; add_frame() takes an explicit class_name
    # so each class's objects land under the right category.
    exporter = LabelExporter(class_name=classes[0], predefined_classes=classes) if save_labels else None

    # ── Run inference ───────────────────────────────────────────────────
    # frame_results[fi] = list of (class_name, obj_id, mask, box, prob)
    frame_results: Dict[int, List[tuple]] = {i: [] for i in range(n)}

    tmp_dir = write_frames_to_temp_dir(frames)
    try:
        session_id = engine.start_session(resource_path=tmp_dir, offload_video_to_cpu=True)

        # 是否跨帧传播: 视频文件始终传播; 图片序列看 image_mode
        do_propagate = (unit.kind == "video") or (image_mode == "sequence")

        for ci, cls in enumerate(classes):
            if ci > 0:
                # reset before switching to a new text class (SAM3 requires
                # reset_state between text prompts, else results are wrong)
                engine.reset_session(session_id)

            if do_propagate:
                # 传播模式: 第 0 帧 (frame_index) 检测 + propagate 跨帧跟踪
                outputs = engine.add_text_prompt(session_id, frame_index, cls)
                n_obj = len(outputs.get("out_obj_ids", []))
                print(f"  [{ci+1}/{len(classes)}] '{cls}': 帧 {frame_index} 检测到 {n_obj} 个对象")
                cls_outputs: Dict[int, dict] = {frame_index: outputs}
                for fi, out in engine.propagate(session_id):
                    if fi == frame_index:
                        continue
                    cls_outputs[fi] = out
            else:
                # 独立模式: 每帧各自 add_text_prompt 检测, 不跨帧传播
                # (图片之间无时序关系, 传播会把第 0 帧对象强行跟踪到不相关帧)
                cls_outputs = {}
                for fi in range(n):
                    outputs = engine.add_text_prompt(session_id, fi, cls)
                    cls_outputs[fi] = outputs
                    if fi % 20 == 0 or fi == n - 1:
                        n_obj = len(outputs.get("out_obj_ids", []))
                        print(f"  [{ci+1}/{len(classes)}] '{cls}': 帧 {fi}/{n-1} 检测到 {n_obj} 个对象")

            for fi, out in cls_outputs.items():
                if fi >= n:
                    continue
                obj_ids = out.get("out_obj_ids", [])
                masks = out.get("out_binary_masks", [])
                boxes = _denormalize_boxes(out.get("out_boxes_xywh", []), w, h)
                probs = out.get("out_probs", [])
                # offset obj_id by class index to avoid cross-class collisions
                id_offset = ci * 100000
                for i, oid in enumerate(obj_ids):
                    frame_results[fi].append((
                        cls,
                        (int(oid) if oid is not None else i) + id_offset,
                        masks[i] if i < len(masks) else None,
                        boxes[i] if i < len(boxes) else None,
                        float(probs[i]) if i < len(probs) else 0.0,
                    ))
            print(f"  [{ci+1}/{len(classes)}] '{cls}' 完成")
        engine.close_session(session_id)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # ── Write per-frame outputs ─────────────────────────────────────────
    class_colors = build_class_colors(classes)  # 每类一色, 全程稳定
    vis_frames: List[np.ndarray] = []
    mask_frames: List[np.ndarray] = []
    total = 0

    for fi in sorted(frame_results.keys()):
        if fi >= n or not frame_results[fi]:
            continue
        frame_rgb = frames[fi]
        # split per-class results for this frame into parallel lists
        cls_names, obj_ids, masks, boxes, probs = [], [], [], [], []
        for cn, oid, m, bx, pr in frame_results[fi]:
            cls_names.append(cn)
            obj_ids.append(oid)
            masks.append(m)
            boxes.append(bx)
            probs.append(pr)
        total += len(obj_ids)

        # labels — register objects grouped by class name
        if exporter is not None:
            name = f"{fi:06d}.jpg"
            for cn in dict.fromkeys(cls_names):  # unique, order-preserving
                idxs = [i for i, c in enumerate(cls_names) if c == cn]
                exporter.add_frame(
                    fi, name, h, w,
                    [obj_ids[i] for i in idxs],
                    [masks[i] for i in idxs],
                    [boxes[i] for i in idxs],
                    [probs[i] for i in idxs],
                    class_name=cn,
                )

        # mask image (color label map) + npz
        if save_masks:
            mask_img = _render_label_map(obj_ids, masks, h, w)
            np.savez_compressed(
                tree.npz / f"{fi:06d}.npz",
                label_map=mask_img,
                meta=np.array([
                    {"obj_id": obj_ids[i],
                     "class": cls_names[i],
                     "score": probs[i],
                     "box_xywh": [float(v) for v in boxes[i]] if boxes[i] else None}
                    for i in range(len(obj_ids))
                ], dtype=object),
            )
            if unit.kind == "video":
                mask_frames.append(mask_img)
            else:
                cv2.imwrite(str(tree.masks / f"{fi:06d}.png"), mask_img)

        # vis image (按类别着色, mask 半透明叠加 + 检测框 + 标签置信度)
        if save_vis:
            vis = draw_mask_overlay(frame_rgb, masks, cls_names, class_colors,
                                    boxes=boxes, probs=probs)
            if unit.kind == "video":
                vis_frames.append(vis)
            else:
                cv2.imwrite(str(tree.vis / f"{fi:06d}.jpg"), vis)

        if fi % 20 == 0 or fi == n - 1:
            print(f"  帧 {fi}/{n-1}: {len(obj_ids)} 对象")

    # ── Compose videos (video input only) ───────────────────────────────
    if unit.kind == "video":
        if save_video and save_vis and vis_frames:
            _write_video(tree.vis / f"{tree.stem}.mp4", vis_frames, fps)
            print(f"  可视化视频: {tree.vis / f'{tree.stem}.mp4'}")
        if save_video and save_masks and mask_frames:
            _write_video(tree.masks / f"{tree.stem}.mp4", mask_frames, fps)
            print(f"  mask 视频: {tree.masks / f'{tree.stem}.mp4'}")
    else:
        if save_vis:
            print(f"  可视化: {tree.vis}")
        if save_masks:
            print(f"  mask: {tree.masks}")

    if save_masks:
        print(f"  mask npz: {tree.npz}")

    # ── Write labels ────────────────────────────────────────────────────
    if exporter is not None:
        if "coco" in label_formats:
            coco_path = tree.labels / f"{tree.stem}_coco.json"
            exporter.write_coco(coco_path)
            print(f"  COCO 标签: {coco_path}")
        if "yolo" in label_formats:
            exporter.write_yolo(tree.labels / f"{tree.stem}_yolo")
            print(f"  YOLO 标签: {tree.labels / f'{tree.stem}_yolo'}")

    print(f"  完成。总对象-帧数: {total}")


# ─── Helpers ──────────────────────────────────────────────────────────────


def _denormalize_boxes(boxes, width: int, height: int) -> list:
    """Convert SAM3's normalized [0,1] xywh boxes to absolute pixels.

    SAM3 returns ``out_boxes_xywh`` as normalized coordinates; downstream
    (vis drawing, npz, labels) expects pixel coordinates. Boxes already in
    pixel space (any value > 1) are passed through unchanged.
    """
    out = []
    for b in boxes:
        x, y, w, h = (float(v) for v in b)
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and w <= 1.0 and h <= 1.0:
            out.append([x * width, y * height, w * width, h * height])
        else:
            out.append([x, y, w, h])
    return out


def _render_label_map(obj_ids, masks, h: int, w: int) -> np.ndarray:
    """Render a color label-map image (BGR) from per-object masks.

    Background is black; each object id gets a distinct color.
    """
    label_map = np.zeros((h, w, 3), dtype=np.uint8)
    for i, (oid, m) in enumerate(zip(obj_ids, masks)):
        m_arr = np.asarray(m).astype(bool)
        if m_arr.shape != (h, w):
            m_arr = cv2.resize(m_arr.astype(np.uint8), (w, h),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
        real_id = int(oid) if oid is not None else i
        # deterministic color from id
        color = (
            (real_id * 47) % 256,
            (real_id * 97) % 256,
            (real_id * 173) % 256,
        )
        label_map[m_arr] = color
    return label_map


def _write_video(path: Path, frames_bgr: List[np.ndarray], fps: float) -> None:
    """Write a list of BGR frames to an mp4 file."""
    if not frames_bgr:
        return
    h, w = frames_bgr[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for fr in frames_bgr:
        vw.write(fr)
    vw.release()


def main():
    args = parse_args()
    try:
        config = {}
        if args.config:
            config = load_yaml_config(args.config)
        cli_config = args_to_config(args)
        config = merge_configs(config, cli_config)
        predict(config)
    except KeyboardInterrupt:
        print("\n推理被用户中断。")
        sys.exit(130)
    except Exception as e:
        print(f"\n{'='*60}\n错误: {e}\n{'='*60}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
