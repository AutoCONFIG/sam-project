"""
SAM 3 Prediction / Inference Module
=====================================
Video inference dispatched by the unified YAML entry point.

Supports text prompts for open-vocabulary segmentation and tracking
across all frames of a video or image sequence.

Usage::

    python sam.py configs/predict/video_text.yaml
"""

import argparse
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

import cv2

from core.engine import (
    Sam3VideoPredictor,
    get_frames,
    save_frame_results,
    write_frames_to_temp_dir,
)
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
    DEFAULT_PREDICT_OUTPUT,
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
        description="SAM 3.1 视频推理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python sam.py configs/predict/video_text.yaml
    python -m commands.predict --config configs/predict/video_text.yaml --text "person"
    python -m commands.predict -c /path/to/sam3.1_multiplex.pt -i video.mp4 -t "person" --image-size 672
        """,
    )

    parser.add_argument("--config", type=str, default=None, help="YAML 配置文件路径")
    parser.add_argument("--checkpoint", "-c", type=str, default=None,
                        help="sam3.1_multiplex.pt 路径")
    parser.add_argument("--input", "-i", type=str, default=None,
                        help="输入视频文件或 jpg 帧目录")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="输出目录")
    parser.add_argument("--text", "-t", type=str, default=None,
                        help="文本提示, 如 'person'")
    parser.add_argument("--frame-index", type=int, default=None,
                        help="添加文本提示的帧索引 (默认 0)")
    parser.add_argument("--image-size", type=int, default=None,
                        help=f"推理分辨率, 须为 {IMAGE_SIZE_STEP} 的倍数; 16G 显存用 672 (默认 {DEFAULT_IMAGE_SIZE})")

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
                         help_true="保存 mask npz (默认)",
                         help_false="不保存 mask")
    set_boolean_argument(parser, "save_video", "save-video",
                         help_true="把 vis 合成 mp4",
                         help_false="不合成 mp4 (默认)")

    return parser.parse_args()


def args_to_config(args: argparse.Namespace) -> Dict[str, Any]:
    """将命令行参数转换为嵌套配置字典。"""
    config: Dict[str, Any] = {}

    model_cfg = config_from_args(
        args,
        plain=("checkpoint", "image_size", "frame_index"),
        boolean=("use_fa3", "use_rope_real", "compile"),
        rename={"checkpoint": "checkpoint", "image_size": "image_size", "frame_index": "frame_index"},
    )
    if model_cfg:
        config["model"] = model_cfg

    io_cfg = config_from_args(
        args,
        plain=("input", "output"),
        boolean=("save_vis", "save_masks", "save_video"),
    )
    if io_cfg:
        config["io"] = io_cfg

    prompt_cfg = config_from_args(args, plain=("text",))
    if prompt_cfg:
        config["prompt"] = prompt_cfg

    return config


# ─── Main prediction orchestration ──────────────────────────────────────────


def predict(config: Dict) -> None:
    """运行 SAM 3.1 视频推理。"""
    # Extract config values
    checkpoint = get_nested_value(config, "model", "checkpoint")
    image_size = get_nested_value(config, "model", "image_size", default=DEFAULT_IMAGE_SIZE)
    use_fa3 = get_nested_value(config, "model", "use_fa3", default=DEFAULT_USE_FA3)
    use_rope_real = get_nested_value(config, "model", "use_rope_real", default=DEFAULT_USE_ROPE_REAL)
    compile_model = get_nested_value(config, "model", "compile", default=DEFAULT_COMPILE)

    input_path = get_nested_value(config, "io", "input")
    output_path = get_nested_value(config, "io", "output", default=DEFAULT_PREDICT_OUTPUT)
    save_vis = get_nested_value(config, "io", "save_vis", default=DEFAULT_SAVE_VIS)
    save_masks = get_nested_value(config, "io", "save_masks", default=DEFAULT_SAVE_MASKS)
    save_video = get_nested_value(config, "io", "save_video", default=DEFAULT_SAVE_VIDEO)

    text = get_nested_value(config, "prompt", "text")
    frame_index = get_nested_value(
        config, "model", "frame_index",
        default=get_nested_value(config, "prompt", "frame_index", default=DEFAULT_FRAME_INDEX),
    )

    # Validate
    if not checkpoint:
        raise ValueError("--checkpoint 或配置 model.checkpoint 是必需的")
    if not input_path:
        raise ValueError("--input 或配置 io.input 是必需的")
    if not text:
        raise ValueError("--text 或配置 prompt.text 是必需的")
    if image_size % IMAGE_SIZE_STEP != 0:
        raise ValueError(
            f"image_size 必须是 {IMAGE_SIZE_STEP} 的倍数 (得到 {image_size})"
        )

    out_dir = Path(output_path)
    if save_vis:
        (out_dir / "vis").mkdir(parents=True, exist_ok=True)
    if save_masks:
        (out_dir / "masks").mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("SAM 3.1 视频推理")
    print(f"{'='*60}")
    print(f"模型权重: {checkpoint}")
    print(f"推理分辨率: {image_size} (backbone ViT 用预训练位置编码, tile_abs_pos 自动适配)")
    print(f"Flash Attention 3: {use_fa3}")
    print(f"实数 RoPE: {use_rope_real}")
    print(f"torch.compile: {compile_model}")
    print(f"文本提示: '{text}' @ 帧 {frame_index}")
    print(f"{'='*60}\n")

    # Load frames
    print(f"加载视频/帧: {input_path}")
    frames, fps, n = get_frames(input_path)
    print(f"共 {n} 帧, fps={fps}")
    if n == 0:
        print("错误: 没有帧可处理")
        sys.exit(1)

    # Build model
    print(f"构建模型 (use_fa3={use_fa3})...")
    engine = Sam3VideoPredictor(
        checkpoint=checkpoint,
        image_size=image_size,
        use_fa3=use_fa3,
        use_rope_real=use_rope_real,
        compile=compile_model,
    )
    print(f"模型构建完成 (image_size={image_size})")

    # Run inference
    # SAM3 start_session needs a disk path, so write frames to a temp dir
    tmp_dir = write_frames_to_temp_dir(frames)
    try:
        print(f"帧已写入临时目录: {tmp_dir}")

        session_id = engine.start_session(resource_path=tmp_dir, offload_video_to_cpu=True)
        print(f"session: {session_id} (offload_video_to_cpu=True)")

        # Add text prompt
        outputs = engine.add_text_prompt(session_id, frame_index, text)
        n_obj = len(outputs.get("out_obj_ids", []))
        print(f"帧 {frame_index} 检测到 {n_obj} 个对象 '{text}'")

        # Save prompt frame result
        save_frame_results(frame_index, frames[frame_index], outputs, out_dir, save_vis, save_masks)

        # Propagate to all frames
        print("传播到全部帧 ...")
        total = n_obj
        for fi, outputs in engine.propagate(session_id):
            if fi >= len(frames) or fi == frame_index:
                continue
            save_frame_results(fi, frames[fi], outputs, out_dir, save_vis, save_masks)
            n_obj_i = len(outputs.get("out_obj_ids", []))
            total += n_obj_i
            if fi % 20 == 0 or fi == n - 1:
                print(f"  帧 {fi}/{n-1}: {n_obj_i} 对象")

        engine.close_session(session_id)

        print(f"\n完成。总对象-帧数: {total}")
        if save_vis:
            print(f"可视化: {out_dir / 'vis'}")
        if save_masks:
            print(f"mask:   {out_dir / 'masks'}")

        # Optionally compose video
        if save_video and fps:
            vis_files = sorted((out_dir / "vis").glob("*.jpg"))
            if vis_files:
                h, w = frames[0].shape[:2]
                vw = cv2.VideoWriter(
                    str(out_dir / "result.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps, (w, h),
                )
                for vf in vis_files:
                    vw.write(cv2.imread(str(vf)))
                vw.release()
                print(f"视频: {out_dir / 'result.mp4'}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


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
