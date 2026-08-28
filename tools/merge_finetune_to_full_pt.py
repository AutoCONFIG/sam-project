#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
微调权重 → 完整推理 pt 合并工具
============================================================
背景: SAM3 训练链路 (build_sam3_image_model) 只保存 image model (detector)
权重, 不含 tracker; 推理时需要 pretrain/sam3/sam3.pt 提供 tracker + detector
基底, 再用 finetune_ckpt 覆盖 detector。这给部署/迁移带来不便 —— 推理依赖
两个文件, 无法像 YOLO 那样单 pt 部署。

本工具把两者合并成单个完整推理 pt (顶层平铺 detector.* + tracker.*, 与
pretrain/sam3/sam3.pt 同结构), 推理 yaml 只需 checkpoint 指向它, 不再需要
finetune_ckpt。合并规则:
  - base pt (pretrain/sam3/sam3.pt 或 sam3.1_multiplex.pt): 提供完整模型
    (detector + tracker) 作基底;
  - finetune ckpt (训练链产物, 含 {"model": ...} 包裹): 取 model state_dict,
    丢 RoPE buffer (与分辨率绑定, 推理时由模型按 image_size 重算),
    每键加 "detector." 前缀后覆盖 base 的对应 detector 键;
  - 训练没有的 detector 键 (如交互式 neck sam2_convs / interactive_convs,
    训练链路不存在该模块) 保留 base 原值;
  - tracker 全部保留 base 原值 (训练不训 tracker)。

适用: sam3 与 sam3.1 (两者完整 pt 都是平铺 detector.* + tracker.* 结构)。

用法:
  python tools/merge_finetune_to_full_pt.py \
      --base pretrain/sam3/sam3.pt \
      --finetune runs/train/highway_road_sam3/202608261/checkpoints/checkpoint_best.pt \
      --output pretrain/sam3/highway_road_sam3_finetune.pt

  # 推理 yaml (单 pt, 无需 finetune_ckpt):
  #   model:
  #     version: sam3
  #     checkpoint: pretrain/sam3/highway_road_sam3_finetune.pt
"""
import argparse
from pathlib import Path

import torch


# RoPE 位置编码 buffer 与分辨率绑定, 一律丢掉 —— 推理时由模型按当前
# image_size 预计算 (与 core/engine.py _load_finetune_ckpt 同一约定)
ROPE_SUFFIXES = ("freqs_cis", "freqs_cis_real", "freqs_cis_imag")


def load_state_dict(path: str) -> dict:
    """加载 pt; 训练产物含 {"model": ...} 包裹则取内层; base pt 顶层平铺则原样返回。"""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(ck, dict) and "model" in ck and isinstance(ck["model"], dict) \
            and not any(k.startswith(("detector.", "tracker.")) for k in ck.keys()):
        # 训练产物: 顶层有 model/optimizer/epoch... 包裹, 取 model
        ck = ck["model"]
    return ck


def merge(base_path: str, finetune_path: str, output_path: str) -> None:
    base = load_state_dict(base_path)
    ft = load_state_dict(finetune_path)

    # 丢 RoPE buffer
    ft = {k: v for k, v in ft.items() if not k.endswith(ROPE_SUFFIXES)}

    base_keys = set(base.keys())
    det_keys = {k for k in base_keys if k.startswith("detector.")}
    trk_keys = {k for k in base_keys if k.startswith("tracker.")}

    # 训练产物键加 detector. 前缀 → 对应 base 的 detector.* 键
    ft_prefixed = {f"detector.{k}": v for k, v in ft.items()}
    ft_keys = set(ft_prefixed.keys())

    covered = ft_keys & det_keys          # 训练覆盖的 detector 键
    ft_extra = ft_keys - det_keys         # 训练有但 base detector 没有 (异常)
    det_kept = det_keys - ft_keys         # base 有但训练没有 (保留原值, 如 sam2_convs)

    if ft_extra:
        print(f"  警告: {len(ft_extra)} 个训练键在 base detector 中不存在 (前 5 个):")
        for k in sorted(ft_extra)[:5]:
            print(f"    {k}")

    # 形状校验 (覆盖的键必须形状一致, 否则 load_state_dict 会崩)
    shape_mismatch = [k for k in covered
                      if base[k].shape != ft_prefixed[k].shape]
    if shape_mismatch:
        print(f"  错误: {len(shape_mismatch)} 个覆盖键形状不一致, 中止:")
        for k in shape_mismatch[:10]:
            print(f"    {k}: base={tuple(base[k].shape)} ft={tuple(ft_prefixed[k].shape)}")
        raise SystemExit(1)

    # 合并: base 全量 + 训练覆盖 detector 部分
    merged = dict(base)
    merged.update(ft_prefixed)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, output_path)

    # 报告
    print(f"base   : {base_path}")
    print(f"        detector.* = {len(det_keys)}, tracker.* = {len(trk_keys)}, 总 = {len(base)}")
    print(f"finetune: {finetune_path}")
    print(f"        (丢 RoPE 后) {len(ft)} 键 → 加 detector. 前缀")
    print(f"覆盖   : {len(covered)} 个 detector 键被微调权重替换")
    print(f"保留   : {len(det_kept)} 个 detector 键保留 base 原值 (训练未含, 如交互式 neck)")
    print(f"        {len(trk_keys)} 个 tracker 键保留 base 原值 (训练不训 tracker)")
    print(f"输出   : {output_path}  (总 {len(merged)} 键)")


def main():
    p = argparse.ArgumentParser(
        description="合并 base 完整 pt + 训练微调权重 → 单个完整推理 pt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--base", required=True,
                   help="基础完整权重 (pretrain/sam3/sam3.pt 或 sam3.1_multiplex.pt), 提供 tracker + detector 基底")
    p.add_argument("--finetune", required=True,
                   help="训练链产物 (runs/train/.../checkpoints/checkpoint[_best].pt), 含 model state_dict")
    p.add_argument("--output", required=True,
                   help="输出单 pt 路径 (如 pretrain/sam3/highway_road_sam3_finetune.pt)")
    args = p.parse_args()

    print("=" * 60)
    print("合并微调权重 → 完整推理 pt")
    print("=" * 60)
    merge(args.base, args.finetune, args.output)
    print("=" * 60)
    print(f"完成。推理 yaml 用法:")
    print(f"  model:")
    print(f"    version: sam3   # (或 sam3.1, 取决于 --base)")
    print(f"    checkpoint: {args.output}")
    print(f"  (无需 finetune_ckpt)")


if __name__ == "__main__":
    main()
