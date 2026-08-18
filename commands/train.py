"""
SAM 3 Training Module
=====================
Forwards training requests to the SAM 3 Hydra-based training system.

SAM 3 training uses Hydra configuration management with configs in
``sam3/sam3/train/configs/``. This module wraps the entry point
``sam3/sam3/train/train.py -c <config>`` via subprocess, which is the
most reliable approach given Hydra's ``initialize_config_module`` global
state constraints.

Usage::

    python sam.py configs/train/roboflow_finetune.yaml
"""

import argparse
import subprocess
import sys
import traceback
from typing import Any, Dict

from utils.config import (
    get_nested_value,
    load_yaml_config,
    setup_sam3_path,
)
from utils.config import SAM3_ROOT

setup_sam3_path()


# ─── Argument parser ────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SAM 3 训练 (转发到 Hydra 训练系统)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python sam.py configs/train/roboflow_finetune.yaml
    python -m commands.train --config configs/train/roboflow_finetune.yaml --num-gpus 2

Available Hydra configs are in sam3/sam3/train/configs/:
    roboflow_v100/roboflow_v100_full_ft_100_images.yaml
    odinw13/odinw_text_only_train.yaml
    (see sam3/README_TRAIN.md for full list)
        """,
    )

    parser.add_argument("--config", type=str, default=None, help="YAML 配置文件路径")
    parser.add_argument("--sam3-config", type=str, default=None,
                        help="SAM3 Hydra config 名 (如 roboflow_v100/roboflow_v100_full_ft_100_images.yaml)")
    parser.add_argument("--use-cluster", type=int, default=None, choices=[0, 1],
                        help="0=本地训练, 1=SLURM 集群")
    parser.add_argument("--partition", type=str, default=None, help="SLURM 分区名")
    parser.add_argument("--account", type=str, default=None, help="SLURM 账户名")
    parser.add_argument("--qos", type=str, default=None, help="SLURM QOS")
    parser.add_argument("--num-gpus", type=int, default=None, help="每节点 GPU 数")
    parser.add_argument("--num-nodes", type=int, default=None, help="节点数")

    return parser.parse_args()


# ─── Main training orchestration ────────────────────────────────────────────


def train(config: Dict) -> None:
    """转发训练请求到 SAM3 Hydra 训练系统。"""
    sam3_config = get_nested_value(config, "train", "config")
    if not sam3_config:
        raise ValueError(
            "--sam3-config 或配置 train.config 是必需的 "
            "(如 roboflow_v100/roboflow_v100_full_ft_100_images.yaml)"
        )

    use_cluster = get_nested_value(config, "train", "use_cluster")
    partition = get_nested_value(config, "train", "partition")
    account = get_nested_value(config, "train", "account")
    qos = get_nested_value(config, "train", "qos")
    num_gpus = get_nested_value(config, "train", "num_gpus")
    num_nodes = get_nested_value(config, "train", "num_nodes")

    train_script = SAM3_ROOT / "sam3" / "train" / "train.py"
    if not train_script.exists():
        raise FileNotFoundError(
            f"SAM3 训练脚本不存在: {train_script}\n"
            "请确保 sam3 子模块已正确初始化 (git submodule update --init)"
        )

    cmd = [sys.executable, str(train_script), "-c", sam3_config]

    if use_cluster is not None:
        cmd += ["--use-cluster", str(use_cluster)]
    if partition is not None:
        cmd += ["--partition", partition]
    if account is not None:
        cmd += ["--account", account]
    if qos is not None:
        cmd += ["--qos", qos]
    if num_gpus is not None:
        cmd += ["--num-gpus", str(num_gpus)]
    if num_nodes is not None:
        cmd += ["--num-nodes", str(num_nodes)]

    print(f"\n{'='*60}")
    print("SAM 3 训练")
    print(f"{'='*60}")
    print(f"Hydra config: {sam3_config}")
    print(f"训练脚本: {train_script}")
    if use_cluster is not None:
        print(f"集群模式: {'SLURM' if use_cluster else '本地'}")
    if num_gpus is not None:
        print(f"GPU 数: {num_gpus}")
    if num_nodes is not None:
        print(f"节点数: {num_nodes}")
    print(f"{'='*60}\n")
    print(f"执行命令: {' '.join(cmd)}\n")

    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    try:
        config = {}
        if args.config:
            config = load_yaml_config(args.config)

        # CLI overrides YAML
        train_cfg: Dict[str, Any] = {}
        for field in ("sam3_config", "use_cluster", "partition", "account", "qos", "num_gpus", "num_nodes"):
            v = getattr(args, field, None)
            if v is not None:
                key = "config" if field == "sam3_config" else field
                train_cfg[key] = v
        if train_cfg:
            config.setdefault("train", {}).update(train_cfg)

        train(config)
    except KeyboardInterrupt:
        print("\n训练被用户中断。")
        sys.exit(130)
    except subprocess.CalledProcessError as e:
        print(f"\n{'='*60}\n训练失败 (exit code {e.returncode})\n{'='*60}")
        sys.exit(e.returncode)
    except Exception as e:
        print(f"\n{'='*60}\n错误: {e}\n{'='*60}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
