"""
SAM 3 Training Module
=====================
Forwards training requests to the SAM 3 Hydra-based training system.

SAM 3 training uses Hydra configuration management with configs in
``sam3/sam3/train/configs/``. This module wraps the entry point
``sam3/sam3/train/train.py -c <config>`` via subprocess, which is the
most reliable approach given Hydra's ``initialize_config_module`` global
state constraints.

Frontend config is split into independent YAMLs (see configs/):

- ``configs/train/xxx.yaml``    — train entry: references model/data configs,
  training knobs (train.*) and output dir (output.path)
- ``configs/models/xxx.yaml``   — model network config: which Hydra config
  (``hydra_config``, inside the sam3 submodule), ``pretrained``,
  ``resolution``, and network-related ``overrides``
- ``configs/datasets/xxx.yaml`` — dataset root + COCO split layout,
  translated into Hydra overrides (``paths.dataset_root``,
  ``trainer.data.*.dataset.img_folder/ann_file``)
- ``configs/export/xxx.yaml``   — ONNX export (mode: export, separate command)

The Hydra config must live inside the sam3 submodule
(``sam3/sam3/train/configs/``) — Hydra requires configs to live inside the
``sam3.train`` module. The ready-made template for custom COCO datasets is
``sam3/sam3/train/configs/custom_image_ft.yaml``.

Common knobs (``model.pretrained`` / ``model.resolution`` / ``train.batch`` /
``train.epochs`` / ``train.device``) are translated into Hydra overrides, and
``train.overrides`` passes any raw Hydra override through verbatim (the
backend ``train.py`` forwards them to ``hydra.compose``). ``paths.bpe_path``
is always injected as an absolute path into the submodule's assets.

Usage::

    python sam.py configs/train/roboflow_finetune.yaml
"""

import argparse
import os

import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.config import (
    get_nested_value,
    load_yaml_config,
    setup_sam3_path,
)
from utils.config import PROJECT_ROOT, SAM3_ROOT
from utils.constants import IMAGE_SIZE_STEP

setup_sam3_path()

# Hydra 配置根目录: 后端 train.py 用 initialize_config_module("sam3.train"),
# 传给 -c 的 config 名最终必须相对于此目录
HYDRA_CONFIG_ROOT = SAM3_ROOT / "sam3" / "train"

# BPE tokenizer 资源 (子模块内, 注入为绝对路径)
BPE_PATH = SAM3_ROOT / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"

# 数据集 YAML 注入依赖 Hydra 配置里的标准键 (paths.dataset_root /
# trainer.data.{train,val}.dataset.*), 即模板 sam3/sam3/train/configs/
# custom_image_ft.yaml 的结构; 后端自带配置 (roboflow_v100 等) 路径键名不同,
# 不支持数据集 YAML 注入
_DEFAULT_TRAIN_SUBDIR = "train"
_DEFAULT_VAL_SUBDIR = "valid"
_DEFAULT_ANN_FILE = "_annotations.coco.json"


# ─── Argument parser ────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SAM 3 训练 (转发到 Hydra 训练系统)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python sam.py configs/train/roboflow_finetune.yaml
    python -m commands.train --config configs/train/roboflow_finetune.yaml --num-gpus 2
    python sam.py configs/train/roboflow_finetune.yaml --batch-size 2 --resolution 672
    python sam.py configs/train/roboflow_finetune.yaml --override scratch.lr_scale=0.05

model.config 指向模型网络配置 (configs/models/xxx.yaml), 其中的 hydra_config
才是 Hydra 训练配置 (子模块内), 如:
    sam3/sam3/train/configs/custom_image_ft.yaml   (自定义数据集微调模板, 推荐)
    sam3/sam3/train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml
        """,
    )

    parser.add_argument("--config", type=str, default=None, help="YAML 配置文件路径")
    parser.add_argument("--sam3-config", type=str, default=None,
                        help="Hydra 训练配置路径 (前端相对/绝对路径, 如 sam3/sam3/train/configs/custom_image_ft.yaml)")
    parser.add_argument("--data", type=str, default=None,
                        help="数据集配置路径 (如 configs/datasets/xxx.yaml)")
    parser.add_argument("--use-cluster", type=int, default=None, choices=[0, 1],
                        help="0=本地训练, 1=SLURM 集群")
    parser.add_argument("--partition", type=str, default=None, help="SLURM 分区名")
    parser.add_argument("--account", type=str, default=None, help="SLURM 账户名")
    parser.add_argument("--qos", type=str, default=None, help="SLURM QOS")
    parser.add_argument("--num-gpus", type=int, default=None, help="每节点 GPU 数")
    parser.add_argument("--num-nodes", type=int, default=None, help="节点数")
    parser.add_argument("--output", type=str, default=None,
                        help="实验输出目录 (checkpoints/tensorboard/logs)")

    # ── 常用训练参数 (翻译为 Hydra override) ──
    parser.add_argument("--pretrained", type=str, default=None,
                        help="预训练权重: 路径 / true (HF 自动下载, 默认) / false (从零训练)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="每卡 batch size (scratch.train_batch_size)")
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="训练轮数 (trainer.max_epochs)")
    parser.add_argument("--resolution", type=int, default=None,
                        help="训练分辨率, 须为 336 的倍数 (同步改 scratch.resolution 与 trainer.model.image_size)")
    parser.add_argument("--gpu-ids", type=str, default=None,
                        help="使用哪些 GPU, 如 '1,2,3' (CUDA_VISIBLE_DEVICES; 未给 num-gpus 时按数量自动设置)")
    parser.add_argument("--override", type=str, default=None, action="append",
                        help="透传任意 Hydra 覆盖, 可多次使用, 如 --override scratch.lr_scale=0.05")

    return parser.parse_args()


# ─── Config path resolution ─────────────────────────────────────────────────


def resolve_hydra_config(config_value: str) -> str:
    """把前端视角的配置路径解析为 Hydra config 名 (相对于 sam3/sam3/train/)。

    ``config_value`` 为项目根相对路径或绝对路径, 必须指向子模块内的配置
    (Hydra initialize_config_module 要求配置必须在 sam3.train 模块内), 如
    ``sam3/sam3/train/configs/custom_image_ft.yaml`` 或
    ``sam3/sam3/train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml``。
    """
    path = _resolve_frontend_path(config_value, must_exist=False)
    if not path.exists():
        raise FileNotFoundError(
            f"训练配置不存在: {config_value} (解析为 {path})"
        )
    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(f"训练配置必须是 YAML 文件: {path}")
    try:
        return path.relative_to(HYDRA_CONFIG_ROOT.resolve()).as_posix()
    except ValueError:
        raise ValueError(
            f"Hydra 训练配置必须位于 sam3 子模块内 (sam3/sam3/train/configs/):\n"
            f"  {path}\n"
            f"前端自定义配置请复制模板 sam3/sam3/train/configs/custom_image_ft.yaml "
            f"到该目录下修改 (改动属于 sam3 fork 仓库)"
        )


def _resolve_frontend_path(value: str, must_exist: bool = False) -> Path:
    """前端视角路径 (相对项目根或绝对) → 绝对路径, 统一为正斜杠供 Hydra 使用。"""
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(f"路径不存在: {value} (解析为 {path})")
    return path


# ─── Hydra override translation ─────────────────────────────────────────────


def _dedupe_overrides(overrides: List[str]) -> List[str]:
    """同一键多次覆盖时后者生效 (Hydra 对重复键报错), 保持首次出现顺序。"""
    seen: Dict[str, str] = {}
    for ov in overrides:
        key = ov.split("=", 1)[0].lstrip("+")
        seen[key] = ov
    return list(seen.values())


def _normalize_gpu_ids(value: Any) -> Optional[List[int]]:
    """device/gpu_ids 归一化为 int 列表: '0,1,3' / [0, 1] / 1 都可以。"""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(x) for x in str(value).split(",") if x.strip()]


def build_dataset_overrides(data_config_value: str) -> List[str]:
    """把独立的数据集 YAML (configs/datasets/xxx.yaml) 翻译为 Hydra overrides。

    覆盖的都是标准键 (微调模板 custom_image_ft.yaml 与后端 roboflow 参考配置
    共有的 Sam3ImageDataset 键), 用 ``=``; paths.dataset_root 只有微调模板有。
    """
    ds_path = _resolve_frontend_path(data_config_value, must_exist=True)
    ds = load_yaml_config(str(ds_path)) or {}

    root = ds.get("path")
    if not root:
        raise ValueError(f"数据集配置缺少 path 字段: {ds_path}")
    root = _resolve_frontend_path(str(root), must_exist=True).as_posix()

    train_sub = str(ds.get("train") or _DEFAULT_TRAIN_SUBDIR).strip("/")
    val_sub = str(ds.get("val") or _DEFAULT_VAL_SUBDIR).strip("/")
    ann_file = str(ds.get("ann_file") or _DEFAULT_ANN_FILE)

    overrides = [
        f'paths.dataset_root="{root}"',
        f'trainer.data.train.dataset.img_folder="{root}/{train_sub}/"',
        f'trainer.data.train.dataset.ann_file="{root}/{train_sub}/{ann_file}"',
        f'trainer.data.val.dataset.img_folder="{root}/{val_sub}/"',
        f'trainer.data.val.dataset.ann_file="{root}/{val_sub}/{ann_file}"',
        # 验证集 COCO 评测的 GT 路径 (模板里是插值, 子目录非默认时会断, 统一覆盖)
        f'trainer.meters.val.custom.detection.pred_file_evaluators.0.gt_path="{root}/{val_sub}/{ann_file}"',
    ]
    if ds.get("num_images") is not None:
        # 两个配置的 limit_ids 键都存在 (值各自插值到 num_images), 直接覆盖
        overrides.append(f"trainer.data.train.dataset.limit_ids={int(ds['num_images'])}")
    return overrides


def build_hydra_overrides(knobs: Dict[str, Any]) -> List[str]:
    """把前端常用配置翻译为 Hydra override 列表。

    ``++`` 前缀 = 键可能不存在于参考配置中 (checkpoint_path/image_size/
    load_from_HF 在参考配置的 model 段里都没有), 让 Hydra 自动 add-or-override;
    其余映射的键都已在模板/参考配置里核实存在, 用普通 ``=``。
    """
    overrides: List[str] = []

    # pretrained: 路径 / true / false (YAML 里加引号的字符串形式也归一化)
    pretrained = knobs.get("pretrained")
    if isinstance(pretrained, str):
        low = pretrained.strip().lower()
        if low in ("false", "none", "null"):
            pretrained = False
        elif low == "true":
            pretrained = None
    if pretrained is False:
        # 从零训练: 关掉 HF 自动下载即可 (checkpoint_path 缺省为 None)
        overrides.append("++trainer.model.load_from_HF=false")
    elif isinstance(pretrained, str):
        ckpt = _resolve_frontend_path(pretrained.strip(), must_exist=True).as_posix()
        overrides.append(f'++trainer.model.checkpoint_path="{ckpt}"')
    # pretrained 缺省或为 true: 后端默认 load_from_HF=True, 自动从 HF 下载 sam3 权重

    resolution = knobs.get("resolution")
    if resolution is not None:
        resolution = int(resolution)
        if resolution % IMAGE_SIZE_STEP != 0:
            raise ValueError(
                f"resolution 必须是 {IMAGE_SIZE_STEP} 的倍数 (得到 {resolution})"
            )
        # 数据 pipeline (transforms pad 到 resolution) 与模型构建 (RoPE 预计算网格)
        # 都要改, 否则 token 数不匹配
        overrides.append(f"scratch.resolution={resolution}")
        overrides.append(f"++trainer.model.image_size={resolution}")

    # ── 训练旋钮 → Hydra 键映射 (键均已对照 roboflow 参考配置与 custom_image_ft.yaml 核实) ──
    key_map = {
        "batch_size": "scratch.train_batch_size",
        "max_epochs": "trainer.max_epochs",
        "lr_scale": "scratch.lr_scale",                 # 各组 lr = base × lr_scale
        "weight_decay": "scratch.wd",
        "grad_accum": "scratch.gradient_accumulation_steps",
        "val_freq": "trainer.val_epoch_freq",
        "workers": "scratch.num_train_workers",
        "save_freq": "trainer.checkpoint.save_freq",    # 0=只存最后一个
        "seed": "trainer.seed_value",
        "amp": "trainer.optim.amp.enabled",             # bf16 autocast
    }
    for knob, hydra_key in key_map.items():
        value = knobs.get(knob)
        if value is None:
            continue
        if isinstance(value, bool):
            value = str(value).lower()
        overrides.append(f"{hydra_key}={value}")

    # 任意 Hydra 覆盖, 原样透传
    overrides.extend(str(o) for o in knobs.get("overrides") or [])

    return _dedupe_overrides(overrides)


# ─── Main training orchestration ────────────────────────────────────────────


def train(config: Dict) -> None:
    """转发训练请求到 SAM3 Hydra 训练系统。"""
    model_cfg: Dict[str, Any] = config.get("model") or {}
    train_cfg: Dict[str, Any] = config.get("train") or {}
    data_cfg: Dict[str, Any] = config.get("data") or {}
    output_cfg: Dict[str, Any] = config.get("output") or {}

    # ── 模型网络配置 (configs/models/xxx.yaml) ──
    model_ref = model_cfg.get("config")
    if not model_ref:
        raise ValueError(
            "配置缺少 model.config (模型网络配置路径, 如 configs/models/sam3_image.yaml)"
        )
    net_path = _resolve_frontend_path(str(model_ref), must_exist=True)
    net_cfg = load_yaml_config(str(net_path)) or {}

    sam3_config = model_cfg.get("hydra_config") or net_cfg.get("hydra_config")
    if not sam3_config:
        raise ValueError(
            f"模型配置缺少 hydra_config 字段: {net_path}\n"
            f"(Hydra 训练配置路径, 如 sam3/sam3/train/configs/custom_image_ft.yaml)"
        )
    # 训练 YAML 的 model.pretrained/resolution 可覆盖模型配置里的同名项
    pretrained = model_cfg.get("pretrained", net_cfg.get("pretrained"))
    resolution = model_cfg.get("resolution", net_cfg.get("resolution"))
    net_overrides = net_cfg.get("overrides") or []

    use_cluster = train_cfg.get("use_cluster")
    partition = train_cfg.get("partition")
    account = train_cfg.get("account")
    qos = train_cfg.get("qos")
    num_gpus = train_cfg.get("num_gpus")
    num_nodes = train_cfg.get("num_nodes")
    gpu_ids = _normalize_gpu_ids(train_cfg.get("device"))
    if gpu_ids and num_gpus is None:
        num_gpus = len(gpu_ids)

    train_script = SAM3_ROOT / "sam3" / "train" / "train.py"
    if not train_script.exists():
        raise FileNotFoundError(
            f"SAM3 训练脚本不存在: {train_script}\n"
            "请确保 sam3 子模块已正确初始化 (git submodule update --init)"
        )

    sam3_config = resolve_hydra_config(sam3_config)

    # ── Hydra overrides: 基础设施 (bpe/output/数据集) → 模型网络配置
    #    (pretrained/resolution/网络 overrides) → 训练旋钮 → 用户透传 ──
    hydra_overrides: List[str] = [
        # bpe_path 在参考配置里是 <BPE_PATH> 占位符, 统一注入绝对路径
        f'paths.bpe_path="{BPE_PATH.as_posix()}"',
    ]
    if not BPE_PATH.exists():
        print(f"警告: BPE 资源不存在: {BPE_PATH} (请确认 sam3 子模块完整)")

    output_path = output_cfg.get("path")
    if output_path:
        log_dir = _resolve_frontend_path(str(output_path)).as_posix()
        hydra_overrides.append(f'paths.experiment_log_dir="{log_dir}"')

    data_config = data_cfg.get("config")
    if data_config:
        hydra_overrides += build_dataset_overrides(str(data_config))

    # 模型网络配置自带的网络相关 overrides (优先级最低的网络层覆盖)
    hydra_overrides += [str(o) for o in net_overrides]

    hydra_overrides += build_hydra_overrides({
        "pretrained": pretrained,
        "resolution": resolution,
        "batch_size": train_cfg.get("batch"),
        "max_epochs": train_cfg.get("epochs"),
        "lr_scale": train_cfg.get("lr_scale"),
        "weight_decay": train_cfg.get("weight_decay"),
        "grad_accum": train_cfg.get("grad_accum"),
        "val_freq": train_cfg.get("val_freq"),
        "workers": train_cfg.get("workers"),
        "save_freq": train_cfg.get("save_freq"),
        "seed": train_cfg.get("seed"),
        "amp": train_cfg.get("amp"),
        "overrides": train_cfg.get("overrides"),
    })
    hydra_overrides = _dedupe_overrides(hydra_overrides)

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
    # 任意 Hydra 覆盖 (后端 train.py 以 parse_known_args 收集并传给 compose)
    cmd += hydra_overrides

    print(f"\n{'='*60}")
    print("SAM 3 训练")
    print(f"{'='*60}")
    print(f"模型配置: {model_ref}")
    print(f"Hydra config: {sam3_config}")
    print(f"训练脚本: {train_script}")
    if data_config:
        print(f"数据集配置: {data_config}")
    if use_cluster is not None:
        print(f"集群模式: {'SLURM' if use_cluster else '本地'}")
    if gpu_ids:
        print(f"GPU: {','.join(str(g) for g in gpu_ids)}")
    if num_gpus is not None:
        print(f"GPU 数: {num_gpus}")
    if num_nodes is not None:
        print(f"节点数: {num_nodes}")
    if hydra_overrides:
        print(f"Hydra overrides: {' '.join(hydra_overrides)}")
    print(f"{'='*60}\n")
    print(f"执行命令: {' '.join(cmd)}\n")

    # 子进程不继承本进程的 sys.path, 通过 PYTHONPATH 保证能 import sam3
    # (即使未执行 pip install -e sam3)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SAM3_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    if gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    subprocess.run(cmd, check=True, env=env)


def main():
    args = parse_args()
    try:
        config = {}
        if args.config:
            config = load_yaml_config(args.config)

        # CLI overrides YAML (写入新结构对应的分区)
        model_cli: Dict[str, Any] = {}
        train_cli: Dict[str, Any] = {}
        data_cli: Dict[str, Any] = {}
        output_cli: Dict[str, Any] = {}

        if args.sam3_config is not None:
            model_cli["hydra_config"] = args.sam3_config  # 覆盖模型配置里的 hydra_config
        if args.pretrained is not None:
            model_cli["pretrained"] = args.pretrained  # 字符串形式在 build 阶段归一化
        if args.resolution is not None:
            model_cli["resolution"] = args.resolution
        if args.data is not None:
            data_cli["config"] = args.data
        if args.output is not None:
            output_cli["path"] = args.output
        for field in ("use_cluster", "partition", "account", "qos", "num_gpus", "num_nodes"):
            v = getattr(args, field, None)
            if v is not None:
                train_cli[field] = v
        if args.batch_size is not None:
            train_cli["batch"] = args.batch_size
        if args.max_epochs is not None:
            train_cli["epochs"] = args.max_epochs
        if args.gpu_ids is not None:
            train_cli["device"] = [int(x) for x in args.gpu_ids.split(",") if x.strip()]
        if args.override:
            # 与 YAML 的 overrides 合并 (CLI 在后, 同键时 CLI 生效)
            yaml_overrides = get_nested_value(config, "train", "overrides") or []
            train_cli["overrides"] = [*yaml_overrides, *args.override]

        # 注意: YAML 里只有注释的分区会被解析成 None, setdefault 不会覆盖 None
        for section, cli in (("model", model_cli), ("train", train_cli),
                             ("data", data_cli), ("output", output_cli)):
            if cli:
                if not isinstance(config.get(section), dict):
                    config[section] = {}
                config[section].update(cli)

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
