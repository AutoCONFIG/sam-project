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

- ``configs/train/xxx.yaml``    — train entry: ``model`` / ``resolution`` /
  ``data.config`` / training knobs (``train.*``) / output dir (``output.path``)
- ``configs/models/xxx.yaml``   — model config: the complete Hydra training
  config flattened into its ``hydra:`` section (materialized into
  ``sam3/sam3/train/configs/_custom/`` at launch — Hydra requires configs to
  live inside the ``sam3.train`` module)
- ``configs/datasets/xxx.yaml`` — dataset root + COCO split layout,
  translated into Hydra overrides (``paths.dataset_root``,
  ``trainer.data.*.dataset.img_folder/ann_file``)
- ``configs/export/xxx.yaml``   — ONNX export (mode: export, separate command)

The train YAML ``model`` field is polymorphic (finetuning never changes the
network architecture, so a weights file alone is enough):

- ``pretrain/xxx.pt``  — finetune from these weights (using the default model
  config ``configs/models/sam3_image.yaml``), injects
  ``++trainer.model.checkpoint_path``
- ``configs/models/xxx.yaml`` — train from scratch with that model config,
  injects ``++trainer.model.load_from_HF=false``
- ``hf`` or omitted    — finetune from the official HF weights (backend
  default ``load_from_HF=True``; gated repo needs a token)

Top-level ``hydra_config`` (or CLI ``--sam3-config``) is an escape hatch
pointing at a ready-made config inside the submodule (e.g. the official
roboflow reference config). ``resolution`` and ``train.*`` knobs are
translated into Hydra overrides and passed as trailing args (the backend
``train.py`` forwards them to ``hydra.compose``); long-tail parameters are
edited directly in the model config's ``hydra:`` section. ``paths.bpe_path``
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

# 默认模型配置 (model 字段指向权重或缺省时使用)
DEFAULT_MODEL_CONFIG = "configs/models/sam3_image.yaml"

# 数据集 YAML 注入依赖 Hydra 配置里的标准键 (paths.dataset_root /
# trainer.data.{train,val}.dataset.*), 即模型配置 sam3_image.yaml hydra 段的
# 结构; 后端自带配置 (roboflow_v100 等) 路径键名不同, 不支持数据集 YAML 注入
_DEFAULT_TRAIN_SUBDIR = "train"
_DEFAULT_VAL_SUBDIR = "valid"
_DEFAULT_ANN_FILE = "_annotations.coco.json"

# 训练旋钮 → Hydra 键映射: 键均已对照官方 roboflow 参考配置与
# configs/models/sam3_image.yaml 的 hydra 段逐一核实存在且语义一致
TRAIN_KEY_MAP = {
    "batch": "scratch.train_batch_size",
    "epochs": "trainer.max_epochs",
    "lr_scale": "scratch.lr_scale",                 # 各组 lr = base × lr_scale
    "weight_decay": "scratch.wd",
    "lrd": "scratch.lrd_vision_backbone",           # ViT layer-wise lr decay
    "scheduler_timescale": "scratch.scheduler_timescale",
    "scheduler_warmup": "scratch.scheduler_warmup",
    "scheduler_cooldown": "scratch.scheduler_cooldown",
    "grad_accum": "scratch.gradient_accumulation_steps",
    "grad_clip": "trainer.optim.gradient_clip.max_norm",
    "amp": "trainer.optim.amp.enabled",             # bf16 autocast
    "amp_dtype": "trainer.optim.amp.amp_dtype",
    "val_freq": "trainer.val_epoch_freq",
    "skip_first_val": "trainer.skip_first_val",
    "val_batch": "scratch.val_batch_size",
    "workers": "scratch.num_train_workers",
    "val_workers": "scratch.num_val_workers",
    "max_ann_per_img": "scratch.max_ann_per_img",   # 单图最多标注数 (超出过滤)
    "save_freq": "trainer.checkpoint.save_freq",    # 0=只存最后一个
    "log_freq": "trainer.logging.log_freq",
    "skip_saving_ckpts": "trainer.skip_saving_ckpts",  # 微调必须 false
    "seed": "trainer.seed_value",
    "timeout_hour": "submitit.timeout_hour",        # 仅集群
    "cpus_per_task": "submitit.cpus_per_task",      # 仅集群
}


# ─── Argument parser ────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SAM 3 训练 (转发到 Hydra 训练系统)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python sam.py configs/train/custom_finetune.yaml
    python -m commands.train --config configs/train/custom_finetune.yaml --num-gpus 2
    python sam.py configs/train/custom_finetune.yaml --model pretrain/sam3/sam3.pt
    python sam.py configs/train/custom_finetune.yaml --batch-size 2 --resolution 672

model 字段 (标量): 权重 .pt = 预训练微调 (默认模型配置 configs/models/sam3_image.yaml)
/ 模型配置 .yaml = 从头训练 / hf 或缺省 = HF 下载官方权重微调; 顶层 hydra_config
字段 (--sam3-config) 可直接指向子模块内现成配置, 如官方参考配置:
    sam3/sam3/train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml
        """,
    )

    parser.add_argument("--config", type=str, default=None, help="YAML 配置文件路径")
    parser.add_argument("--sam3-config", type=str, default=None,
                        help="顶层 hydra_config 的 CLI 形式: 直接指定子模块内的 Hydra 训练配置 (绕过模型配置的 hydra 段)")
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
    parser.add_argument("--model", type=str, default=None,
                        help="模型: 权重 .pt=预训练微调 / 模型配置 .yaml=从头训练 / hf=HF 下载官方权重微调 (默认)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="每卡 batch size (scratch.train_batch_size)")
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="训练轮数 (trainer.max_epochs)")
    parser.add_argument("--resolution", type=int, default=None,
                        help="训练分辨率, 须为 336 的倍数 (同步改 scratch.resolution 与 trainer.model.image_size)")
    parser.add_argument("--gpu-ids", type=str, default=None,
                        help="使用哪些 GPU, 如 '1,2,3' (CUDA_VISIBLE_DEVICES; 未给 num-gpus 时按数量自动设置)")

    return parser.parse_args()


# ─── Config path resolution ─────────────────────────────────────────────────


def resolve_hydra_config(config_value: str) -> str:
    """把前端视角的配置路径解析为 Hydra config 名 (相对于 sam3/sam3/train/)。

    ``config_value`` 为项目根相对路径或绝对路径, 必须指向子模块内的配置
    (Hydra initialize_config_module 要求配置必须在 sam3.train 模块内), 如
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
            f"自定义训练请用 configs/models/ 模型配置的 hydra: 段 (平铺完整配置)"
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


def materialize_hydra_config(net_path: Path, net_cfg: Dict[str, Any]) -> str:
    """把模型配置 (configs/models/xxx.yaml) 的 ``hydra:`` 段原样落到子模块
    ``sam3/sam3/train/configs/_custom/<模型配置文件名>.yaml``, 返回 Hydra config 名。

    Hydra initialize_config_module 要求配置必须在 sam3.train 模块内, 所以平铺在
    前端模型配置里的完整训练配置要生成进子模块才能被后端加载。按文本原样搬运
    (不经 YAML 序列化往返), ``${...}`` 插值与注释完全保真; 内容不变时不重写,
    避免无意义的 mtime 变化。
    """
    if not isinstance(net_cfg.get("hydra"), dict):
        raise ValueError(
            f"模型配置缺少 hydra 段 (完整 Hydra 训练配置的平铺内容): {net_path}\n"
            f"或用训练配置顶层 hydra_config 字段指向子模块内现成配置"
        )
    lines = net_path.read_text(encoding="utf-8").splitlines()
    start = next((i for i, line in enumerate(lines) if line == "hydra:"), None)
    if start is None:
        raise ValueError(f"模型配置里找不到顶层 hydra: 段: {net_path}")

    body: List[str] = []
    for line in lines[start + 1:]:
        if not line.strip():
            body.append("")
        elif line.startswith("  "):
            body.append(line[2:])
        elif line.startswith("#"):
            body.append(line)   # 未缩进的注释行也保留
        else:
            break               # 回到顶层键 → hydra 段结束
    text = "# @package _global_\n" + "\n".join(body).strip("\n") + "\n"

    custom_dir = HYDRA_CONFIG_ROOT / "configs" / "_custom"
    custom_dir.mkdir(parents=True, exist_ok=True)
    dst = custom_dir / (net_path.stem + ".yaml")
    if not dst.exists() or dst.read_text(encoding="utf-8") != text:
        dst.write_text(text, encoding="utf-8")
        print(f"已由模型配置生成 Hydra 配置: {dst}")
    return f"configs/_custom/{dst.name}"


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

    覆盖的都是标准键 (模型配置 sam3_image.yaml hydra 段与后端 roboflow 参考配置
    共有的 Sam3ImageDataset 键), 用 ``=``; paths.dataset_root 只有 hydra 段有。
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

    ``++`` 前缀 = 键可能不存在于参考配置中 (image_size 在参考配置的 model 段里
    没有; checkpoint_path/load_from_HF 由 train() 按 model 字段注入), 让 Hydra
    自动 add-or-override; 其余映射的键都已在模板/参考配置里核实存在, 用普通 ``=``。
    """
    overrides: List[str] = []

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

    # ── 训练旋钮 → Hydra 键 (TRAIN_KEY_MAP, 均为后端配置里确认存在的键) ──
    for knob, hydra_key in TRAIN_KEY_MAP.items():
        value = knobs.get(knob)
        if value is None:
            continue
        if isinstance(value, bool):
            value = str(value).lower()
        overrides.append(f"{hydra_key}={value}")

    return _dedupe_overrides(overrides)


# ─── Main training orchestration ────────────────────────────────────────────


def train(config: Dict) -> None:
    """转发训练请求到 SAM3 Hydra 训练系统。"""
    train_cfg: Dict[str, Any] = config.get("train") or {}
    data_cfg: Dict[str, Any] = config.get("data") or {}
    output_cfg: Dict[str, Any] = config.get("output") or {}

    # ── model 字段 (标量多态; 预训练微调不改变网络结构, 指权重即可) ──
    #   权重 .pt   → 预训练微调 (默认模型配置 DEFAULT_MODEL_CONFIG), 注入 checkpoint_path
    #   配置 .yaml → 从头训练 (该模型配置的 hydra 段), 注入 load_from_HF=false
    #   hf / 缺省  → HF 下载官方权重微调 (后端默认 load_from_HF=True, 不注入)
    model_val = config.get("model")
    if isinstance(model_val, dict):
        raise ValueError(
            "model 字段是标量, 不是分区:\n"
            "  model: pretrain/sam3/sam3.pt           # 权重 = 预训练微调\n"
            "  model: configs/models/sam3_image.yaml  # 模型配置 = 从头训练\n"
            "  model: hf                              # 或不写 = HF 下载官方权重微调"
        )
    model_str = str(model_val).strip() if model_val is not None else ""
    weights: Optional[str] = None
    from_scratch = False
    if not model_str or model_str.lower() == "hf":
        net_ref = DEFAULT_MODEL_CONFIG
    elif model_str.lower().endswith((".yaml", ".yml")):
        net_ref = model_str
        from_scratch = True
    else:
        net_ref = DEFAULT_MODEL_CONFIG
        weights = model_str

    net_path = _resolve_frontend_path(net_ref, must_exist=True)
    net_cfg = load_yaml_config(str(net_path)) or {}

    # Hydra 配置来源: 顶层 hydra_config 逃生舱 (--sam3-config) 指向子模块内现成
    # 配置 (如官方 roboflow 参考配置); 否则用模型配置里平铺的 hydra: 段
    # (启动时按文本原样生成进子模块 configs/_custom/)
    hydra_ref = config.get("hydra_config")
    if hydra_ref:
        sam3_config = resolve_hydra_config(str(hydra_ref))
    else:
        sam3_config = materialize_hydra_config(net_path, net_cfg)

    resolution = config.get("resolution")

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

    # ── Hydra overrides: 基础设施 (bpe/output/数据集) → 预训练权重 (model 字段)
    #    → resolution/训练旋钮 ──
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

    # 预训练权重注入 (后端 build_sam3_image_model: checkpoint_path 优先;
    # 为 None 且 load_from_HF=True 时从 HF 下载; 两者都空 = 从头训练)
    if from_scratch:
        hydra_overrides.append("++trainer.model.load_from_HF=false")
    elif weights:
        ckpt = _resolve_frontend_path(weights, must_exist=True).as_posix()
        hydra_overrides.append(f'++trainer.model.checkpoint_path="{ckpt}"')
    # hf/缺省: 不注入, 后端默认 load_from_HF=True 自动下载 HF 官方权重

    hydra_overrides += build_hydra_overrides({
        "resolution": resolution,
        **{k: train_cfg.get(k) for k in TRAIN_KEY_MAP},
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
    if weights:
        print(f"模型: {weights} (预训练微调)")
    elif from_scratch:
        print(f"模型: 从头训练 (无预训练权重)")
    else:
        print(f"模型: HF 官方权重 (自动下载, 微调)")
    print(f"模型配置: {net_ref}")
    print(f"Hydra config: {sam3_config}")
    if resolution is not None:
        print(f"分辨率: {resolution}")
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

        # CLI overrides YAML (model/resolution/hydra_config 是顶层标量)
        train_cli: Dict[str, Any] = {}
        data_cli: Dict[str, Any] = {}
        output_cli: Dict[str, Any] = {}

        if args.model is not None:
            config["model"] = args.model
        if args.resolution is not None:
            config["resolution"] = args.resolution
        if args.sam3_config is not None:
            config["hydra_config"] = args.sam3_config
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

        # 注意: YAML 里只有注释的分区会被解析成 None, setdefault 不会覆盖 None
        for section, cli in (("train", train_cli), ("data", data_cli),
                             ("output", output_cli)):
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
