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

# 默认训练模板 (后端 Hydra 配置全量默认值: transforms/loss/优化器等超参数;
# 模型定义在模型配置里, 物化时文本合并)
DEFAULT_TEMPLATE = "configs/train/template_image.yaml"

# 数据集 YAML 注入依赖 Hydra 配置里的标准键 (paths.dataset_root /
# trainer.data.{train,val}.dataset.*), 即模型配置 sam3_image.yaml hydra 段的
# 结构; 后端自带配置 (roboflow_v100 等) 路径键名不同, 不支持数据集 YAML 注入
_DEFAULT_TRAIN_SUBDIR = "train"
_DEFAULT_VAL_SUBDIR = "valid"
_DEFAULT_ANN_FILE = "_annotations.coco.json"

# 训练旋钮 → Hydra 键映射: 键均已对照官方 roboflow 参考配置与
# 训练模板 configs/train/template_image.yaml 逐一核实存在且语义一致
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
    "early_stop": "++trainer.early_stop.enabled",      # 早停开关 (按验证次数计)
    "early_stop_patience": "++trainer.early_stop.patience",  # 连续 N 次验证无改进即停
    # ── 验证可视化 (YOLO 风格马赛克; 只影响出图, 不影响训练与 mAP) ──
    "viz_max_files": "trainer.meters.val.custom.viz.max_files",
    "viz_per_file": "trainer.meters.val.custom.viz.per_file",
    "viz_score_threshold": "trainer.meters.val.custom.viz.score_threshold",
    "viz_min_per_img": "trainer.meters.val.custom.viz.min_per_img",
    "viz_max_per_img": "trainer.meters.val.custom.viz.max_per_img",
    # ── 损失权重 / 匹配器成本 ──
    # loss 段源在 custom_data.loss (经插值 ${custom_data.loss} 进 trainer.loss.all,
    # 无法从 trainer 侧覆盖, 必须覆盖源); loss_fns_find 列表顺序固定: 0=Boxes 1=IABCEMdetr;
    # hydra_config 逃生舱指向官方参考配置时段名不同, train() 里统一换成 roboflow_train
    "loss_bbox": "custom_data.loss.loss_fns_find.0.weight_dict.loss_bbox",
    "loss_giou": "custom_data.loss.loss_fns_find.0.weight_dict.loss_giou",  # 小目标多可适当抬高
    "loss_ce": "custom_data.loss.loss_fns_find.1.weight_dict.loss_ce",
    "presence_loss": "custom_data.loss.loss_fns_find.1.weight_dict.presence_loss",
    "pos_weight": "custom_data.loss.loss_fns_find.1.pos_weight",  # 分类 BCE 正样本权重
    "focal_alpha": "custom_data.loss.loss_fns_find.1.alpha",
    "focal_gamma": "custom_data.loss.loss_fns_find.1.gamma",
    "o2m_weight": "custom_data.loss.o2m_weight",            # one-to-many 辅助分支权重
    "matcher_cost_class": "scratch.matcher.cost_class",     # 匈牙利匹配分类成本
    "matcher_cost_bbox": "scratch.matcher.cost_bbox",
    "matcher_cost_giou": "scratch.matcher.cost_giou",
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
                        help="顶层 hydra_config 的 CLI 形式: 直接指定子模块内的 Hydra 训练配置 (绕过训练模板+模型配置)")
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
            f"自定义训练请用训练模板 configs/train/template_image.yaml "
            f"+ configs/models/ 模型配置 (trainer.model 段)"
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


def _extract_hydra_lines(path: Path) -> List[str]:
    """提取 yaml 顶层 ``hydra:`` 段的文本行 (去 2 空格缩进; 未缩进注释行保留)。"""
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next((i for i, line in enumerate(lines) if line == "hydra:"), None)
    if start is None:
        raise ValueError(f"配置里找不到顶层 hydra: 段: {path}")
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
    return body


def materialize_hydra_config(
    tpl_path: Path,
    net_path: Path,
    net_cfg: Dict[str, Any],
    dataset_blocks: Optional[Dict[str, List[str]]] = None,
) -> str:
    """把训练模板 (configs/train/template_image.yaml) 与模型配置
    (configs/models/xxx.yaml) 的 ``trainer.model`` 段做文本合并, 落到子模块
    ``sam3/sam3/train/configs/_custom/<模板文件名>.yaml``, 返回 Hydra config 名。

    Hydra initialize_config_module 要求配置必须在 sam3.train 模块内, 所以平铺在
    前端的完整训练配置要生成进子模块才能被后端加载。按文本原样搬运 (不经 YAML
    序列化往返), ``${...}`` 插值与注释完全保真; 内容不变时不重写, 避免无意义的
    mtime 变化。模板里的 ``__MODEL_BLOCK__`` 占位行被替换为模型配置的 model 段;
    ``__TRAIN_DATASET_BLOCK__`` / ``__VAL_DATASET_BLOCK__`` 占位行 (多数据集时)
    被替换为 ConcatSam3Datasets 拼接块 (单数据集时不传, 模板默认块原样保留)。
    """
    if not isinstance(net_cfg.get("hydra"), dict) \
            or not isinstance(net_cfg["hydra"].get("trainer"), dict) \
            or not isinstance(net_cfg["hydra"]["trainer"].get("model"), dict):
        raise ValueError(
            f"模型配置缺少 hydra.trainer.model 段 (模型构建定义): {net_path}\n"
            f"或用训练配置顶层 hydra_config 字段指向子模块内现成配置"
        )

    # ── 模型配置: 提取 trainer.model 段文本 (去缩进后 model: 在 trainer: 下 2 空格) ──
    net_lines = _extract_hydra_lines(net_path)
    try:
        trainer_idx = net_lines.index("trainer:")
        model_idx = next(i for i in range(trainer_idx + 1, len(net_lines))
                         if net_lines[i].startswith("  model:"))
    except (ValueError, StopIteration):
        raise ValueError(f"模型配置的 hydra 段里找不到 trainer.model: {net_path}") from None
    block: List[str] = []
    for line in net_lines[model_idx:]:
        if block and line.strip() and not line.startswith("    "):
            break               # model 段的子键/注释缩进 ≥4; 同级或更浅的键 → 段结束
        block.append(line)

    # ── 模板: __MODEL_BLOCK__ 占位行替换为模型段 ──
    tpl_lines = _extract_hydra_lines(tpl_path)
    try:
        slot = next(i for i, line in enumerate(tpl_lines) if "__MODEL_BLOCK__" in line)
    except StopIteration:
        raise ValueError(f"训练模板缺少 __MODEL_BLOCK__ 占位行: {tpl_path}") from None
    # 提取后文本里 trainer 的子键在 2 空格缩进, model 段文本缩进正好匹配, 直接拼接
    body = tpl_lines[:slot] + block + tpl_lines[slot + 1:]

    # ── 多数据集: __TRAIN_DATASET_BLOCK__ / __VAL_DATASET_BLOCK__ 占位替换 ──
    #   dataset_blocks 形如 {"train": [yaml 行...], "val": [yaml 行...]}
    #   (commands/train.py build_dataset_blocks 生成, 已含正确缩进)。单数据集时不
    #   传, 模板里默认的单 Sam3ImageDataset 块原样保留 → 行为零变化。
    if dataset_blocks:
        for marker, ds_block in dataset_blocks.items():
            tag = f"__{marker.upper()}_DATASET_BLOCK__"
            tag_end = f"__{marker.upper()}_DATASET_BLOCK_END__"
            start_idx = next(
                (i for i, line in enumerate(body) if tag in line), None)
            if start_idx is None:
                raise ValueError(
                    f"训练模板缺少 {tag} 占位行: {tpl_path}")
            end_idx = next(
                (i for i in range(start_idx + 1, len(body))
                 if tag_end in body[i]),
                None)
            if end_idx is None:
                raise ValueError(
                    f"训练模板缺少 {tag_end} 占位行: {tpl_path}")
            # 占位块内 (含两个标记行) 整体替换为生成的 yaml 块
            body = body[:start_idx] + ds_block + body[end_idx + 1:]

    text = "# @package _global_\n" + "\n".join(body).strip("\n") + "\n"

    custom_dir = HYDRA_CONFIG_ROOT / "configs" / "_custom"
    custom_dir.mkdir(parents=True, exist_ok=True)
    dst = custom_dir / (tpl_path.stem + ".yaml")
    if not dst.exists() or dst.read_text(encoding="utf-8") != text:
        dst.write_text(text, encoding="utf-8")
        print(f"已由训练模板+模型配置生成 Hydra 配置: {dst}")
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


def _resolve_dataset_entry(entry: Dict[str, Any], ds_path: Path) -> Dict[str, str]:
    """单条数据集描述 (path/train/val/ann_file/limit_ratio/num_images) → 标准化字段。

    多数据集 yaml 里 ``datasets:`` 列表的一项, 或单数据集 yaml 的顶层。返回的字段
    都是绝对/正斜杠路径, 供 Hydra override 或 yaml 块拼接直接使用。
    """
    root = entry.get("path")
    if not root:
        raise ValueError(f"数据集配置缺少 path 字段: {ds_path}")
    root = _resolve_frontend_path(str(root), must_exist=True).as_posix()
    train_sub = str(entry.get("train") or _DEFAULT_TRAIN_SUBDIR).strip("/")
    val_sub = str(entry.get("val") or _DEFAULT_VAL_SUBDIR).strip("/")
    ann_file = str(entry.get("ann_file") or _DEFAULT_ANN_FILE)
    limit_ratio = entry.get("limit_ratio")
    if limit_ratio is not None:
        limit_ratio = float(limit_ratio)
        if not (0.0 < limit_ratio <= 1.0):
            raise ValueError(
                f"limit_ratio 必须在 (0, 1] 范围, 得到 {limit_ratio} ({ds_path})")
    return {
        "root": root,
        "train_sub": train_sub,
        "val_sub": val_sub,
        "ann_file": ann_file,
        "limit_ratio": limit_ratio,
        "num_images": entry.get("num_images"),
    }


def build_dataset_overrides(data_config_value: str) -> List[str]:
    """把独立的数据集 YAML (configs/datasets/xxx.yaml) 翻译为 Hydra overrides。

    单数据集格式 (顶层 path/train/val/ann_file) → 返回 scalar overrides, 行为与
    历史一致。多数据集格式 (顶层 ``datasets:`` 列表) → 返回空列表 (改由
    build_dataset_blocks 生成 yaml 块拼进模板, CLI override 无法表达 list-of-dicts)。
    """
    ds_path = _resolve_frontend_path(data_config_value, must_exist=True)
    ds = load_yaml_config(str(ds_path)) or {}

    if ds.get("datasets"):  # 多数据集 → 走 yaml 块, 不发 scalar override
        return []

    f = _resolve_dataset_entry(ds, ds_path)
    root, train_sub, val_sub, ann_file = (
        f["root"], f["train_sub"], f["val_sub"], f["ann_file"])
    overrides = [
        f'paths.dataset_root="{root}"',
        f'trainer.data.train.dataset.img_folder="{root}/{train_sub}/"',
        f'trainer.data.train.dataset.ann_file="{root}/{train_sub}/{ann_file}"',
        f'trainer.data.val.dataset.img_folder="{root}/{val_sub}/"',
        f'trainer.data.val.dataset.ann_file="{root}/{val_sub}/{ann_file}"',
        # 验证集 COCO 评测的 GT 路径 (模板里是插值, 子目录非默认时会断, 统一覆盖)
        f'trainer.meters.val.custom.detection.pred_file_evaluators.0.gt_path="{root}/{val_sub}/{ann_file}"',
    ]
    if f["num_images"] is not None:
        # 两个配置的 limit_ids 键都存在 (值各自插值到 num_images), 直接覆盖
        overrides.append(f"trainer.data.train.dataset.limit_ids={int(f['num_images'])}")
    return overrides


# 多数据集 yaml 块: 每个 Sam3ImageDataset 子项共享的字段 (与模板默认块一致)。
#   值为 str → ``key: value``; 为 dict → 展开为嵌套块 (key: / 子键)。
_CONCAT_TRAIN_DS_FIELDS: Dict[str, Any] = {
    "transforms": "${custom_data.train_transforms}",
    "load_segmentation": "${scratch.enable_segmentation}",
    "max_ann_per_img": 500000,
    "multiplier": 1,
    "max_train_queries": 50000,
    "max_val_queries": 50000,
    "training": "true",
    "use_caching": "False",
}
_CONCAT_VAL_DS_FIELDS: Dict[str, Any] = {
    "load_segmentation": "${scratch.enable_segmentation}",
    "coco_json_loader": {
        "_target_": "sam3.train.data.coco_json_loaders.COCO_FROM_JSON",
        "include_negatives": "true",
        "category_chunk_size": 2,
        "_partial_": "true",
    },
    "transforms": "${custom_data.val_transforms}",
    "max_ann_per_img": 100000,
    "multiplier": 1,
    "training": "false",
}


def _emit_sam3_dataset_yaml(
    fields: Dict[str, Any],
    img_folder: str,
    ann_file: str,
    limit_ratio: Optional[float],
    indent: str,
) -> List[str]:
    """生成一个 Sam3ImageDataset 子项的 yaml 行 (list item, 以 ``- `` 开头)。

    ``indent`` 是 list item ``-`` 所在列; 子键比它多 2 空格。字段值为 dict 时展开
    为嵌套块 (``key:`` + 缩进 2 的子键), 与模板默认 val 块的 coco_json_loader 一致。
    """
    lines = [f"{indent}- _target_: sam3.train.data.sam3_image_dataset.Sam3ImageDataset"]
    child = indent + "  "  # list item 子键比 ``-`` 多 2 空格
    grandchild = child + "  "  # 嵌套块的子键再 +2
    for k, v in fields.items():
        if isinstance(v, dict):
            lines.append(f"{child}{k}:")
            for sk, sv in v.items():
                lines.append(f"{grandchild}{sk}: {sv}")
        else:
            lines.append(f"{child}{k}: {v}")
    lines.append(f"{child}img_folder: {img_folder}")
    lines.append(f"{child}ann_file: {ann_file}")
    if limit_ratio is not None and limit_ratio < 1.0:
        lines.append(f"{child}limit_ratio: {limit_ratio}")
    return lines


def build_dataset_blocks(
    data_config_value: str,
) -> Optional[Dict[str, List[str]]]:
    """多数据集 yaml (顶层 ``datasets:`` 列表) → ConcatSam3Datasets 的 yaml 块文本。

    返回 ``{"train": [yaml 行...], "val": [yaml 行...]}`` (行已含模板要求的缩进,
    供 materialize_hydra_config 替换 __TRAIN/VAL_DATASET_BLOCK__ 占位)。单数据集
    yaml 返回 None (走 build_dataset_overrides 的 scalar override 路径)。
    """
    ds_path = _resolve_frontend_path(data_config_value, must_exist=True)
    ds = load_yaml_config(str(ds_path)) or {}
    entries = ds.get("datasets")
    if not entries:
        return None

    resolved = [_resolve_dataset_entry(e, ds_path) for e in entries]

    # ── train 块: dataset: ConcatSam3Datasets(datasets=[Sam3ImageDataset, ...]) ──
    #   缩进对齐模板 (提取后, 已去 hydra: 的 2 空格): dataset: 在 6 空格, 其子键 8
    #   空格, list item ``-`` 在 8 空格 (与 datasets: 同级)。
    train_lines = [
        "      dataset:",
        "        _target_: sam3.train.data.sam3_image_dataset.ConcatSam3Datasets",
        "        datasets:",
    ]
    for f in resolved:
        train_lines += _emit_sam3_dataset_yaml(
            _CONCAT_TRAIN_DS_FIELDS,
            img_folder=f"{f['root']}/{f['train_sub']}/",
            ann_file=f"{f['root']}/{f['train_sub']}/{f['ann_file']}",
            limit_ratio=f["limit_ratio"],
            indent="        ",  # list item 与 datasets: 同级 (8 空格)
        )

    # ── val 块: 同结构, 但字段用 val 配置 ──
    val_lines = [
        "      dataset:",
        "        _target_: sam3.train.data.sam3_image_dataset.ConcatSam3Datasets",
        "        datasets:",
    ]
    for f in resolved:
        val_lines += _emit_sam3_dataset_yaml(
            _CONCAT_VAL_DS_FIELDS,
            img_folder=f"{f['root']}/{f['val_sub']}/",
            ann_file=f"{f['root']}/{f['val_sub']}/{f['ann_file']}",
            limit_ratio=f["limit_ratio"],
            indent="        ",
        )

    return {"train": train_lines, "val": val_lines}


def build_hydra_overrides(knobs: Dict[str, Any], escape_hatch: bool = False) -> List[str]:
    """把前端常用配置翻译为 Hydra override 列表。

    ``++`` 前缀 = 键可能不存在于参考配置中 (image_size 在参考配置的 model 段里
    没有; checkpoint_path/load_from_HF 由 train() 按 model 字段注入), 让 Hydra
    自动 add-or-override; 其余映射的键都已在模板/参考配置里核实存在, 用普通 ``=``。
    """
    overrides: List[str] = []

    # viz meter 只存在于训练模板 (template_image*.yaml); hydra_config 逃生舱指向
    # 官方参考配置时没有该键, 普通 override 会报 "key not found" 启动失败
    if escape_hatch and any(
        k.startswith("viz_") and knobs.get(k) is not None for k in TRAIN_KEY_MAP
    ):
        print("提示: hydra_config 逃生舱配置没有 viz meter, 已跳过 viz_* 覆盖")

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
        if escape_hatch and knob.startswith("viz_"):
            continue
        if knob == "val_freq" and int(value) < 1:
            # 后端 trainer 用 epoch % val_epoch_freq 排验证, 0 会在首轮结束时
            # ZeroDivisionError (且 checkpoint 尚未保存); 不支持 "0=不验证"
            raise ValueError(
                f"val_freq 必须 >= 1 (每隔几轮验证; 得到 {value})。"
                "最后一轮无论如何都会验证, 不存在完全关闭验证的写法"
            )
        if isinstance(value, bool):
            value = str(value).lower()
        overrides.append(f"{hydra_key}={value}")

    # freeze 是 pattern 列表 (trainer.freeze 为 list 键, 走不了上面的标量映射):
    # unix fnmatch 匹配参数名, 真冻结 (requires_grad=False, 不进优化器/DDP 不同步)。
    # 用 ++ 前缀: 官方参考配置 (roboflow_v100 等, hydra_config 逃生舱) 没有该键
    freeze = knobs.get("freeze")
    if freeze is not None:
        pats = ",".join(f"'{p}'" for p in freeze)
        overrides.append(f"++trainer.freeze=[{pats}]")

    return _dedupe_overrides(overrides)


# ─── Main training orchestration ────────────────────────────────────────────


def train(config: Dict) -> None:
    """转发训练请求到 SAM3 Hydra 训练系统。"""
    train_cfg: Dict[str, Any] = config.get("train") or {}
    data_cfg: Dict[str, Any] = config.get("data") or {}
    output_cfg: Dict[str, Any] = config.get("output") or {}

    # ── model 字段 (标量多态; 预训练微调不改变网络结构, 指权重即可) ──
    #   权重 .pt   → 预训练微调 (默认模型配置 DEFAULT_MODEL_CONFIG), 注入 checkpoint_path
    #   配置 .yaml → 从头训练 (模型构建定义), 注入 load_from_HF=false
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
    # 配置 (如官方 roboflow 参考配置); 否则用训练模板 (template 字段, 默认
    # template_image.yaml) + 模型配置的 trainer.model 段, 启动时文本合并
    # 生成进子模块 configs/_custom/
    hydra_ref = config.get("hydra_config")
    tpl_ref: Optional[str] = None
    # 多数据集: 若 data.config 指向的 yaml 是 datasets: 列表, 生成 ConcatSam3Datasets
    # 块拼进模板 (而非 CLI scalar override); 仅用模板路径时才支持 (hydra_config 逃生
    # 舱指向现成配置时不走文本拼接, 多数据集需另配)。单数据集 yaml → build_dataset_blocks
    # 返回 None, 走 build_dataset_overrides 的 scalar override 路径 (向后兼容)。
    dataset_blocks: Optional[Dict[str, List[str]]] = None
    data_config = data_cfg.get("config")
    if not hydra_ref and data_config:
        dataset_blocks = build_dataset_blocks(str(data_config))  # None=单数据集; 多数据集路径不存在会抛错 (应失败, 不静默回退)
    if hydra_ref:
        sam3_config = resolve_hydra_config(str(hydra_ref))
    else:
        tpl_ref = str(config.get("template") or DEFAULT_TEMPLATE)
        tpl_path = _resolve_frontend_path(tpl_ref, must_exist=True)
        sam3_config = materialize_hydra_config(
            tpl_path, net_path, net_cfg, dataset_blocks=dataset_blocks)

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

    if data_config:
        hydra_overrides += build_dataset_overrides(str(data_config))
        # 多数据集: dataset 块已拼进模板, 但 meters 段的 gt_path 仍插值
        # ${paths.dataset_root}/valid/... (多根时 dataset_root 无意义)。指向第一个
        # 数据集的 val 标注, 评测只覆盖它 (best-effort; 多数据集完整评测需后端改造)。
        if dataset_blocks:
            ds_path = _resolve_frontend_path(str(data_config), must_exist=True)
            ds = load_yaml_config(str(ds_path)) or {}
            first = ds.get("datasets", [{}])[0]
            f = _resolve_dataset_entry(first, ds_path)
            hydra_overrides.append(
                f'trainer.meters.val.custom.detection.pred_file_evaluators.0'
                f'.gt_path="{f["root"]}/{f["val_sub"]}/{f["ann_file"]}"')

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
        "freeze": train_cfg.get("freeze"),
        **{k: train_cfg.get(k) for k in TRAIN_KEY_MAP},
    }, escape_hatch=bool(hydra_ref))
    hydra_overrides = _dedupe_overrides(hydra_overrides)
    if hydra_ref:
        # 逃生舱指向官方参考配置: 数据/loss 段名为 roboflow_train (结构与 custom_data 相同)
        hydra_overrides = [
            o.replace("custom_data.", "roboflow_train.", 1) if o.startswith("custom_data.") else o
            for o in hydra_overrides
        ]

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
    if tpl_ref:
        print(f"训练模板: {tpl_ref}")
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
