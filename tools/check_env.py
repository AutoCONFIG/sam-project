#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环境自检脚本 — 验证 sam-project 训练/推理环境是否就绪
============================================================
检查项 (分两级):
  [必需] 缺失则训练/推理必失败, 脚本退出码 1:
    - Python >= 3.8; numpy < 2 (sam3 后端硬性约束)
    - torch 是 GPU 版而非 CPU 版 (torch.version.cuda), CUDA 可用,
      一次真实 GPU 计算冒烟; bf16 支持 (训练 amp_dtype)
    - sam3 后端可 import, 且指向本仓库 sam3/ 子模块 (editable 安装)
    - 训练依赖: hydra/submitit/fvcore/fairscale/torchmetrics/tensorboard
      /scipy/scikit-image/scikit-learn/zstandard (sam3 [train] extra)
    - pycocotools (val 评测 import; 在 [dev]/[notebooks] extra, 易漏装)
    - 前端依赖: pyyaml / opencv-python
  [资源] 缺失只告警 (WARN), 不阻断:
    - sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz (文本编码器分词表, 子模块完整性)
    - 训练配置的 model 权重文件 (pretrain/sam3/sam3.pt)
    - 训练配置的 data.config 指向的数据集 train/val 标注文件
    - 训练配置的 num_gpus/device 与实际可见 GPU 数是否匹配

用法:
  python tools/check_env.py                                # 默认检查 highway_road_sam3_finetune.yaml
  python tools/check_env.py configs/train/custom_finetune.yaml
"""

import importlib
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN_CONFIG = PROJECT_ROOT / "configs/train/highway_road_sam3_finetune.yaml"

# ── 输出 ──────────────────────────────────────────────────────────────────
if sys.stdout.isatty():
    _OK, _WARN, _FAIL = "\033[32m[ OK ]\033[0m", "\033[33m[WARN]\033[0m", "\033[31m[FAIL]\033[0m"
else:
    _OK, _WARN, _FAIL = "[ OK ]", "[WARN]", "[FAIL]"

n_fail = 0


def ok(msg):
    print(f"{_OK} {msg}")


def warn(msg):
    print(f"{_WARN} {msg}")


def fail(msg):
    global n_fail
    n_fail += 1
    print(f"{_FAIL} {msg}")


def check_import(module, pip_name=None, required=True):
    """import 检查; 返回模块或 None。pip_name 用于报错提示里的安装名。"""
    try:
        m = importlib.import_module(module)
        ok(f"import {module}")
        return m
    except Exception as e:
        tip = f" (pip install {pip_name or module})"
        (fail if required else warn)(f"import {module} 失败: {e}{tip}")
        return None


# ── 1. Python / numpy ─────────────────────────────────────────────────────
print("== Python ==")
v = sys.version_info
if v >= (3, 8):
    ok(f"Python {v.major}.{v.minor}.{v.micro}")
else:
    fail(f"Python {v.major}.{v.minor}.{v.micro} < 3.8 (sam3 后端要求 >=3.8, 推荐 3.12)")

np = check_import("numpy")
if np is not None:
    if int(np.__version__.split(".")[0]) >= 2:
        fail(f"numpy {np.__version__} >= 2, 与 sam3 后端约束 numpy>=1.26,<2 冲突, "
             f"请降级: pip install 'numpy<2'")
    else:
        ok(f"numpy {np.__version__} (<2, 符合 sam3 约束)")

def _driver_cuda_hint():
    """nvidia-smi 查驱动支持的 CUDA 版本, 用于 torch 与驱动不匹配时给出针对性提示。"""
    import re
    import subprocess

    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True,
                             timeout=10).stdout
        drv = re.search(r"Driver Version:\s*([\d.]+)", out)
        cuda = re.search(r"CUDA Version:\s*([\d.]+)", out)
        if cuda:
            return (f"驱动 {drv.group(1) if drv else '?'} 最高支持 CUDA "
                    f"{cuda.group(1)}")
    except Exception:
        pass
    return "未能查询 nvidia-smi"


# ── 2. PyTorch / CUDA ─────────────────────────────────────────────────────
print("== PyTorch / CUDA ==")
torch = check_import("torch")
if torch is not None:
    # 先区分 GPU 版 / CPU 版 (编译期), 再查 CUDA 可用性 (运行期), 避免两者混淆
    if torch.version.cuda is None:
        fail(f"torch {torch.__version__} 是 CPU 版 (torch.version.cuda=None); "
             f"请重装 CUDA 版, 如: pip install torch torchvision "
             f"--index-url https://download.pytorch.org/whl/cu128")
    else:
        ok(f"torch {torch.__version__} (GPU 版, 编译 CUDA {torch.version.cuda})")
        if not torch.cuda.is_available():
            fail(f"torch 是 GPU 版 (编译 CUDA {torch.version.cuda}) 但 CUDA 不可用: "
                 f"{_driver_cuda_hint()}. torch 编译 CUDA 高于驱动支持上限时, "
                 f"重装匹配轮子即可, 如: pip install torch torchvision "
                 f"--index-url https://download.pytorch.org/whl/cu128 "
                 f"(否则查 CUDA_VISIBLE_DEVICES / nvidia-smi)")
        else:
            n = torch.cuda.device_count()
            names = {torch.cuda.get_device_name(i) for i in range(n)}
            ok(f"CUDA 可用, {n} 张卡: {', '.join(sorted(names))}")
            try:
                x = torch.randn(64, 64, device="cuda:0")
                torch.mm(x, x).sum().item()
                ok("GPU 计算冒烟 (cuda:0 matmul)")
            except Exception as e:
                fail(f"GPU 计算冒烟失败: {e}")
            if torch.cuda.is_bf16_supported():
                ok("bf16 支持 (训练 amp_dtype: bfloat16)")
            else:
                fail("当前 GPU 不支持 bf16 (训练配置 amp_dtype: bfloat16 将无法运行)")
check_import("torchvision")

# ── 3. sam3 后端 ──────────────────────────────────────────────────────────
print("== sam3 后端 (pip install -e sam3/) ==")
sam3 = check_import("sam3")
if sam3 is not None:
    p = Path(sam3.__file__).resolve()
    if str(p).startswith(str((PROJECT_ROOT / "sam3").resolve())):
        ok(f"sam3 指向本仓库子模块: {p}")
    else:
        warn(f"sam3 不在本仓库子模块内 ({p}); 若非 editable 安装, 对子模块的改动不会生效")

# ── 4. 训练依赖 (sam3 [train] extra) + 评测 + 前端 ─────────────────────────
print("== 训练/评测/前端依赖 ==")
for mod, pkg in [
    ("hydra", "hydra-core"), ("submitit", "submitit"), ("fvcore", "fvcore"),
    ("fairscale", "fairscale"), ("torchmetrics", "torchmetrics"),
    ("tensorboard", "tensorboard"), ("scipy", "scipy"), ("skimage", "scikit-image"),
    ("sklearn", "scikit-learn"), ("zstandard", "zstandard"),
    ("pycocotools", "pycocotools"),  # 在 [dev]/[notebooks] extra, [train] 没有, 易漏
    ("yaml", "pyyaml"), ("cv2", "opencv-python"),
]:
    check_import(mod, pkg)

# ── 5. 项目资源 (缺失只 WARN) ──────────────────────────────────────────────
print("== 项目资源 (缺失仅告警) ==")
bpe = PROJECT_ROOT / "sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"  # 包内 assets (非仓库根 assets/)
ok(f"BPE 分词表: {bpe}") if bpe.exists() else warn(
    f"BPE 分词表缺失: {bpe} (子模块不完整? git submodule update --init)")

train_yaml = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TRAIN_CONFIG
print(f"-- 依据训练配置: {train_yaml}")
try:
    import yaml as _yaml

    cfg = _yaml.safe_load(open(train_yaml, encoding="utf-8")) or {}
except Exception as e:
    warn(f"训练配置读取失败 ({e}), 跳过资源检查")
    cfg = {}

model = cfg.get("model") or ""
if model and not str(model).endswith((".yaml", ".yml")) and str(model).lower() != "hf":
    w = PROJECT_ROOT / str(model)
    ok(f"预训练权重: {w}") if w.exists() else warn(
        f"预训练权重缺失: {w} (需下载 sam3.pt, 见 README 第 4 节)")

ds_ref = ((cfg.get("data") or {}).get("config")) or ""
if ds_ref:
    ds_path = PROJECT_ROOT / str(ds_ref)
    if not ds_path.exists():
        warn(f"数据集配置不存在: {ds_path}")
    else:
        ds = _yaml.safe_load(open(ds_path, encoding="utf-8")) or {}
        if "datasets" in ds:  # 多数据集类型
            entries = ds["datasets"]
            missing = [e["path"] for e in entries if not Path(e["path"]).exists()]
            ok(f"多数据集 {ds_ref}: {len(entries)} 个子集路径全部存在") if not missing \
                else warn(f"多数据集 {ds_ref}: {len(missing)}/{len(entries)} 个路径缺失: {missing[:3]}")
        else:  # 单数据集类型
            root = Path(ds.get("path", ""))
            for split_key in ("train", "val"):
                ann = root / ds.get(split_key, "") / ds.get("ann_file", "")
                ok(f"数据集 {split_key} 标注: {ann}") if ann.exists() else warn(
                    f"数据集 {split_key} 标注缺失: {ann}")

train_cfg = cfg.get("train") or {}
num_gpus = train_cfg.get("num_gpus")
device = train_cfg.get("device")
if torch is not None and torch.cuda.is_available() and (num_gpus or device):
    visible = torch.cuda.device_count()  # 注: 未设 CUDA_VISIBLE_DEVICES 时 = 物理卡数
    want = len(str(device).split(",")) if device else num_gpus
    if want and want <= visible:
        ok(f"训练配置需 {want} 卡 (num_gpus={num_gpus}, device={device}), 机器可见 {visible} 卡")
    else:
        warn(f"训练配置需 {want} 卡 (num_gpus={num_gpus}, device={device}), "
             f"但当前可见 {visible} 卡")

# ── 汇总 ──────────────────────────────────────────────────────────────────
print("=" * 60)
if n_fail:
    print(f"{_FAIL} 环境未就绪: {n_fail} 项必需检查失败, 请按上方提示安装/修复")
    sys.exit(1)
print(f"{_OK} 环境就绪 (WARN 项不阻断训练, 但建议补齐)")
sys.exit(0)
