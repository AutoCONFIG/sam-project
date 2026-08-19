# SAM3 Project 交接文档

> 最后更新：2026-08-19
> 仓库：`git@github.com:AutoCONFIG/sam-project.git`
> 子模块：`git@github.com:AutoCONFIG/sam3.git` (fork, 分支 main)

---

## 1. 项目概述

`sam-project` 是 SAM3 (Segment Anything Model 3) 的统一 CLI 框架，参照 yolo-segdet 的 `yolo.py` 模式，实现 YAML 配置挂载 + CLI 覆盖的统一入口。

- **主仓库**：`/media/yun/706bc403-c76c-4fdd-8a3f-d954b6189048/sam-project/` → `AutoCONFIG/sam-project.git`
- **后端子模块**：`sam3/` → `AutoCONFIG/sam3.git` (fork of facebookresearch/sam3)
- **Python 环境**：conda env `sam3` (Python 3.13, torch 2.10.0+cu128)
- **GPU**：RTX 5060 Ti (Blackwell sm_120, 16GB)

## 2. 目录结构

```
sam-project/
├── sam.py                          # 统一入口: 读 YAML mode → 分发到 commands/
├── commands/
│   ├── predict.py                  # 视频推理 (文本提示)
│   └── train.py                    # 训练 (subprocess 转发到 Hydra)
├── core/
│   ├── engine.py                   # Sam3VideoPredictor 封装
│   └── visualization.py            # mask 叠加可视化
├── utils/
│   ├── config.py                   # YAML 加载、深合并、argparse 工具
│   └── constants.py                # 默认常量
├── configs/
│   ├── predict/video_text.yaml     # 推理配置示例
│   └── train/README.md             # 训练配置说明
├── docs/
│   ├── HANDOVER.md                 # 本文件
│   └── ONNX_EXPORT_PLAN.md         # ONNX 导出计划 (暂不实现)
├── sam3/                           # git submodule
├── runs/                           # 输出目录 (.gitignore)
├── .gitignore / .gitmodules / README.md / requirements.txt
```

## 3. 功能完成度

### 3.1 统一框架 — ✅ 100%

- `sam.py` 读 YAML `mode` 字段 → `importlib` 分发到 `commands/<mode>.py`
- CLI 参数覆盖 YAML（深合并，CLI 显式指定的值覆盖 YAML）
- 支持 `python sam.py configs/xxx.yaml --flag value` 和 `python -m commands.xxx --config ...` 两种调用方式

### 3.2 推理 (predict) — ✅ 90%

**已验证跑通：**
- 文本提示视频推理：`python sam.py configs/predict/video_text.yaml --image-size 672`
- bedroom.mp4 200帧全部传播完成，检测到 2 个对象
- 输出：200 张 vis JPG（mask 叠加 + box + id + score）+ 200 个 mask NPZ（label_map + meta）
- `core/engine.py` 的 `Sam3VideoPredictor` 类封装了 SAM3 的 session API

**关键设计：**
- `image_size` 参数贯穿模型构建链（在 sam3 子模块的 `model_builder.py` 中实现）
- `image_size != 1008` 时自动过滤 RoPE 位置编码 buffer（`freqs_cis` 等），避免 shape mismatch
- 16GB 显卡用 `image_size=672`（须为 336 的倍数）
- 模型自带全局 bf16 autocast，权重保持 fp32，不要手动转 dtype
- `offload_video_to_cpu=True` 节省 GPU 显存
- `start_session` 需要磁盘路径，推理时先把帧写成临时 jpg 目录

**缺失项：**
- ❌ 点/框提示：predict 命令只有 `--text`，没有 `--points`/`--boxes`（SAM3 API 支持，只是没暴露）
- ❌ 单张图像推理：只支持视频/帧目录
- ❌ `--save-video` 合成 mp4 功能写了但未测试

### 3.3 训练 (train) — ⚠️ 60%

**已实现：**
- `commands/train.py` 通过 subprocess 调用 `sam3/sam3/train/train.py -c <hydra_config>`
- CLI 参数透传（`--num-gpus`, `--use-cluster`, `--partition`, `--account`, `--qos`, `--num-nodes`）
- SAM3 Hydra 训练脚本存在，依赖已安装（hydra 1.3.5, submitit）
- 15+ 个 Hydra 训练配置文件存在于 `sam3/sam3/train/configs/`（roboflow_v100, odinw13, saco_video_evals 等）

**缺失项：**
- ❌ **没有实际跑通过训练**：只是 subprocess 转发，未验证端到端
- ❌ **没有训练用 YAML 配置示例**：`configs/train/` 只有 README，没有实际 `.yaml`
- ❌ **没有微调工作流**：用户需要"预训练+微调"，当前只转发到 SAM3 原始训练系统
- ❌ **没有 checkpoint 管理**：训练产出 → 推理加载的衔接缺失

### 3.4 ONNX 导出 — 📄 计划已存，暂不实现

详见 `docs/ONNX_EXPORT_PLAN.md`。SAM3 无内置导出功能，需拆分为 6 个子模型分别导出。

## 4. 关键技术决策记录

| 决策 | 选择 | 原因 |
|------|------|------|
| 框架模式 | yolo-segdet 的 `yolo.py` 模式 | 用户熟悉，已验证可行 |
| sam3 后端 | git submodule 指向 fork | 隔离后端修改，便于版本管理 |
| 权重路径 | YAML 里写绝对路径 | 单机开发最直接 |
| train 命令 | subprocess 转发 | Hydra `initialize_config_module` 有全局状态约束，直接 import 会冲突 |
| 推理引擎 | `Sam3VideoPredictor` 类封装 | 便于扩展和后续 ONNX 导出复用 |
| image_size | 参数化，默认 1008 | 16GB 显卡用 672，避免 OOM |
| bf16 | 模型自带全局 autocast | decoder FFN 有 `autocast(enabled=False)`，手动转权重会冲突 |

## 5. sam3 子模块的关键修改

fork 的 main 分支包含以下修改（commit `bfa05a7`）：

1. **`sam3/model_builder.py`** — `image_size` 参数贯穿 multiplex 构建链：
   - `_create_vit_backbone(img_size=1008)`
   - `_create_multiplex_tri_backbone(img_size=1008)`
   - `_create_multiplex_maskmem_backbone(image_size=1008)` — `feat_size=image_size//14`
   - `_create_multiplex_transformer(image_size=1008)` — `feat_sizes=[feat_size, feat_size]`
   - `build_sam3_multiplex_video_model(image_size=1008)`
   - `build_sam3_multiplex_video_predictor(image_size=1008)`
   - `image_size != 1008` 时过滤 `freqs_cis`/`freqs_cis_real`/`freqs_cis_imag` buffer

2. **`sam3/model/sam3_base_predictor.py`** — `start_session` 用 `inspect` 过滤 `init_state` 不接受的 kwargs

3. **`infer_video.py`** — 独立推理脚本（sam-project 的 `core/engine.py` 是它的工程化重构）

## 6. 环境信息

```bash
# Python 环境
conda activate sam3
# Python 3.13, torch 2.10.0+cu128

# 模型权重位置
/media/yun/706bc403-c76c-4fdd-8a3f-d954b6189048/sam3/sam3.1/sam3.1_multiplex.pt  (3.3GB)

# 测试视频
/media/yun/706bc403-c76c-4fdd-8a3f-d954b6189048/sam3/assets/videos/bedroom.mp4  (200帧, 540x960, 30fps)

# 快速验证推理
cd /media/yun/706bc403-c76c-4fdd-8a3f-d954b6189048/sam-project
conda run -n sam3 python sam.py configs/predict/video_text.yaml --image-size 672
```

## 7. 待办事项（按优先级）

1. **训练实际跑通验证** — 用 SAM3 自带的 roboflow 配置试跑（需下载数据集）
2. **微调工作流封装** — 预训练权重加载 → 自定义数据 fine-tune → checkpoint 保存
3. **训练配置模板** — 提供可直接使用的 YAML 模板
4. **点/框提示推理** — predict 命令增加 `--points`/`--boxes` 参数
5. **checkpoint 管理** — 训练产出 → 推理加载的衔接
6. **ONNX 导出** — 按 `docs/ONNX_EXPORT_PLAN.md` 实现（暂缓）
