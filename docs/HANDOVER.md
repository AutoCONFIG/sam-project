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
│   ├── predict.py                  # 推理 (文本提示, 输出形态跟随输入 + 标签导出)
│   ├── train.py                    # 训练 (subprocess 转发到 Hydra)
│   └── export.py                   # ONNX 导出 (multiplex 拆分 6 子模型)
├── core/
│   ├── engine.py                   # Sam3VideoPredictor 封装 (sam3 / sam3.1 双版本)
│   ├── visualization.py            # mask 叠加可视化
│   ├── labels.py                   # COCO + YOLO 标签导出 (框 + 多边形)
│   ├── io_dispatch.py             # 输入扫描 + 输出目录镜像
│   └── export/                     # ONNX 导出包 (wrappers + 6 个导出器 + utils)
├── utils/
│   ├── config.py                   # YAML 加载、深合并、argparse 工具
│   └── constants.py                # 默认常量
├── configs/
│   ├── predict/video_text.yaml     # sam3.1 推理配置示例
│   ├── predict/video_text_sam3.yaml# sam3 原版推理配置示例 (需 ≥24GB)
│   ├── export/default.yaml         # ONNX 导出配置示例
│   ├── train/                      # 训练入口配置 (model/resolution/data/train/output + README)
│   ├── models/                     # 模型配置 (只定义模型构建 trainer.model; 网络结构在后端代码里)
│   └── datasets/                   # 独立数据集配置 (COCO 格式, 由 train 配置的 data.config 引用)
├── docs/
│   ├── HANDOVER.md                 # 本文件
│   └── ONNX_EXPORT_PLAN.md         # ONNX 导出计划 + 第一版实现要点
├── pretrain/                       # 预训练权重 (.gitignore, 需手动下载)
│   ├── sam3.1/                     # SAM 3.1 multiplex 权重 + tokenizer
│   └── sam3/                       # SAM 3 原版权重 + tokenizer
├── sam3/                           # git submodule (SAM3.1 代码)
├── runs/                           # 输出目录 (.gitignore)
├── .gitignore / .gitmodules / README.md / requirements.txt
```

## 3. 功能完成度

### 3.1 统一框架 — ✅ 100%

- `sam.py` 读 YAML `mode` 字段 → `importlib` 分发到 `commands/<mode>.py`
- CLI 参数覆盖 YAML（深合并，CLI 显式指定的值覆盖 YAML）
- 支持 `python sam.py configs/xxx.yaml --flag value` 和 `python -m commands.xxx --config ...` 两种调用方式

### 3.2 推理 (predict) — ✅ 95%

**已验证跑通：**
- 文本提示视频推理：`python sam.py configs/predict/video_text.yaml`
- bedroom.mp4 200帧全部传播完成，检测到 2 个对象
- `core/engine.py` 的 `Sam3VideoPredictor` 类封装了 SAM3 的 session API

**双版本支持 (sam3 / sam3.1)：**
- `engine.py` 改走子模块统一入口 `build_sam3_predictor(version=...)`，按配置 `model.version` 选择
  - `sam3.1` — Object Multiplex，支持 `image_size` 参数化（16G 显卡用 672）。推荐默认。
  - `sam3`   — 原版 dense tracking，`image_size` 后端固定 1008（需 ≥24GB 显存）
- 两个版本共用同一套 `handle_request`/`handle_stream_request` API，predict.py 无需区分
- 配置示例：`configs/predict/video_text.yaml` (sam3.1) / `video_text_sam3.yaml` (sam3)

**输出形态跟随输入：**
- 输入是视频 → 输出 `<stem>.mp4`（vis 视频 + mask 视频）+ npz 帧序列 + 标签
- 输入是图片目录 → 镜像原目录结构输出 vis jpg + mask png + npz + 标签
- 输入是混合目录 → 递归扫描，视频/图片序列各自成处理单元，输出镜像原结构
- 模型只构建一次，多个输入单元顺序复用 session（start → process → close → 下一个）

**标签导出 (`core/labels.py`)：**
- COCO 格式：`images[]` + `annotations[]`（segmentation 多边形 + bbox + area）+ `categories[]`
- YOLO 格式：`det/<stem>.txt`（cls cx cy w h 归一化）+ `seg/<stem>.txt`（归一化多边形）+ `classes.txt`
- 类别名 = `prompt.text`，类别 id 按出现顺序自动编号
- 多边形用 `cv2.findContours` + `approxPolyDP` 从 binary mask 提取

**关键设计：**
- `image_size` 参数贯穿模型构建链（在 sam3 子模块的 `model_builder.py` 中实现，仅 sam3.1）
- `image_size != 1008` 时自动过滤 RoPE 位置编码 buffer（`freqs_cis` 等），避免 shape mismatch
- `model.finetune_ckpt` 支持加载微调权重：训练产出的 image model 裸 state_dict 在模型构建后灌进
  `predictor.model.detector`（strict=False），基础 checkpoint 仍提供 tracker 等其余权重（仅 sam3.1）
- 16GB 显卡用 `image_size=672`（须为 336 的倍数）
- 模型自带全局 bf16 autocast，权重保持 fp32，不要手动转 dtype
- `offload_video_to_cpu=True` 节省 GPU 显存
- `start_session` 需要磁盘路径，推理时先把帧写成临时 jpg 目录

**缺失项：**
- ❌ 点/框提示：predict 命令只有 `--text`，没有 `--points`/`--boxes`（SAM3 API 支持，只是没暴露）
- ❌ 实际推理流程尚未端到端验证（代码重构完成，待跑通 sam3.1 + 672）

**前后端对齐审计修复（2026-08-19）：**
- ✅ 修复 `version="sam3"` 必崩：engine 无条件传 `image_size`，而 sam3 原版 `Sam3VideoPredictor.__init__` 不接受该参数 → 现在仅 sam3.1 传
- ✅ 修复 `LabelExporter.add_frame` 重复 append 同一 FrameAnnotations（COCO 输出 images/annotations 整倍重复）
- ✅ 接通 `save_video`（原来解析后从未使用）；`DEFAULT_SAVE_VIDEO` 改为 True，与文档/示例 YAML 一致
- ✅ 统一图片扩展名集合到 `utils/constants.IMAGE_EXT`（原 io_dispatch 7 种 vs engine.get_frames 4 种不一致，webp/tiff 目录会扫描后加载失败）
- ✅ 删除死代码 `engine.save_frame_results`（重构残留，npz 格式与现行实现不一致）、`utils/config.resolve_config_value`（无人调用）、predict 恒等 rename 映射
- ✅ 已核验对齐：session 请求类型/字段、add_prompt 与 propagate 的 `{"frame_index", "outputs"` 返回结构、`out_obj_ids/out_binary_masks/out_boxes_xywh/out_probs` 输出键、multiplex predictor 继承 Sam3BasePredictor API 一致
- ⚠️ 已知未改：`sam3` 路径下 `use_fa3`/`use_rope_real` 被 `build_sam3_predictor` 吸收但不转发（静默无效）；`frame_index` 可从 `model.` 或 `prompt.` 两处读取（重复配置路径，行为兼容）

### 3.3 训练 (train) — ⚠️ 70%

**已实现：**
- `commands/train.py` 通过 subprocess 调用 `sam3/sam3/train/train.py -c <hydra_config>`
- CLI 参数透传（`--num-gpus`, `--use-cluster`, `--partition`, `--account`, `--qos`, `--num-nodes`）
- SAM3 Hydra 训练脚本存在，依赖已安装（hydra 1.3.5, submitit）
- 15+ 个 Hydra 训练配置文件存在于 `sam3/sam3/train/configs/`（roboflow_v100, odinw13, saco_video_evals 等）
- ✅ 前端训练配置分区为 `model`（标量多态：权重 `.pt`=预训练微调（默认模型配置 `configs/models/sam3_image.yaml`）/ 模型配置 `.yaml`=从头训练 / `hf` 或缺省=HF 下载微调）+ 顶层 `resolution` / `data`（`config` 引用独立数据集 YAML）/ `train`（资源与旋钮）/ `output`（`path` → `paths.experiment_log_dir`）/ `template`（训练模板，默认 `configs/train/template_image.yaml`）；顶层 `hydra_config` 为指向子模块现成配置的逃生舱，示例 `configs/train/custom_finetune.yaml`
- ✅ 配置三层拆分（2026-08-20）：**训练模板** `configs/train/template_image.yaml` = 后端 Hydra 配置全量默认值（transforms/loss/优化器/调度器/评测/分布式，即全部训练超参数）；**模型配置** `configs/models/sam3_image.yaml` 瘦身只剩 `trainer.model` 构建调用——本后端网络结构由 `build_sam3_image_model` 代码硬编码，yaml 本无网络结构可配（`scratch.d_model`/`pos_embed` 为上游遗留死配置，配置树无引用，模板里已注明）；物化 = 模板文本 + 模型配置 `trainer.model` 段在 `__MODEL_BLOCK__` 占位处合并（`commands/train.py: materialize_hydra_config`）
- ✅ 数据集配置独立成 `configs/datasets/*.yaml`（path/train/val/ann_file/num_images），前端翻译为 `paths.dataset_root` + `trainer.data.{train,val}.dataset.{img_folder,ann_file}` + 验证 GT 路径等 override；依赖模板标准键，配合训练模板 `configs/train/template_image.yaml` 使用（含这些键）
- ✅ `paths.bpe_path` 自动注入子模块内绝对路径，无需手配
- ✅ 训练旋钮全量前端直配（37 个，均映射到后端配置里逐一核实存在的键，默认值与后端一致，删掉/留空即不改）：batch/epochs/lr_scale/weight_decay/lrd/scheduler_timescale/scheduler_warmup/scheduler_cooldown/grad_accum/grad_clip/amp/amp_dtype/val_freq/skip_first_val/val_batch/early_stop/early_stop_patience/workers/val_workers/max_ann_per_img/save_freq/log_freq/skip_saving_ckpts/seed/timeout_hour/cpus_per_task + 损失权重/匹配器成本 11 个：loss_bbox/loss_giou/loss_ce/presence_loss/pos_weight/focal_alpha/focal_gamma/o2m_weight/matcher_cost_class/matcher_cost_bbox/matcher_cost_giou（映射表 `commands/train.py: TRAIN_KEY_MAP`；loss 权重覆盖 `custom_data.loss` 源——经插值进 `trainer.loss.all` 无法从 trainer 侧覆盖，`loss_fns_find` 列表顺序固定 0=Boxes 1=IABCEMdetr；`hydra_config` 逃生舱指向官方参考配置时前缀自动换为 `roboflow_train`，结构相同）；长尾参数直接编辑训练模板 `configs/train/template_image.yaml`
- ✅ subprocess 透传 `PYTHONPATH=<sam3 子模块>`，未 `pip install -e sam3` 也能 import
- ✅ `build_sam3_image_model(image_size=...)` 训练链路分辨率参数化（须为 336 倍数）
- ✅ 早停与真冻结（2026-08-20，后端 `trainer.py` 最小新增）：`trainer.early_stop`（enabled/patience/metric/mode/min_delta，按验证次数计，rank 0 判定 + broadcast；指标键后缀匹配，找不到只警告不误停）与 `trainer.freeze`（unix pattern 真冻结 `requires_grad=False`——不算梯度/不进优化器（经 `construct_optimizer` 的 `param_allowlist` 排除）/DDP 不同步，比 lr=0 省算力省显存）；默认值在训练模板 `configs/train/template_image.yaml` 的 trainer 节，开关与 patience 有前端旋钮

**后端机制备忘（2026-08-19 代码走读确认）：**
- 后端 `-c` 是 Hydra config 名，相对于 `sam3/sam3/train/`（`initialize_config_module` 限制，配置必须在子模块内）；前端已屏蔽此细节——训练模板与模型配置的 `trainer.model` 段在启动时文本合并生成到子模块 `configs/_custom/<模板文件名>.yaml`（自动管理，内容不变不重写，勿手改）；训练配置顶层 `hydra_config` 字段（或 CLI `--sam3-config`）为指向子模块内现成配置（如官方 roboflow 参考配置）的逃生舱
- 参考配置默认 `submitit.use_cluster: True`（本地必须显式 `--use-cluster 0`）且 `trainer.skip_saving_ckpts: true`（微调必须改 false，否则不存 checkpoint）
- 预训练权重：后端 `build_sam3_image_model(checkpoint_path=None, load_from_HF=True)`——`checkpoint_path` 优先，为 None 且 `load_from_HF=True` 时从 HF 下载 sam3 原版（gated repo 需 token），两者都空 = 从头训练；前端 `model` 字段：指 `.pt` → 注入 `++trainer.model.checkpoint_path`，指模型配置 yaml → 注入 `++trainer.model.load_from_HF=false`，`hf`/缺省 → 不注入
- 数据格式为 COCO（img_folder + `_annotations.coco.json`），文本 prompt = `categories[].name`；只支持图片级训练（video dataset 类存在但无训练配置）
- odinw 配置里的 `freeze_*` / `use_act_checkpoint_*` 键无代码消费，是死配置——官方开箱只有全量微调；真冻结与早停由本 fork 在 `trainer.py` 新增（见第 5 节）
- 前端注入的所有 Hydra 覆盖（旋钮/数据集/权重路径等）都作为尾随参数传给后端 `train.py`（fork 已改 `parse_known_args` 收集 + `compose(overrides=)`）；键在参考配置里不存在的用 `++` 前缀（前端已自动处理）；用户侧长尾参数直接编辑训练模板，不提供透传字段

**缺失项：**
- ❌ **没有实际跑通过训练**：只是 subprocess 转发，未验证端到端
- ✅ **checkpoint 管理已打通（2026-08-19，纯前端实现）**：predict 配置 `model.finetune_ckpt` 指向训练产出的 `checkpoint.pt`（image model 裸 state_dict），engine 在构建 predictor 后加载进 `predictor.model.detector`（strict=False，复用后端 `Sam3MultiplexBase` 的既定约定；RoPE buffer 一律丢弃用模型自身预计算的，训练/推理分辨率需一致）；仅支持 sam3.1，sam3 原版键结构不同

### 3.4 ONNX 导出 (export) — ⚠️ 第一版已实现，未实际运行验证

详见 `docs/ONNX_EXPORT_PLAN.md`（含实现要点与和原计划有出入的地方）。
- `commands/export.py` + `core/export/` 包 + `configs/export/default.yaml`，`sam.py` MODES 已挂 `export`
- 6 个子模型（A 图像编码器 / B 文本编码器 / C 提示编码器 / D 记忆注意力 / E 掩码解码器 / F 记忆编码器）全部实现，均来自 `build_sam3_multiplex_video_predictor` 一次构建（与 predict 同一构建链）
- 导出需要 CUDA 与额外依赖（onnx 必需；onnxruntime/onnxsim/onnxconverter-common 按需，见 requirements.txt 注释）
- ❌ **未跑通验证**：实现环境无 Python/GPU，只有静态代码走读；需在有环境的机器上跑 `python sam.py configs/export/default.yaml` 验证可导出性与数值一致性

## 4. 关键技术决策记录

| 决策 | 选择 | 原因 |
|------|------|------|
| 框架模式 | yolo-segdet 的 `yolo.py` 模式 | 用户熟悉，已验证可行 |
| sam3 后端 | git submodule 指向 fork | 隔离后端修改，便于版本管理 |
| 权重路径 | YAML 里写绝对路径 | 单机开发最直接 |
| 权重入库 | 不入库 (.gitignore) | 单文件 3.3GB 超 GitHub LFS 2GB 上限, 手动下载到 pretrain/ |
| train 命令 | subprocess 转发 | Hydra `initialize_config_module` 有全局状态约束，直接 import 会冲突 |
| 推理引擎 | `Sam3VideoPredictor` 类封装 | 便于扩展和后续 ONNX 导出复用 |
| 双版本 | 走子模块 `build_sam3_predictor(version=)` 统一入口 | 两版本共用 request API, engine 只需传 version |
| 输出形态 | 跟随输入 (视频→视频, 图片→图片目录镜像) | 输入什么样输出什么样, 直觉清晰 |
| 标签格式 | COCO + YOLO (框+多边形) | 兼容主流训练框架 (mmdet/yolo-segdet) |
| image_size | 参数化，默认 1008 | 16GB 显卡用 672，避免 OOM (仅 sam3.1) |
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

4. **`sam3/model_builder.py`** — 训练链路（image model）分辨率参数化（2026-08-19）：
   - `_create_vision_backbone(img_size=1008)` 贯穿 `position_encoding` 与 `_create_vit_backbone`
   - `build_sam3_image_model(image_size=1008)`（原 3 个 `_create_vision_backbone` 调用方不受影响，默认 1008）
   - `image_size != 1008` 时 `_load_checkpoint(drop_rope_buffers=True)` 过滤 `freqs_cis*` buffer
   - 训练侧用法：Hydra 配置里 `trainer.model.image_size` 与 `scratch.resolution` 一起改

5. **`sam3/train/train.py`** — 接受前端透传的 Hydra overrides（`parse_known_args` 收集 + `compose(overrides=)`）

6. **`sam3/train/trainer.py`** — 早停与真冻结（2026-08-20 新增，均为可选配置，官方配置不受影响）：
   - `Trainer.__init__` 新增 `freeze: List[str]` / `early_stop: Dict` 两个可选参数
   - `_apply_freeze()`：unix pattern 匹配参数名 → `requires_grad_(False)`（在 `_construct_optimizers` 之前调用）；冻结参数经 `construct_optimizer(param_allowlist=...)` 排除出优化器（不分配 AdamW 状态），此时跳过全参数覆盖校验
   - `_check_early_stop()` / `_early_stop_decision()`：`run_val` 返回验证指标 dict，`run_train` 在每次中间验证后判定；rank 0 判定 + `dist.broadcast_object_list` 广播；指标键按 精确→后缀→唯一子串 匹配，找不到只警告一次不早停

7. **`.gitignore`** — 忽略 `sam3/train/configs/_custom/`（前端启动训练时由训练模板 `configs/train/template_image.yaml` + 模型配置自动合并生成的 Hydra 配置目录）；原自定义模板 `custom_image_ft.yaml` 已移出子模块，完整平铺在主仓库训练模板里（相对 roboflow 参考配置：`skip_saving_ckpts: false`、`use_cluster: False`、无 job_array、`paths.dataset_root` 由前端数据集 YAML 注入）

## 6. 环境信息

```bash
# Python 环境
conda activate sam3
# Python 3.13, torch 2.10.0+cu128

# 模型权重位置 (pretrain/, .gitignore, 需手动从 HuggingFace 下载)
# sam3.1 (推荐, 16G 显卡可用):
pretrain/sam3.1/sam3.1_multiplex.pt  (3.3GB)
# sam3 原版 (需 ≥24GB 显存):
pretrain/sam3/sam3.pt  (3.3GB)  + pretrain/sam3/model.safetensors (HF transformers 格式, 可选)

# 测试视频
/media/yun/706bc403-c76c-4fdd-8a3f-d954b6189048/sam3/assets/videos/bedroom.mp4  (200帧, 540x960, 30fps)

# 快速验证推理 (sam3.1, 16G 显卡)
cd /media/yun/706bc403-c76c-4fdd-8a3f-d954b6189048/sam-project
conda run -n sam3 python sam.py configs/predict/video_text.yaml

# sam3 原版 (需 ≥24GB, 16G 显卡会 OOM)
conda run -n sam3 python sam.py configs/predict/video_text_sam3.yaml
```

## 6.1 预训练权重说明

SAM3 和 SAM3.1 是**同一模型的版本迭代**（不是不同任务）：
- **SAM 3.1** = SAM 3 + Object Multiplex（多对象共享内存，~7x 更快），推荐使用
- 两者架构都是 `Sam3VideoModel`，子模块 `sam3/` 代码库本身是 3.1 实现
- sam3.1 权重只提供 `.pt`（无 HF transformers 集成）；sam3 额外有 `.safetensors`（HF 格式）
- 权重不入库（3.3GB 超 GitHub LFS 单文件 2GB 上限），从 HuggingFace 下载到 `pretrain/`

## 7. 待办事项（按优先级）

1. **推理端到端验证** — 用 sam3.1 + 672 跑通重构后的 predict（输出形态跟随 + 标签导出）
2. **训练实际跑通验证** — 用 SAM3 自带的 roboflow 配置试跑（需下载数据集）
3. **微调工作流封装** — 预训练权重加载 → 自定义数据 fine-tune → checkpoint 保存
4. ~~训练配置模板~~（已完成：三层拆分——训练模板 `configs/train/template_image.yaml` 平铺全部训练超参数 + 模型配置 `configs/models/sam3_image.yaml` 只含 `trainer.model`，启动时文本合并生成进子模块 `configs/_custom/`；+ 独立 `configs/datasets/`）
5. **点/框提示推理** — predict 命令增加 `--points`/`--boxes` 参数
6. ~~checkpoint 管理~~（已完成：predict `model.finetune_ckpt` → detector 运行时加载，见 3.3 缺失项）
7. **Windows 推理+ONNX 导出评估** — 评估 Win 下能否跑推理和导出（基本开发用，重活留 Linux）
8. ~~**ONNX 导出**~~（第一版已实现，见 3.4）→ **导出实际跑通验证** — 在有 GPU 的环境跑 `configs/export/default.yaml` 并核对 onnxruntime 数值对比
