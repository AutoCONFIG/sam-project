# SAM 3 训练配置说明
# =====================
#
# 配置按用途分成独立 YAML (参照 yolo-project 的分离模式):
#   configs/train/xxx.yaml    — 训练入口配置 (本目录), mode: train, 由 sam.py 读取
#   configs/models/xxx.yaml   — 模型网络配置, 由训练配置的 model.config 引用
#   configs/datasets/xxx.yaml — 数据集配置 (COCO 格式), 由训练配置的 data.config 引用
#   configs/export/xxx.yaml   — ONNX 导出配置, mode: export (独立命令, 见 configs/export/)
#
# 训练涉及两层配置:
#   1. 前端训练配置 (本目录): model / data / train / output 四个分区
#   2. Hydra 训练配置: 完整的训练参数 (模型结构/transforms/优化器/loss),
#      由模型配置 (configs/models/) 里的 hydra_config 字段指向, 必须位于
#      sam3 子模块内 (Hydra initialize_config_module 要求配置必须在
#      sam3.train 包内, 见 sam3/sam3/train/train.py):
#       - 自定义数据微调模板 (推荐):
#           sam3/sam3/train/configs/custom_image_ft.yaml
#           本 fork 提供的完整模板 (COCO 图片微调), 复制改名后按需修改
#           (改动属于 sam3 fork 仓库)
#       - 子模块自带参考配置:
#           sam3/sam3/train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml  — Roboflow 100 全量微调
#           sam3/sam3/train/configs/odinw13/odinw_text_only_train.yaml                  — ODinW-13 文本训练
#
# 前端训练配置示例 (分区字段均按 sam3 后端的 Hydra 键翻译):
#
# mode: train
# model:
#   config: configs/models/sam3_image.yaml   # 模型网络配置 (独立 YAML, 见 configs/models/)
# data:
#   config: configs/datasets/roboflow_vl_100.yaml   # 数据集配置 (独立 YAML, 见 configs/datasets/)
# train:
#   # 资源
#   use_cluster: 0          # 0=本地, 1=SLURM 集群
#   num_gpus: 4             # 每节点 GPU 数
#   num_nodes: 1            # 节点数
#   device: "0,1,2,3"       # 指定用哪几张卡 (CUDA_VISIBLE_DEVICES); 空=全部可见卡
#   partition/account/qos   # SLURM 参数 (仅集群)
#   timeout_hour: 72        # → submitit.timeout_hour (仅集群)
#   cpus_per_task: 10       # → submitit.cpus_per_task (仅集群)
#   # 优化器 / 学习率
#   lr_scale: 0.1           # → scratch.lr_scale (各组 lr = base × lr_scale)
#   weight_decay: 0.1       # → scratch.wd
#   lrd: 0.9                # → scratch.lrd_vision_backbone (ViT 逐层衰减)
#   scheduler_timescale: 20 # → scratch.scheduler_timescale
#   scheduler_warmup: 20    # → scratch.scheduler_warmup
#   scheduler_cooldown: 20  # → scratch.scheduler_cooldown
#   grad_clip: 0.1          # → trainer.optim.gradient_clip.max_norm
#   amp: true               # → trainer.optim.amp.enabled
#   amp_dtype: bfloat16     # → trainer.optim.amp.amp_dtype
#   # 训练循环
#   batch: 1                # → scratch.train_batch_size
#   epochs: 20              # → trainer.max_epochs
#   grad_accum: 1           # → scratch.gradient_accumulation_steps
#   seed: 123               # → trainer.seed_value
#   val_freq: 10            # → trainer.val_epoch_freq
#   skip_first_val: true    # → trainer.skip_first_val
#   val_batch: 1            # → scratch.val_batch_size
#   # 数据加载
#   workers: 10             # → scratch.num_train_workers (Windows 建议 0)
#   val_workers: 0          # → scratch.num_val_workers
#   max_ann_per_img: 200    # → scratch.max_ann_per_img (单图最多标注数)
#   # 记录 / 保存
#   save_freq: 0            # → trainer.checkpoint.save_freq (0=只存最后一个)
#   log_freq: 10            # → trainer.logging.log_freq
#   # 任意 Hydra 覆盖 (原样透传, 同键靠后生效, 优先级最高) —— 未列长尾键都走这里
#   overrides:
#     - scratch.enable_segmentation=true   # 开分割训练 (还需模板里 3 处联动, 见模板注释)
# output:
#   path: runs/train/custom_ft    # 实验输出目录 → paths.experiment_log_dir
#
# 说明:
#   - 上面 train.* 的每个旋钮都映射到后端配置里确认存在的键 (已在官方 roboflow
#     参考配置与本 fork 模板中逐一核实), 未列出的长尾参数用 overrides 透传
#   - paths.bpe_path 由前端自动注入为子模块内绝对路径, 不用配
#   - data.config 的数据集注入依赖标准 Hydra 键 (paths.dataset_root /
#     trainer.data.{train,val}.dataset.{img_folder,ann_file}), 请配合模板
#     custom_image_ft.yaml 使用; 直接用后端自带配置时数据路径在其 paths 段里改
#
# 运行:
#   python sam.py configs/train/<your_config>.yaml
#
# CLI 覆盖 (与 YAML 同义, CLI 优先):
#   python sam.py configs/train/<your_config>.yaml --num-gpus 8 --batch-size 2
#   python sam.py configs/train/<your_config>.yaml --data configs/datasets/xxx.yaml
#   python sam.py configs/train/<your_config>.yaml --sam3-config sam3/sam3/train/configs/my_ft.yaml
#   python sam.py configs/train/<your_config>.yaml --override scratch.lr_scale=0.05
#   python sam.py configs/train/<your_config>.yaml --use-cluster 1 --partition <name>
#
# 重要提醒 (后端参考配置的默认值是 Meta 为集群评测设计的, 直接用会踩坑):
#   1. 参考配置 submitit.use_cluster 默认 True —— 本地训练必须在前端 YAML 显式 use_cluster: 0
#   2. 参考配置 trainer.skip_saving_ckpts 默认 true —— 微调前务必用 overrides 改成 false, 否则不存 checkpoint
#      (本 fork 模板 custom_image_ft.yaml 里已设为 false)
#   3. pretrained 不配则默认 load_from_HF=True, 自动从 HuggingFace 下载 sam3 原版权重 (gated repo 需 token),
#      建议显式指向本地 .pt
#   4. 数据格式: COCO (img_folder + _annotations.coco.json), 类别 name 即文本 prompt
#   5. resolution 修改要求训练镜像模型已支持 (fork 已打通 build_sam3_image_model(image_size=...)),
#      且只影响 image model; 微调产物用于 sam3.1 multiplex 推理时需注意分辨率一致
