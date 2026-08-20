# SAM 3 训练配置说明
# =====================
#
# 配置按用途分成独立 YAML (参照 yolo-project 的分离模式):
#   configs/train/xxx.yaml    — 训练入口配置 (本目录), mode: train, 由 sam.py 读取
#   configs/models/xxx.yaml   — 模型配置 (hydra: 段平铺完整训练配置), 由训练配置的 model 字段引用
#   configs/datasets/xxx.yaml — 数据集配置 (COCO 格式), 由训练配置的 data.config 引用
#   configs/export/xxx.yaml   — ONNX 导出配置, mode: export (独立命令, 见 configs/export/)
#
# 训练涉及两层配置:
#   1. 前端训练配置 (本目录): model / resolution / data / train / output
#   2. Hydra 训练配置: 完整的训练参数 (模型结构/transforms/优化器/loss),
#      完整平铺在模型配置 (configs/models/sam3_image.yaml) 的 hydra: 段里,
#      启动训练时前端原样生成到子模块 sam3/sam3/train/configs/_custom/
#      <文件名>.yaml 再传给后端 (Hydra initialize_config_module 要求配置
#      必须在 sam3.train 包内, 见 sam3/sam3/train/train.py; 生成文件勿手改)
#
# model 字段 (标量; 预训练微调不改变网络结构, 指权重即可):
#   model: pretrain/sam3/sam3.pt           # 权重 .pt = 预训练微调 (默认模型配置 sam3_image.yaml)
#   model: configs/models/sam3_image.yaml  # 模型配置 .yaml = 从头训练
#   model: hf                              # 或不写 = HF 自动下载官方权重微调 (gated repo 需 token)
#
# 例外 (逃生舱): 要直接复用子模块内的现成配置时, 用顶层 hydra_config 字段
# (或 CLI --sam3-config) 指向它, 见 configs/train/roboflow_finetune.yaml:
#   sam3/sam3/train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml  — Roboflow 100 全量微调
#   sam3/sam3/train/configs/odinw13/odinw_text_only_train.yaml                  — ODinW-13 文本训练
#
# 前端训练配置示例 (分区字段均按 sam3 后端的 Hydra 键翻译):
#
# mode: train
# model: pretrain/sam3/sam3.pt   # 权重 .pt=微调 / 模型配置 .yaml=从头训练 / hf 或不写=HF 下载微调
# resolution: 1008               # 训练分辨率, 336 的倍数
# # hydra_config: sam3/sam3/train/configs/...   # 逃生舱: 直接用子模块内现成配置
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
#   skip_saving_ckpts: false  # → trainer.skip_saving_ckpts (微调必须 false)
# output:
#   path: runs/train/custom_ft    # 实验输出目录 → paths.experiment_log_dir
#
# 说明:
#   - 上面 train.* 的每个旋钮都映射到后端配置里确认存在的键 (已对照官方 roboflow
#     参考配置与 sam3_image.yaml 的 hydra 段逐一核实); 未列出的长尾参数直接编辑
#     模型配置的 hydra 段 (那里是完整平铺的全部训练配置)
#   - paths.bpe_path 由前端自动注入为子模块内绝对路径, 不用配
#   - data.config 的数据集注入依赖标准 Hydra 键 (paths.dataset_root /
#     trainer.data.{train,val}.dataset.{img_folder,ann_file}), 请配合
#     configs/models/sam3_image.yaml 使用 (其 hydra 段含这些键); 直接用后端
#     自带配置时数据路径在其 paths 段里改
#
# 运行:
#   python sam.py configs/train/<your_config>.yaml
#
# CLI 覆盖 (与 YAML 同义, CLI 优先):
#   python sam.py configs/train/<your_config>.yaml --model pretrain/sam3/sam3.pt --resolution 672
#   python sam.py configs/train/<your_config>.yaml --num-gpus 8 --batch-size 2
#   python sam.py configs/train/<your_config>.yaml --data configs/datasets/xxx.yaml
#   python sam.py configs/train/<your_config>.yaml --sam3-config sam3/sam3/train/configs/my_ft.yaml
#   python sam.py configs/train/<your_config>.yaml --use-cluster 1 --partition <name>
#
# 重要提醒 (后端参考配置的默认值是 Meta 为集群评测设计的, 直接用会踩坑):
#   1. 参考配置 submitit.use_cluster 默认 True —— 本地训练必须在前端 YAML 显式 use_cluster: 0
#   2. 参考配置 trainer.skip_saving_ckpts 默认 true —— 微调前务必显式 skip_saving_ckpts: false,
#      否则不存 checkpoint (configs/models/sam3_image.yaml 的 hydra 段已设为 false)
#   3. model 不写或写 hf 则默认 load_from_HF=True, 自动从 HuggingFace 下载 sam3 原版权重
#      (gated repo 需 token), 建议显式指向本地 .pt
#   4. 数据格式: COCO (img_folder + _annotations.coco.json), 类别 name 即文本 prompt
#   5. resolution 修改要求训练镜像模型已支持 (fork 已打通 build_sam3_image_model(image_size=...)),
#      且只影响 image model; 微调产物用于 sam3.1 multiplex 推理时需注意分辨率一致
