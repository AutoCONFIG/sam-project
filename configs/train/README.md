# SAM 3 训练配置说明
# =====================
#
# 训练涉及两层配置:
#   1. 前端转发配置 (本目录, 如 roboflow_finetune.yaml):
#      mode: train + train.config/use_cluster/num_gpus 等, 由 sam.py 读取
#   2. 训练配置 (Hydra): 真正的训练参数 (数据路径/模型/优化器/loss),
#      由前端配置的 train.config 字段指向
#
# train.config 写前端视角的路径 (相对于项目根或绝对路径):
#   - 子模块参考配置:
#       sam3/sam3/train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml  — Roboflow 100 全量微调
#       sam3/sam3/train/configs/odinw13/odinw_text_only_train.yaml                  — ODinW-13 文本训练
#   - 前端自定义配置 (自定义训练推荐):
#       复制参考配置到 configs/train/hydra/my_ft.yaml, 按需修改后, train.config 直接写
#       configs/train/hydra/my_ft.yaml; 启动时会自动同步到子模块
#       sam3/sam3/train/configs/_custom/ (Hydra initialize_config_module 要求配置
#       必须在 sam3.train 模块内, 见 sam3/sam3/train/train.py)
#   (向后兼容: 也接受纯 Hydra 名, 如 configs/roboflow_v100/..., 相对于 sam3/sam3/train/)
#
# 前端转发配置示例:
#
# mode: train
# train:
#   config: sam3/sam3/train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml
#   use_cluster: 0    # 0=本地, 1=SLURM 集群
#   num_gpus: 4       # 每节点 GPU 数
#   num_nodes: 1      # 节点数
#
# 运行:
#   python sam.py configs/train/<your_config>.yaml
#
# CLI 覆盖:
#   python sam.py configs/train/<your_config>.yaml --num-gpus 8
#   python sam.py configs/train/<your_config>.yaml --use-cluster 1 --partition <name>
#
# 重要提醒 (后端参考配置的默认值是 Meta 为集群评测设计的, 直接用会踩坑):
#   1. 参考配置 submitit.use_cluster 默认 True —— 本地训练必须在前端 YAML 显式 use_cluster: 0
#   2. 参考配置 trainer.skip_saving_ckpts 默认 true —— 微调前在 Hydra 配置里改成 false, 否则不存 checkpoint
#   3. 加载本地预训练权重: 在 Hydra 配置 trainer.model 下加 checkpoint_path: /path/to/sam3.pt
#      (不配则默认 load_from_HF=True, 自动从 HuggingFace 下载 sam3 原版权重, gated repo 需 token)
#   4. 数据格式: COCO (img_folder + _annotations.coco.json), 类别 name 即文本 prompt
#   5. 训练分辨率: Hydra 配置里 scratch.resolution 和 trainer.model.image_size 一起改
#      (须为 336 的倍数, 如 672; 默认 1008)
