# SAM 3 训练配置说明
# =====================
#
# SAM 3 训练使用 Hydra 配置管理系统, 配置文件位于:
#   sam3/sam3/train/configs/
#
# 可用的训练配置示例:
#   roboflow_v100/roboflow_v100_full_ft_100_images.yaml  — Roboflow 100 微调
#   odinw13/odinw_text_only_train.yaml                    — ODinW-13 文本训练
#
# 完整列表见 sam3/README_TRAIN.md
#
# 创建训练配置 YAML:
#
# mode: train
# train:
#   config: roboflow_v100/roboflow_v100_full_ft_100_images.yaml  # Hydra config 名
#   use_cluster: 0    # 0=本地, 1=SLURM 集群
#   num_gpus: 1       # 每节点 GPU 数
#   num_nodes: 1      # 节点数
#
# 运行:
#   python sam.py configs/train/<your_config>.yaml
#
# CLI 覆盖:
#   python sam.py configs/train/<your_config>.yaml --num-gpus 2
#   python sam.py configs/train/<your_config>.yaml --use-cluster 1 --partition <name>
