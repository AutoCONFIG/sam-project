# SAM 3 模型网络配置说明
# ========================
#
# 模型网络配置是独立 YAML, 由训练配置 (configs/train/xxx.yaml) 的 model.config 引用:
#
#   model:
#     config: configs/models/sam3_image.yaml
#
# SAM3 的网络结构不像 YOLO 那样由 YAML 模块化拼装 —— 结构定义固定在 Hydra
# 配置里 (hydra_config 指向的模板, 位于 sam3 子模块内), 官方默认值已调好,
# 一般不需要改。本目录的配置只暴露最常用的网络相关项:
#
#   hydra_config: sam3/sam3/train/configs/custom_image_ft.yaml
#       # 完整 Hydra 训练配置 (网络结构/transforms/loss/优化器), 必须位于
#       # sam3 子模块内; fork 自带自定义数据微调模板, 官方参考配置也在同目录
#   pretrained: true
#       # 预训练权重: true=HF 自动下载 sam3 原版 (gated repo 需 token) /
#       # 路径=指定本地 ckpt / false=从零训练
#   resolution: 1008
#       # 网络输入分辨率, 必须是 336 的倍数; RoPE 位置编码按分辨率预计算,
#       # 数据 pipeline 会由前端同步改 (scratch.resolution + trainer.model.image_size)
#   overrides: []
#       # 网络相关的任意 Hydra 覆盖 (如 trainer.model.enable_segmentation=true),
#       # 同键时训练配置的 train.overrides 优先
#
# 现有配置:
#   sam3_image.yaml        — 自定义数据微调 (配 fork 模板 custom_image_ft.yaml)
#   sam3_roboflow_ref.yaml — 官方 Roboflow 100 参考配置 (复现/对照用)
