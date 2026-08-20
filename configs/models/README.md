# SAM 3 模型网络配置说明
# ========================
#
# 模型网络配置是独立 YAML, 由训练配置 (configs/train/xxx.yaml) 的 model.config 引用:
#
#   model:
#     config: configs/models/sam3_image.yaml
#
# SAM3 的网络结构不像 YOLO 那样由 YAML 模块化拼装 —— 结构定义是 Hydra 配置的
# 一部分。本目录的模型配置是**自包含**的, 由两部分组成:
#
#   1. 前端常用项:
#      pretrained: true     # 预训练权重: true=HF 自动下载 (gated repo 需 token) /
#                           # 路径=指定本地 ckpt / false=从零训练
#      resolution: 1008     # 网络输入分辨率, 必须是 336 的倍数; RoPE 位置编码按
#                           # 分辨率预计算, 数据 pipeline 由前端同步改
#      overrides: []        # 网络相关的任意 Hydra 覆盖 (同键时 train.overrides 优先)
#
#   2. hydra: 段 — 完整 Hydra 训练配置的全部平铺 (网络结构/transforms/loss/
#      优化器/分布式/日志等), 默认值已按官方参考配置调好, 一般不用改;
#      要改高级项时直接编辑这一段。启动训练时前端把本段原样生成到子模块
#      sam3/sam3/train/configs/_custom/<文件名>.yaml 再传给后端
#      (Hydra 要求配置必须在 sam3.train 包内; 生成的文件勿手改)。
#
# 例外: 直接复用子模块内的现成配置 (如 Meta 官方参考配置, 平铺拷贝会过时)
# 时, 可以不写 hydra: 段, 改用 hydra_config 字段指向它, 见
# sam3_roboflow_ref.yaml。
#
# 现有配置:
#   sam3_image.yaml        — 自定义数据微调 (自包含 hydra 段, 推荐)
#   sam3_roboflow_ref.yaml — 官方 Roboflow 100 参考配置 (hydra_config 指针, 复现/对照用)
