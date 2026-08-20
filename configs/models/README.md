# SAM 3 模型配置说明
# ==================
#
# 模型配置由训练配置 (configs/train/xxx.yaml) 的 model 字段引用:
#
#   model: configs/models/sam3_image.yaml   # 指向模型配置 = 从头训练
#   model: pretrain/sam3/sam3.pt            # 指向权重 = 预训练微调 (此时默认模型配置 = sam3_image.yaml)
#   model: hf                               # 或不写 = HF 自动下载官方权重微调 (gated repo 需 token)
#
# SAM3 的网络结构不像 YOLO 那样由 YAML 模块化拼装 —— 结构定义是 Hydra 配置的
# 一部分, 且预训练微调不改变网络结构, 所以指权重时无需再指模型配置。
# 本目录的模型配置是**自包含**的: hydra: 段 = 完整 Hydra 训练配置的全部平铺
# (网络结构/transforms/loss/优化器/分布式/日志等), 默认值已按官方参考配置调好,
# 一般不用改; 要改高级项时直接编辑这一段。启动训练时前端把本段原样生成到子模块
# sam3/sam3/train/configs/_custom/<文件名>.yaml 再传给后端
# (Hydra 要求配置必须在 sam3.train 包内; 生成的文件勿手改)。
#
# 例外: 要直接复用子模块内的现成配置 (如 Meta 官方参考配置, 平铺拷贝会过时),
# 用训练配置顶层的 hydra_config 字段 (或 CLI --sam3-config) 指向它, 见
# configs/train/roboflow_finetune.yaml。
#
# 现有配置:
#   sam3_image.yaml — 自定义数据微调 (自包含 hydra 段, 默认模型配置)
