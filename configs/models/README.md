# SAM 3 模型配置说明
# ==================
#
# 模型配置由训练配置 (configs/train/xxx.yaml) 的 model 字段引用:
#
#   model: configs/models/sam3_image.yaml   # 指向模型配置 = 从头训练
#   model: pretrain/sam3/sam3.pt            # 指向权重 = 预训练微调 (此时默认模型配置 = sam3_image.yaml)
#   model: hf                               # 或不写 = HF 自动下载官方权重微调 (gated repo 需 token)
#
# SAM3 的网络结构不像 YOLO 那样由 YAML 模块化拼装 —— ViT 主干 / transformer /
# 检测头都由后端 sam3/sam3/model_builder.py 的 build_sam3_image_model 代码构建,
# yaml 里没有网络结构可配。所以本目录的模型配置很薄: 只有 trainer.model 一段
# 构建调用 (_target_ / bpe_path / enable_segmentation 等构建参数)。
# 预训练微调不改变网络结构, 指权重时无需再指模型配置。
#
# 训练超参数 (transforms/loss/优化器/调度器/评测等全部默认值) 都在训练配方
# configs/train/example/recipe_image.yaml —— 包括训练策略相关的两个高级项
# (本 fork 新增的后端支持, 在配方的 trainer 节里):
#   freeze: []        — 真冻结 (requires_grad=False: 不算梯度/不进优化器/DDP 不同步,
#                       比把某组 lr 设为 0 省算力省显存), unix pattern 列表, 如
#                       ["backbone.vision_backbone.*", "backbone.language_backbone.*"]
#                       (全部可冻模块清单见 configs/train/custom_finetune.yaml 的
#                        freeze 段注释)
#   early_stop:       — 早停 (enabled/patience/metric/mode/min_delta), 开关与
#                       patience 也可在训练入口配置的 train.early_stop* 旋钮里调
#
# 启动训练时前端把模型配置的 trainer.model 段文本合并进训练配方, 生成到子模块
# sam3/sam3/train/configs/_custom/<配方文件名>.yaml 再传给后端
# (Hydra 要求配置必须在 sam3.train 包内; 生成的文件勿手改)。
#
# 例外: 要直接复用子模块内的现成配置 (如 Meta 官方参考配置, 平铺拷贝会过时),
# 用训练配置顶层的 hydra_config 字段 (或 CLI --sam3-config) 指向它, 见
# configs/train/roboflow_finetune.yaml。
#
# 现有配置:
#   sam3_image.yaml — 图片模型构建定义 (默认模型配置)
