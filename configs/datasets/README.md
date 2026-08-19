# SAM 3 数据集配置说明
# =======================
#
# 数据集配置是独立 YAML, 由训练配置 (configs/train/xxx.yaml) 的 data.config 引用:
#
#   data:
#     config: configs/datasets/roboflow_vl_100.yaml
#
# 字段 (均为 sam3 后端 Sam3ImageDataset 的 COCO 格式假设):
#
#   path: /data/my_dataset            # 数据集根目录 (绝对路径, 或相对项目根)
#   train: train                      # 训练集子目录名 (内含 ann_file)
#   val: valid                        # 验证集子目录名 (只有 test/ 就改成 test)
#   ann_file: _annotations.coco.json  # COCO 标注文件名 (Roboflow 导出一般固定)
#   num_images: null                  # 限制训练图片数 (debug 用); null = 全部
#
# 前端会把以上字段翻译为 Hydra override:
#   paths.dataset_root, trainer.data.{train,val}.dataset.{img_folder,ann_file},
#   验证评测 GT 路径, trainer.data.train.dataset.limit_ids (num_images)
#
# 注意:
#   - 数据集注入依赖这些标准 Hydra 键, 请配合前端模板
#     sam3/sam3/train/configs/custom_image_ft.yaml 使用; 直接用后端自带配置
#     (roboflow_v100 等) 时, 数据路径在那些配置自己的 paths 段里改
#   - COCO categories[].name 即开放词表检测的文本 prompt, 类别名要有语义
#   - 分割训练需要 json 里带 segmentation (polygon 或 RLE), 并在模板里开启
#     (见 custom_image_ft.yaml 中 custom_data 末尾的 4 处联动说明)
