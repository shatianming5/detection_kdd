# utils/

工具脚本目录，主要负责把 VOC 风格的 Repro 子集转换为 COCO 格式以供基线训练。

## 功能与作用

- `prepare_coco_data.py`：VOC → COCO 转换脚本
  - 读取 `Annotations/*.xml` 与 `JPEGImages/*.jpg`
  - 随机划分 train/val（默认 0.9/0.1）
  - 在目标目录下生成 `annotations/instances_{train2017,val2017}.json`
  - `train2017/`、`val2017/` 内对原图建立符号链接

## 使用方式

```bash
python prepare_coco_data.py <voc_root> <output_dir>
```

示例（生成 Repro-10k COCO 数据）：

```bash
python utils/prepare_coco_data.py repro_10k/VOC2012 baselines/data/repro_10k
```

## 如何验证功能是否实现

1. **脚本运行无报错**，输出提示 train/val 数量。
2. **输出结构存在**：
   - `<output_dir>/annotations/instances_train2017.json`
   - `<output_dir>/annotations/instances_val2017.json`
   - `<output_dir>/train2017/`、`val2017/`
3. **基线可读取**：
   - 使用 `baselines/detr` 或 `baselines/DiffusionDet` 的 smoke test 能正常开始训练，说明 COCO 格式符合预期。

