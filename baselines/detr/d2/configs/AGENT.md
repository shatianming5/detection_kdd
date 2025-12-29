# baselines/detr/d2/configs/

Detectron2 wrapper 使用的 YAML 配置文件。

## 功能与作用

- 定义 DETR 在 Detectron2 框架下的模型、数据、solver、评估等超参。

## 结构与关键文件

- `detr_256_6_6_torchvision.yaml`：默认 DETR 检测配置（R50 backbone，256 hidden dim，6e/6d layers）。
- `detr_segm_256_6_6_torchvision.yaml`：分割微调配置（开启 mask 分支，通常需加载 frozen weights）。

## 如何验证功能是否实现

- 运行 Detectron2 wrapper：
  ```bash
  python ../train_net.py --config detr_256_6_6_torchvision.yaml --num-gpus 1 --eval-only
  ```
  若配置可被解析并启动（即便无权重也能走到构建阶段），说明 YAML 与 `add_detr_config` 匹配。

