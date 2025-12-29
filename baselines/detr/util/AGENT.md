# baselines/detr/util/

DETR 训练/推理中通用的工具函数与结构体。

## 功能与作用

- 提供分布式训练辅助、日志平滑、`NestedTensor` 等通用组件。
- 实现 bbox 变换、IoU/GIoU 计算等几何函数。
- 提供训练曲线与结果可视化工具（可选）。

## 结构与关键文件

- `misc.py`：
  - 分布式初始化、all-reduce、进度/日志辅助。
  - `NestedTensor` 及相关构造函数，供模型统一处理 padding 后的多尺寸图像。
- `box_ops.py`：
  - `box_cxcywh_to_xyxy` / `box_xyxy_to_cxcywh` 转换。
  - IoU/GIoU、面积、NMS 相关操作。
- `plot_utils.py`：绘制 loss/指标曲线（上游工具，可选使用）。
- `__init__.py`：导出。

## 如何验证功能是否实现

- 直接方式：跑 DETR smoke test/完整训练，若训练过程中无分布式/几何计算相关报错，说明 util 组件正常。
- （可选）单独验证 bbox ops：
  ```bash
  python -c "from util.box_ops import box_cxcywh_to_xyxy; import torch; print(box_cxcywh_to_xyxy(torch.tensor([[0.5,0.5,1.0,1.0]])))"
  ```

