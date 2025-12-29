from __future__ import annotations

from typing import Sequence

import torch
from mmengine.registry import FUNCTIONS

from mmdet.structures.bbox import BaseBoxes
from mmdet.structures.det_data_sample import DetDataSample


def _boxes_to_tensor(boxes: BaseBoxes | torch.Tensor) -> torch.Tensor:
    if isinstance(boxes, torch.Tensor):
        return boxes
    if isinstance(boxes, BaseBoxes):
        return boxes.tensor
    raise TypeError(f"Unsupported boxes type: {type(boxes)}")


def _sanitize_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    x1n = torch.minimum(x1, x2)
    x2n = torch.maximum(x1, x2)
    y1n = torch.minimum(y1, y2)
    y2n = torch.maximum(y1, y2)
    out = torch.stack([x1n, y1n, x2n, y2n], dim=-1)
    return out


def _random_noise_boxes(num: int, *, height: int, width: int, device: torch.device) -> torch.Tensor:
    if num <= 0:
        return torch.empty((0, 4), device=device, dtype=torch.float32)
    x1 = torch.rand((num,), device=device, dtype=torch.float32) * float(width)
    y1 = torch.rand((num,), device=device, dtype=torch.float32) * float(height)
    x2 = torch.rand((num,), device=device, dtype=torch.float32) * float(width)
    y2 = torch.rand((num,), device=device, dtype=torch.float32) * float(height)
    boxes = torch.stack([x1, y1, x2, y2], dim=-1)
    boxes = _sanitize_xyxy(boxes)
    # avoid degenerate boxes
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    x2 = torch.maximum(x2, x1 + 1e-3)
    y2 = torch.maximum(y2, y1 + 1e-3)
    return torch.stack([x1, y1, x2, y2], dim=-1)


@FUNCTIONS.register_module()
def coco_graph_collate(
    data_batch: Sequence[dict],
    *,
    num_nodes: int,
    unk_id: int,
    shuffle_nodes: bool = True,
) -> dict:
    """
    MMEngine collate_fn that augments each DetDataSample with a fixed-N "graph"
    (gt nodes + random noise nodes + unk labels).

    Expected per-sample format (after `PackDetInputs`):
    - {"inputs": CHW Tensor, "data_samples": DetDataSample}

    Adds fields to each DetDataSample:
    - graph_boxes: (N,4) xyxy abs float32
    - graph_labels: (N,) int64, padding=unk_id
    - graph_mask: (N,) bool, True=gt nodes
    - graph_image_size: (2,) int64 (H, W)
    """
    if num_nodes <= 0:
        raise ValueError(f"num_nodes must be > 0, got {num_nodes}")

    inputs = []
    data_samples: list[DetDataSample] = []

    for item in data_batch:
        if "inputs" not in item or "data_samples" not in item:
            raise KeyError("Each batch item must contain keys: `inputs`, `data_samples`.")
        img = item["inputs"]
        sample = item["data_samples"]
        if not isinstance(sample, DetDataSample):
            raise TypeError(f"data_samples must be DetDataSample, got {type(sample)}")

        inputs.append(img)
        data_samples.append(sample)

        img_shape = sample.metainfo.get("img_shape", None)
        if img_shape is None:
            # Fallback: CHW tensor.
            height, width = int(img.shape[-2]), int(img.shape[-1])
        else:
            height, width = int(img_shape[0]), int(img_shape[1])

        if hasattr(sample, "gt_instances") and sample.gt_instances is not None and "bboxes" in sample.gt_instances:
            gt_boxes = _boxes_to_tensor(sample.gt_instances.bboxes).to(dtype=torch.float32)
            gt_labels = sample.gt_instances.labels.to(dtype=torch.long)
        else:
            gt_boxes = torch.empty((0, 4), dtype=torch.float32)
            gt_labels = torch.empty((0,), dtype=torch.long)

        gt_boxes = _sanitize_xyxy(gt_boxes)
        m = int(gt_boxes.shape[0])
        m_keep = min(m, int(num_nodes))
        gt_boxes = gt_boxes[:m_keep]
        gt_labels = gt_labels[:m_keep]

        pad = int(num_nodes) - m_keep
        if pad > 0:
            noise_boxes = _random_noise_boxes(pad, height=height, width=width, device=gt_boxes.device)
            graph_boxes = torch.cat([gt_boxes, noise_boxes], dim=0)
            graph_labels = torch.cat(
                [gt_labels, torch.full((pad,), int(unk_id), dtype=torch.long, device=gt_labels.device)],
                dim=0,
            )
            graph_mask = torch.cat(
                [
                    torch.ones((m_keep,), dtype=torch.bool, device=gt_labels.device),
                    torch.zeros((pad,), dtype=torch.bool, device=gt_labels.device),
                ],
                dim=0,
            )
        else:
            graph_boxes = gt_boxes
            graph_labels = gt_labels
            graph_mask = torch.ones((num_nodes,), dtype=torch.bool, device=gt_labels.device)

        if shuffle_nodes:
            perm = torch.randperm(int(num_nodes), device=graph_boxes.device)
            graph_boxes = graph_boxes[perm]
            graph_labels = graph_labels[perm]
            graph_mask = graph_mask[perm]

        sample.set_field(graph_boxes, "graph_boxes")
        sample.set_field(graph_labels, "graph_labels")
        sample.set_field(graph_mask, "graph_mask")
        sample.set_field(torch.tensor([height, width], dtype=torch.int64), "graph_image_size")
        sample.set_metainfo({"graph_num_nodes": int(num_nodes), "graph_unk_id": int(unk_id)})

    return {"inputs": inputs, "data_samples": data_samples}

