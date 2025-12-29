from __future__ import annotations

from typing import Any, Dict, List

import torch


def _sanitize_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    x1n = torch.minimum(x1, x2)
    x2n = torch.maximum(x1, x2)
    y1n = torch.minimum(y1, y2)
    y2n = torch.maximum(y1, y2)
    out = torch.stack([x1n, y1n, x2n, y2n], dim=-1)
    return out


def _random_noise_boxes(num: int, *, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if num <= 0:
        return torch.empty((0, 4), device=device, dtype=dtype)
    x1 = torch.rand((num,), device=device, dtype=dtype) * float(width)
    y1 = torch.rand((num,), device=device, dtype=dtype) * float(height)
    x2 = torch.rand((num,), device=device, dtype=dtype) * float(width)
    y2 = torch.rand((num,), device=device, dtype=dtype) * float(height)
    boxes = torch.stack([x1, y1, x2, y2], dim=-1)
    boxes = _sanitize_xyxy(boxes)
    # avoid degenerate boxes
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    x2 = torch.maximum(x2, x1 + 1e-3)
    y2 = torch.maximum(y2, y1 + 1e-3)
    return torch.stack([x1, y1, x2, y2], dim=-1)


def collate_coco_graph(
    batch: List[Dict[str, Any]],
    *,
    num_nodes: int,
    unk_id: int,
) -> Dict[str, torch.Tensor]:
    """
    Custom collate to build a fixed-N node "graph batch" for detection diffusion.

    Input item formats supported (per sample):
    - {"image": CHW Tensor, "boxes": (M,4) xyxy abs, "labels": (M,) long, "height": int, "width": int}
    - Detectron2-style: {"image": CHW Tensor, "instances": Instances, "height": int, "width": int}

    Returns a dict:
    - pixel_values: (B, 3, H, W) float/uint8 stacked (no padding)
    - boxes: (B, N, 4) float32 xyxy abs
    - labels: (B, N) int64 with padding=unk_id
    - mask: (B, N) bool, True=GT nodes, False=noise/pad nodes
    - image_sizes: (B, 2) int64 (H, W)
    """
    if num_nodes <= 0:
        raise ValueError(f"num_nodes must be > 0, got {num_nodes}")

    pixel_values = []
    boxes_list = []
    labels_list = []
    mask_list = []
    sizes = []

    for sample in batch:
        img = sample["image"]
        if not isinstance(img, torch.Tensor) or img.dim() != 3:
            raise ValueError("Each sample must provide an `image` CHW tensor.")
        height = int(sample.get("height", img.shape[-2]))
        width = int(sample.get("width", img.shape[-1]))
        sizes.append(torch.tensor([height, width], dtype=torch.int64))

        if "instances" in sample:
            inst = sample["instances"]
            gt_boxes = inst.gt_boxes.tensor
            gt_labels = inst.gt_classes
        else:
            gt_boxes = sample.get("boxes", None)
            gt_labels = sample.get("labels", None)
            if gt_boxes is None or gt_labels is None:
                raise ValueError("Each sample must provide (`boxes`,`labels`) or `instances`.")

        gt_boxes = gt_boxes.to(dtype=torch.float32)
        gt_boxes = _sanitize_xyxy(gt_boxes)
        gt_labels = gt_labels.to(dtype=torch.long)

        m = int(gt_boxes.shape[0])
        m_keep = min(m, int(num_nodes))
        gt_boxes = gt_boxes[:m_keep]
        gt_labels = gt_labels[:m_keep]

        pad = int(num_nodes) - m_keep
        if pad > 0:
            noise_boxes = _random_noise_boxes(
                pad, height=height, width=width, device=gt_boxes.device, dtype=gt_boxes.dtype
            )
            boxes = torch.cat([gt_boxes, noise_boxes], dim=0)
            labels = torch.cat(
                [gt_labels, torch.full((pad,), int(unk_id), device=gt_labels.device, dtype=torch.long)],
                dim=0,
            )
            mask = torch.cat(
                [torch.ones((m_keep,), device=gt_labels.device, dtype=torch.bool), torch.zeros((pad,), device=gt_labels.device, dtype=torch.bool)],
                dim=0,
            )
        else:
            boxes = gt_boxes
            labels = gt_labels
            mask = torch.ones((num_nodes,), device=gt_labels.device, dtype=torch.bool)

        pixel_values.append(img)
        boxes_list.append(boxes)
        labels_list.append(labels)
        mask_list.append(mask)

    return {
        "pixel_values": torch.stack(pixel_values, dim=0),
        "boxes": torch.stack(boxes_list, dim=0),
        "labels": torch.stack(labels_list, dim=0),
        "mask": torch.stack(mask_list, dim=0),
        "image_sizes": torch.stack(sizes, dim=0),
    }

