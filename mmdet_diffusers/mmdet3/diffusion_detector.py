from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.structures import InstanceData
from scipy.optimize import linear_sum_assignment
from torchvision.ops import nms

from mmdet.models.detectors.base import BaseDetector
from mmdet.registry import MODELS

from mmdet_diffusers.mmdet3.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from mmdet_diffusers.mmdet3.dynamic_head import GraphDenoisingNetwork
from mmdet_diffusers.schedulers import (
    D3PMConfig,
    D3PMLabelScheduler,
    build_ddim_scheduler_from_betas,
    cosine_beta_schedule,
)


def _sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Sigmoid focal loss (RetinaNet), reduction="none".
    inputs/targets: (..., K)
    """
    targets = targets.to(dtype=inputs.dtype)
    prob = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
    loss = ce_loss * ((1.0 - p_t) ** float(gamma))

    a = float(alpha)
    if a >= 0.0:
        alpha_t = a * targets + (1.0 - a) * (1.0 - targets)
        loss = alpha_t * loss
    return loss


def _sanitize_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    x1n = torch.minimum(x1, x2)
    x2n = torch.maximum(x1, x2)
    y1n = torch.minimum(y1, y2)
    y2n = torch.maximum(y1, y2)
    out = torch.stack([x1n, y1n, x2n, y2n], dim=-1)
    return out


def _clamp_xyxy_single(boxes: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    w = float(max(int(width), 1))
    h = float(max(int(height), 1))
    x1 = boxes[..., 0].clamp(min=0.0)
    y1 = boxes[..., 1].clamp(min=0.0)
    x2 = boxes[..., 2].clamp(min=0.0)
    y2 = boxes[..., 3].clamp(min=0.0)
    x1 = torch.minimum(x1, torch.as_tensor(w, device=boxes.device, dtype=boxes.dtype))
    x2 = torch.minimum(x2, torch.as_tensor(w, device=boxes.device, dtype=boxes.dtype))
    y1 = torch.minimum(y1, torch.as_tensor(h, device=boxes.device, dtype=boxes.dtype))
    y2 = torch.minimum(y2, torch.as_tensor(h, device=boxes.device, dtype=boxes.dtype))
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _clamp_xyxy_to_image(boxes: torch.Tensor, image_sizes: torch.Tensor) -> torch.Tensor:
    """
    boxes: (B, N, 4) abs xyxy
    image_sizes: (B, 2) int64 [H, W]
    """
    if boxes.dim() != 3 or boxes.shape[-1] != 4:
        raise ValueError(f"Expected boxes (B,N,4), got {tuple(boxes.shape)}")
    if image_sizes.dim() != 2 or image_sizes.shape[-1] != 2 or image_sizes.shape[0] != boxes.shape[0]:
        raise ValueError(f"Expected image_sizes (B,2), got {tuple(image_sizes.shape)} for boxes {tuple(boxes.shape)}")

    dtype = boxes.dtype
    h = image_sizes[:, 0].to(device=boxes.device, dtype=dtype).clamp(min=1).view(-1, 1)
    w = image_sizes[:, 1].to(device=boxes.device, dtype=dtype).clamp(min=1).view(-1, 1)

    x1 = boxes[..., 0].clamp(min=0.0)
    y1 = boxes[..., 1].clamp(min=0.0)
    x2 = boxes[..., 2].clamp(min=0.0)
    y2 = boxes[..., 3].clamp(min=0.0)
    x1 = torch.minimum(x1, w)
    x2 = torch.minimum(x2, w)
    y1 = torch.minimum(y1, h)
    y2 = torch.minimum(y2, h)
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _clip_xyxy_to_image_min_size(
    boxes: torch.Tensor,
    image_sizes: torch.Tensor,
    *,
    min_size: float = 1.0,
) -> torch.Tensor:
    """
    boxes: (B,N,4) abs xyxy
    image_sizes: (B,2) int64 [H,W]
    """
    boxes = _sanitize_xyxy(boxes)
    boxes = _clamp_xyxy_to_image(boxes, image_sizes)

    dtype = boxes.dtype
    h = image_sizes[:, 0].to(device=boxes.device, dtype=dtype).clamp(min=1).view(-1, 1)
    w = image_sizes[:, 1].to(device=boxes.device, dtype=dtype).clamp(min=1).view(-1, 1)

    eps = torch.as_tensor(float(min_size), device=boxes.device, dtype=dtype).clamp(min=0.0)
    min_w = torch.minimum(w, eps)
    min_h = torch.minimum(h, eps)

    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    x1 = torch.minimum(x1, w - min_w)
    y1 = torch.minimum(y1, h - min_h)
    x1 = x1.clamp(min=0.0)
    y1 = y1.clamp(min=0.0)
    x2 = torch.maximum(x2, x1 + min_w)
    y2 = torch.maximum(y2, y1 + min_h)
    x2 = torch.minimum(x2, w)
    y2 = torch.minimum(y2, h)
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _clip_xyxy_to_image_min_size_ste(
    boxes: torch.Tensor,
    image_sizes: torch.Tensor,
    *,
    min_size: float = 1.0,
) -> torch.Tensor:
    boxes_s = _sanitize_xyxy(boxes)
    clipped = _clip_xyxy_to_image_min_size(boxes_s, image_sizes, min_size=float(min_size))
    return boxes_s + (clipped - boxes_s).detach()


def _boxes_abs_to_diffusion(
    boxes_xyxy_abs: torch.Tensor,
    image_sizes: torch.Tensor,
    *,
    box_scale: float,
) -> torch.Tensor:
    # boxes_xyxy_abs: (B,N,4) abs xyxy -> diffusion space: scaled cxcywh in [-scale, scale]
    h = image_sizes[:, 0].to(dtype=torch.float32)
    w = image_sizes[:, 1].to(dtype=torch.float32)
    whwh = torch.stack([w, h, w, h], dim=-1).clamp(min=1.0)  # (B,4)

    boxes = boxes_xyxy_abs.to(dtype=torch.float32) / whwh[:, None, :]
    boxes = boxes.clamp(0.0, 1.0)
    cxcywh = box_xyxy_to_cxcywh(boxes)
    x = (cxcywh * 2.0 - 1.0) * float(box_scale)
    x = x.clamp(min=-float(box_scale), max=float(box_scale))
    return x


def _boxes_diffusion_to_abs(
    boxes_diff: torch.Tensor,
    image_sizes: torch.Tensor,
    *,
    box_scale: float,
) -> torch.Tensor:
    h = image_sizes[:, 0].to(dtype=torch.float32)
    w = image_sizes[:, 1].to(dtype=torch.float32)
    whwh = torch.stack([w, h, w, h], dim=-1).clamp(min=1.0)  # (B,4)

    x = boxes_diff.to(dtype=torch.float32).clamp(min=-float(box_scale), max=float(box_scale))
    x = ((x / float(box_scale)) + 1.0) * 0.5
    x = x.clamp(0.0, 1.0)
    boxes = box_cxcywh_to_xyxy(x)
    boxes = boxes * whwh[:, None, :]
    boxes = _sanitize_xyxy(boxes)
    # avoid degenerate boxes
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    x2 = torch.maximum(x2, x1 + 1e-3)
    y2 = torch.maximum(y2, y1 + 1e-3)
    boxes = torch.stack([x1, y1, x2, y2], dim=-1)
    return boxes


def _generalized_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    # boxes: (...,4) xyxy
    boxes1 = _sanitize_xyxy(boxes1)
    boxes2 = _sanitize_xyxy(boxes2)

    lt = torch.maximum(boxes1[..., :2], boxes2[..., :2])
    rb = torch.minimum(boxes1[..., 2:], boxes2[..., 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    area1 = (boxes1[..., 2] - boxes1[..., 0]).clamp(min=0) * (boxes1[..., 3] - boxes1[..., 1]).clamp(min=0)
    area2 = (boxes2[..., 2] - boxes2[..., 0]).clamp(min=0) * (boxes2[..., 3] - boxes2[..., 1]).clamp(min=0)
    union = area1 + area2 - inter
    iou = inter / union.clamp(min=1e-6)

    lt_c = torch.minimum(boxes1[..., :2], boxes2[..., :2])
    rb_c = torch.maximum(boxes1[..., 2:], boxes2[..., 2:])
    wh_c = (rb_c - lt_c).clamp(min=0)
    area_c = wh_c[..., 0] * wh_c[..., 1]
    giou = iou - (area_c - union) / area_c.clamp(min=1e-6)
    return giou


def _pairwise_iou_diag(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU for corresponding pairs (diag of box_iou) without allocating NxN.

    boxes1/boxes2: (N,4) xyxy
    returns: (N,) IoU in [0,1]
    """
    if boxes1.shape != boxes2.shape:
        raise ValueError(f"Expected boxes1/boxes2 same shape, got {tuple(boxes1.shape)} vs {tuple(boxes2.shape)}")
    boxes1 = _sanitize_xyxy(boxes1)
    boxes2 = _sanitize_xyxy(boxes2)

    lt = torch.maximum(boxes1[:, :2], boxes2[:, :2])
    rb = torch.minimum(boxes1[:, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    union = area1 + area2 - inter
    return inter / union.clamp(min=1e-6)


def _pairwise_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    boxes1: (N,4) xyxy
    boxes2: (M,4) xyxy
    returns: (N,M) IoU in [0,1]
    """
    boxes1 = _sanitize_xyxy(boxes1)
    boxes2 = _sanitize_xyxy(boxes2)

    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-6)


def _timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10_000) -> torch.Tensor:
    # timesteps: (B,) long
    half = dim // 2
    freqs = torch.exp(-torch.log(torch.tensor(float(max_period), device=timesteps.device)) * torch.arange(half, device=timesteps.device) / half)
    args = timesteps.to(dtype=torch.float32).unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros((emb.shape[0], 1), device=emb.device, dtype=emb.dtype)], dim=-1)
    return emb


@dataclass(frozen=True)
class _MatchResult:
    pred_indices: torch.Tensor  # (P,)
    gt_indices: torch.Tensor  # (P,)


def _get_in_boxes_info(
    boxes_cxcywh: torch.Tensor,  # (N,4) abs cxcywh
    target_gts_cxcywh: torch.Tensor,  # (M,4) abs cxcywh
    *,
    center_radius: float = 2.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    SimOTA-style candidate filtering used by DiffusionDet DynamicK matching.
    Returns:
      is_in_boxes_anchor: (N,) bool
      is_in_boxes_and_center: (N,M) bool
    """
    if boxes_cxcywh.numel() == 0 or target_gts_cxcywh.numel() == 0:
        n = int(boxes_cxcywh.shape[0])
        m = int(target_gts_cxcywh.shape[0])
        return torch.zeros((n,), device=boxes_cxcywh.device, dtype=torch.bool), torch.zeros(
            (n, m), device=boxes_cxcywh.device, dtype=torch.bool
        )

    xy_target_gts = box_cxcywh_to_xyxy(target_gts_cxcywh)  # (M,4) xyxy

    anchor_center_x = boxes_cxcywh[:, 0].unsqueeze(1)
    anchor_center_y = boxes_cxcywh[:, 1].unsqueeze(1)

    b_l = anchor_center_x > xy_target_gts[:, 0].unsqueeze(0)
    b_r = anchor_center_x < xy_target_gts[:, 2].unsqueeze(0)
    b_t = anchor_center_y > xy_target_gts[:, 1].unsqueeze(0)
    b_b = anchor_center_y < xy_target_gts[:, 3].unsqueeze(0)
    is_in_boxes = ((b_l.long() + b_r.long() + b_t.long() + b_b.long()) == 4)
    is_in_boxes_all = is_in_boxes.sum(1) > 0

    cx = target_gts_cxcywh[:, 0]
    cy = target_gts_cxcywh[:, 1]
    w = (xy_target_gts[:, 2] - xy_target_gts[:, 0]).clamp(min=1e-6)
    h = (xy_target_gts[:, 3] - xy_target_gts[:, 1]).clamp(min=1e-6)

    r = float(center_radius)
    c_l = anchor_center_x > (cx - r * w).unsqueeze(0)
    c_r = anchor_center_x < (cx + r * w).unsqueeze(0)
    c_t = anchor_center_y > (cy - r * h).unsqueeze(0)
    c_b = anchor_center_y < (cy + r * h).unsqueeze(0)

    is_in_centers = ((c_l.long() + c_r.long() + c_t.long() + c_b.long()) == 4)
    is_in_centers_all = is_in_centers.sum(1) > 0

    is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all
    is_in_boxes_and_center = is_in_boxes & is_in_centers
    return is_in_boxes_anchor, is_in_boxes_and_center


def _dynamic_k_matching(
    cost: torch.Tensor,  # (N,M) lower is better
    pair_wise_ious: torch.Tensor,  # (N,M)
    *,
    ota_k: int,
) -> _MatchResult:
    n, m = int(cost.shape[0]), int(cost.shape[1])
    if n == 0 or m == 0:
        return _MatchResult(pred_indices=torch.empty((0,), dtype=torch.long, device=cost.device), gt_indices=torch.empty((0,), dtype=torch.long, device=cost.device))

    cost = cost.clone()
    matching_matrix = torch.zeros_like(cost)

    k = max(int(ota_k), 1)
    k = min(k, n)
    topk_ious, _ = torch.topk(pair_wise_ious, k=k, dim=0)
    dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)

    for gt_idx in range(m):
        kk = int(dynamic_ks[gt_idx].item())
        kk = min(kk, n)
        _, pos_idx = torch.topk(cost[:, gt_idx], k=kk, largest=False)
        matching_matrix[pos_idx, gt_idx] = 1.0

    anchor_matching_gt = matching_matrix.sum(1)
    if (anchor_matching_gt > 1).any():
        _, cost_argmin = torch.min(cost[anchor_matching_gt > 1], dim=1)
        matching_matrix[anchor_matching_gt > 1] *= 0
        matching_matrix[anchor_matching_gt > 1, cost_argmin] = 1.0

    # Ensure every GT has at least one match. In some pathological cases
    # (e.g., NaNs in the cost matrix), this loop can otherwise spin forever.
    max_refine_steps = 50
    refine_step = 0
    while (matching_matrix.sum(0) == 0).any():
        refine_step += 1
        if refine_step > max_refine_steps:
            break
        matched_query = matching_matrix.sum(1) > 0
        cost[matched_query] = cost[matched_query] + 100000.0

        unmatch_id = torch.nonzero(matching_matrix.sum(0) == 0, as_tuple=False).squeeze(1)
        for gt_idx in unmatch_id.tolist():
            pos_idx = torch.argmin(cost[:, gt_idx])
            matching_matrix[pos_idx, gt_idx] = 1.0

        anchor_matching_gt = matching_matrix.sum(1)
        if (anchor_matching_gt > 1).any():
            _, cost_argmin = torch.min(cost[anchor_matching_gt > 1], dim=1)
            matching_matrix[anchor_matching_gt > 1] *= 0
            matching_matrix[anchor_matching_gt > 1, cost_argmin] = 1.0

    selected_query = matching_matrix.sum(1) > 0
    pred_indices = selected_query.nonzero(as_tuple=False).squeeze(1)
    gt_indices = matching_matrix[selected_query].max(1)[1]
    return _MatchResult(pred_indices=pred_indices, gt_indices=gt_indices)


def _dynamic_k_match(
    *,
    pred_boxes: torch.Tensor,  # (N,4) abs xyxy
    pred_logits: torch.Tensor,  # (N,K+1)
    gt_boxes: torch.Tensor,  # (M,4) abs xyxy
    gt_labels: torch.Tensor,  # (M,)
    image_size: Tuple[int, int],  # (H,W)
    num_classes: int,
    ota_k: int,
    cost_class: float = 1.0,
    cost_bbox: float = 5.0,
    cost_giou: float = 2.0,
    use_sigmoid_cls: bool = False,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
) -> _MatchResult:
    n = int(pred_boxes.shape[0])
    m = int(gt_boxes.shape[0])
    if m == 0 or n == 0:
        return _MatchResult(
            pred_indices=torch.empty((0,), dtype=torch.long, device=pred_boxes.device),
            gt_indices=torch.empty((0,), dtype=torch.long, device=pred_boxes.device),
        )

    height, width = int(image_size[0]), int(image_size[1])
    whwh = torch.tensor([width, height, width, height], device=pred_boxes.device, dtype=torch.float32).clamp(min=1.0)

    if use_sigmoid_cls:
        out_prob = pred_logits[:, : int(num_classes)].sigmoid()
        alpha = float(focal_alpha)
        alpha = min(max(alpha, 0.0), 1.0)
        gamma = max(float(focal_gamma), 0.0)
        neg_cost = (1.0 - alpha) * (out_prob**gamma) * (-(1.0 - out_prob + 1e-8).log())
        pos_cost = alpha * ((1.0 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
    else:
        out_prob = pred_logits.softmax(dim=-1)[..., : int(num_classes)]
        cls_cost = -out_prob[:, gt_labels]

    pred_boxes_n = pred_boxes / whwh
    gt_boxes_n = gt_boxes / whwh
    bbox_cost = torch.cdist(pred_boxes_n, gt_boxes_n, p=1)
    giou_cost = -_generalized_iou(pred_boxes[:, None, :], gt_boxes[None, :, :])

    pred_cxcywh = box_xyxy_to_cxcywh(pred_boxes)
    gt_cxcywh = box_xyxy_to_cxcywh(gt_boxes)
    fg_mask, is_in_boxes_and_center = _get_in_boxes_info(pred_cxcywh, gt_cxcywh)

    pair_wise_ious = _pairwise_iou(pred_boxes, gt_boxes)
    cost = cost_bbox * bbox_cost + cost_class * cls_cost + cost_giou * giou_cost + 100.0 * (~is_in_boxes_and_center)
    cost[~fg_mask] = cost[~fg_mask] + 10000.0

    return _dynamic_k_matching(cost, pair_wise_ious, ota_k=int(ota_k))


def _hungarian_match(
    *,
    pred_boxes: torch.Tensor,  # (N,4) abs
    pred_logits: torch.Tensor,  # (N,K+1)
    gt_boxes: torch.Tensor,  # (M,4) abs
    gt_labels: torch.Tensor,  # (M,)
    image_size: Tuple[int, int],  # (H,W)
    num_classes: int,
    cost_class: float = 1.0,
    cost_bbox: float = 5.0,
    cost_giou: float = 2.0,
    use_sigmoid_cls: bool = False,
    focal_cost: bool = False,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
) -> _MatchResult:
    n = int(pred_boxes.shape[0])
    m = int(gt_boxes.shape[0])
    if m == 0 or n == 0:
        return _MatchResult(pred_indices=torch.empty((0,), dtype=torch.long), gt_indices=torch.empty((0,), dtype=torch.long))

    height, width = int(image_size[0]), int(image_size[1])
    whwh = torch.tensor([width, height, width, height], device=pred_boxes.device, dtype=torch.float32).clamp(min=1.0)

    pred_boxes_n = pred_boxes / whwh
    gt_boxes_n = gt_boxes / whwh

    if use_sigmoid_cls:
        prob = pred_logits[:, : int(num_classes)].sigmoid()
        if focal_cost:
            alpha = float(focal_alpha)
            alpha = min(max(alpha, 0.0), 1.0)
            gamma = max(float(focal_gamma), 0.0)
            neg_cost = (1.0 - alpha) * (prob**gamma) * (-(1.0 - prob + 1e-8).log())
            pos_cost = alpha * ((1.0 - prob) ** gamma) * (-(prob + 1e-8).log())
            cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
        else:
            cls_cost = -torch.log(prob[:, gt_labels].clamp(min=1e-8))
    else:
        prob = pred_logits.softmax(dim=-1)
        cls_cost = -torch.log(prob[:, gt_labels].clamp(min=1e-8))

    bbox_cost = torch.cdist(pred_boxes_n, gt_boxes_n, p=1)
    giou = _generalized_iou(pred_boxes_n[:, None, :], gt_boxes_n[None, :, :])
    giou_cost = -giou

    total_cost = cost_class * cls_cost + cost_bbox * bbox_cost + cost_giou * giou_cost
    row, col = linear_sum_assignment(total_cost.detach().cpu().numpy())

    return _MatchResult(
        pred_indices=torch.as_tensor(row, dtype=torch.long, device=pred_boxes.device),
        gt_indices=torch.as_tensor(col, dtype=torch.long, device=pred_boxes.device),
    )


@MODELS.register_module()
class GraphDiffusionDetector(BaseDetector):
    """
    Diffusion-based detector for MMDetection 3.x that consumes a fixed-N "graph"
    (built by `coco_graph_collate`) and trains with DETR-style Hungarian matching.

    `check.md` alignment notes:
    - Fixed-N graph batch (gt + noise + unk padding): `coco_graph_collate`
    - Graph Transformer with geometric bias: `GraphDenoisingNetwork`
    - Label diffusion state (D3PM-like): `D3PMLabelScheduler`
    - Optional quality head + Langevin-style energy guidance: `use_quality_head` + `quality_guidance_*`
    - Optional graph topology consistency loss: `graph_topo_loss_weight`
    """

    def __init__(
        self,
        *,
        backbone: dict,
        neck: Optional[dict] = None,
        num_classes: int,
        num_proposals: int = 500,
        diffusion_timesteps: int = 1000,
        sampling_timesteps: int = 1,
        box_scale: float = 2.0,
        # optional torch.compile for the denoising head
        torch_compile: bool = False,
        torch_compile_mode: str = "default",
        torch_compile_backend: str = "inductor",
        torch_compile_dynamic: bool = False,
        # diffusion head / dynamic head
        hidden_dim: int = 256,
        dim_feedforward: int = 2048,
        nhead: int = 8,
        dropout: float = 0.0,
        activation: str = "relu",
        num_heads: int = 6,
        deep_supervision: bool = True,
        pooler_resolution: int = 7,
        roi_featmap_strides: Tuple[int, ...] = (4, 8, 16, 32),
        dim_dynamic: int = 64,
        num_dynamic: int = 2,
        num_cls_layers: int = 1,
        num_reg_layers: int = 1,
        # graph transformer / geometric bias
        use_geo_bias: bool = True,
        geo_bias_type: str = "mlp",
        geo_bias_scale: float = 1.0,
        capture_graph_attn: bool = False,
        # label diffusion state injection
        use_label_state: bool = True,
        label_state_scale: float = 0.1,
        # quality head + guidance (energy sampling)
        use_quality_head: bool = False,
        quality_head_hidden_dim: int = 256,
        quality_head_include_t: bool = True,
        quality_head_use_log_wh: bool = True,
        quality_box_norm: float = 1000.0,
        quality_loss_type: str = "bce",
        quality_loss_weight: float = 0.0,
        quality_guidance_scale: float = 0.0,
        quality_guidance_topk: int = 50,
        quality_guidance_score_weight: bool = True,
        quality_guidance_grad_norm: str = "proposal",
        quality_guidance_mode: str = "final",
        quality_guidance_t_threshold: int = 0,
        quality_guidance_langevin_steps: int = 1,
        quality_guidance_langevin_noise: float = 0.0,
        quality_guidance_step_schedule: str = "constant",
        quality_guidance_time_power: float = 1.0,
        # anisotropic noise (per-coordinate)
        aniso_noise: bool = False,
        aniso_noise_sigma_xy: float = 1.0,
        aniso_noise_sigma_w: float = 1.0,
        aniso_noise_sigma_h: float = 1.0,
        ddim_sampling_eta: float = 1.0,
        cls_loss_type: str = "ce",
        qfl_beta: float = 2.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        prior_prob: float = 0.01,
        init_head_xavier: bool = False,
        loss_cls_weight: float = 1.0,
        loss_bbox_weight: float = 1.0,
        loss_giou_weight: float = 1.0,
        no_object_weight: float = 1.0,
        use_dynamic_k_matching: bool = False,
        ota_k: int = 5,
        box_loss_iou_weight_power: float = 0.0,
        graph_topo_loss_weight: float = 0.0,
        graph_topo_iou_thresh: float = 0.1,
        graph_topo_target_norm: str = "row",
        score_thr: float = 0.05,
        nms_iou_thr: float = 0.6,
        max_per_img: int = 100,
        # sampler distillation (teacher multi-step -> student one-step)
        sampler_distill: bool = False,
        sampler_distill_teacher_weights: str = "",
        sampler_distill_teacher_sample_step: int = 20,
        sampler_distill_teacher_eta: float = 1.0,
        sampler_distill_topk: int = 0,
        sampler_distill_box_weight: float = 0.0,
        sampler_distill_cls_weight: float = 0.0,
        sampler_distill_cls_temperature: float = 1.0,
        data_preprocessor: Optional[dict] = None,
        init_cfg=None,
    ):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.backbone = MODELS.build(backbone)
        self.neck = MODELS.build(neck) if neck is not None else None

        self.num_classes = int(num_classes)
        self.num_proposals = int(num_proposals)
        self.diffusion_timesteps = int(diffusion_timesteps)
        self.sampling_timesteps = int(sampling_timesteps)
        self.box_scale = float(box_scale)
        self.torch_compile = bool(torch_compile)
        self.torch_compile_mode = str(torch_compile_mode)
        self.torch_compile_backend = str(torch_compile_backend)
        self.torch_compile_dynamic = bool(torch_compile_dynamic)
        self.aniso_noise = bool(aniso_noise)
        sigma_xy = max(float(aniso_noise_sigma_xy), 1e-6)
        sigma_w = max(float(aniso_noise_sigma_w), 1e-6)
        sigma_h = max(float(aniso_noise_sigma_h), 1e-6)
        self.register_buffer(
            "aniso_noise_sigma",
            torch.tensor([sigma_xy, sigma_xy, sigma_w, sigma_h], dtype=torch.float32),
            persistent=False,
        )
        self.ddim_sampling_eta = float(ddim_sampling_eta)

        self.cls_loss_type = str(cls_loss_type).lower()
        if self.cls_loss_type not in {"ce", "focal", "qfl"}:
            raise ValueError(f"Unsupported cls_loss_type={cls_loss_type!r}; expected one of: 'ce', 'focal', 'qfl'")
        self.qfl_beta = max(float(qfl_beta), 0.0)
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)
        self.prior_prob = float(prior_prob)
        self.init_head_xavier = bool(init_head_xavier)
        self.loss_cls_weight = float(loss_cls_weight)
        self.loss_bbox_weight = float(loss_bbox_weight)
        self.loss_giou_weight = float(loss_giou_weight)
        self.no_object_weight = float(no_object_weight)
        self.use_dynamic_k_matching = bool(use_dynamic_k_matching)
        self.ota_k = max(int(ota_k), 1)
        self.box_loss_iou_weight_power = max(float(box_loss_iou_weight_power), 0.0)

        self.graph_topo_loss_weight = max(float(graph_topo_loss_weight), 0.0)
        self.graph_topo_iou_thresh = float(graph_topo_iou_thresh)
        self.graph_topo_target_norm = str(graph_topo_target_norm).lower()
        if self.graph_topo_target_norm not in {"row", "none"}:
            raise ValueError(
                f"Unsupported graph_topo_target_norm={graph_topo_target_norm!r}; expected one of: 'row', 'none'"
            )
        if not (0.0 <= self.graph_topo_iou_thresh <= 1.0):
            raise ValueError(f"graph_topo_iou_thresh must be in [0,1], got {graph_topo_iou_thresh}")

        self.score_thr = float(score_thr)
        self.nms_iou_thr = float(nms_iou_thr)
        self.max_per_img = int(max_per_img)

        self.sampler_distill = bool(sampler_distill)
        self.sampler_distill_teacher_weights = str(sampler_distill_teacher_weights).strip()
        self.sampler_distill_teacher_sample_step = max(int(sampler_distill_teacher_sample_step), 1)
        self.sampler_distill_teacher_eta = float(sampler_distill_teacher_eta)
        self.sampler_distill_topk = max(int(sampler_distill_topk), 0)
        self.sampler_distill_box_weight = float(sampler_distill_box_weight)
        self.sampler_distill_cls_weight = float(sampler_distill_cls_weight)
        self.sampler_distill_cls_temperature = float(sampler_distill_cls_temperature)

        self.use_label_state = bool(use_label_state)
        self.use_quality_head = bool(use_quality_head)
        self.quality_loss_type = str(quality_loss_type).lower()
        if self.quality_loss_type not in {"bce", "l1", "mse"}:
            raise ValueError(f"Unsupported quality_loss_type={self.quality_loss_type!r}; expected one of: 'bce','l1','mse'")
        self.quality_loss_weight = float(quality_loss_weight)

        self.quality_guidance_scale = float(quality_guidance_scale)
        self.quality_guidance_topk = int(quality_guidance_topk)
        self.quality_guidance_score_weight = bool(quality_guidance_score_weight)
        self.quality_guidance_grad_norm = str(quality_guidance_grad_norm).lower()
        if self.quality_guidance_grad_norm not in {"proposal", "global"}:
            raise ValueError(
                f"Unsupported quality_guidance_grad_norm={self.quality_guidance_grad_norm!r}; expected one of: 'proposal','global'"
            )
        self.quality_guidance_mode = str(quality_guidance_mode).lower()
        if self.quality_guidance_mode not in {"final", "all", "threshold"}:
            raise ValueError(
                f"Unsupported quality_guidance_mode={self.quality_guidance_mode!r}; expected one of: 'final','all','threshold'"
            )
        self.quality_guidance_t_threshold = int(quality_guidance_t_threshold)
        self.quality_guidance_langevin_steps = max(int(quality_guidance_langevin_steps), 0)
        self.quality_guidance_langevin_noise = float(quality_guidance_langevin_noise)
        self.quality_guidance_step_schedule = str(quality_guidance_step_schedule).lower()
        if self.quality_guidance_step_schedule not in {"constant", "linear"}:
            raise ValueError(
                f"Unsupported quality_guidance_step_schedule={self.quality_guidance_step_schedule!r}; expected one of: 'constant','linear'"
            )
        self.quality_guidance_time_power = float(quality_guidance_time_power)

        betas = cosine_beta_schedule(self.diffusion_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod), persistent=False)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod), persistent=False)

        try:
            self.box_scheduler = build_ddim_scheduler_from_betas(self.betas)
        except Exception:
            self.box_scheduler = None

        self.label_scheduler = D3PMLabelScheduler(
            betas=self.betas,
            config=D3PMConfig(num_classes=self.num_classes, kernel="mask", keep_prob_schedule="sqrt_alphas_cumprod"),
        )
        self.roi_featmap_strides = tuple(int(s) for s in roi_featmap_strides)

        roi_extractor = dict(
            type="SingleRoIExtractor",
            roi_layer=dict(type="RoIAlign", output_size=int(pooler_resolution), sampling_ratio=0),
            out_channels=int(hidden_dim),
            featmap_strides=list(self.roi_featmap_strides),
        )

        self.head = GraphDenoisingNetwork(
            num_proposals=self.num_proposals,
            num_classes=self.num_classes,
            hidden_dim=int(hidden_dim),
            dim_feedforward=int(dim_feedforward),
            nhead=int(nhead),
            dropout=float(dropout),
            activation=str(activation),
            num_heads=int(num_heads),
            deep_supervision=bool(deep_supervision),
            roi_extractor=roi_extractor,
            pooler_resolution=int(pooler_resolution),
            diffusion_timesteps=int(self.diffusion_timesteps),
            dim_dynamic=int(dim_dynamic),
            num_dynamic=int(num_dynamic),
            num_cls_layers=int(num_cls_layers),
            num_reg_layers=int(num_reg_layers),
            use_geo_bias=bool(use_geo_bias),
            geo_bias_type=str(geo_bias_type),
            geo_bias_scale=float(geo_bias_scale),
            capture_graph_attn=bool(capture_graph_attn) or (self.graph_topo_loss_weight > 0.0),
            use_label_state=bool(use_label_state),
            label_state_scale=float(label_state_scale),
            use_quality_head=bool(self.use_quality_head),
            quality_head_hidden_dim=int(quality_head_hidden_dim),
            quality_head_include_t=bool(quality_head_include_t),
            quality_head_use_log_wh=bool(quality_head_use_log_wh),
            quality_box_norm=float(quality_box_norm),
        )

        if self.init_head_xavier and hasattr(self.head, "reset_parameters"):
            try:
                self.head.reset_parameters(
                    use_sigmoid_cls=self.cls_loss_type in {"focal", "qfl"},
                    prior_prob=float(self.prior_prob),
                )
            except Exception:
                pass

        can_compile = (self.graph_topo_loss_weight <= 0.0) and (self.quality_guidance_scale <= 0.0)
        if self.torch_compile and can_compile and hasattr(torch, "compile"):
            try:
                self.head = torch.compile(  # type: ignore[assignment]
                    self.head,
                    mode=self.torch_compile_mode,
                    backend=self.torch_compile_backend,
                    dynamic=self.torch_compile_dynamic,
                )
            except Exception:
                pass

        if self.sampler_distill and (self.sampler_distill_box_weight > 0.0 or self.sampler_distill_cls_weight > 0.0):
            if not self.sampler_distill_teacher_weights:
                raise ValueError("sampler_distill enabled but sampler_distill_teacher_weights is empty.")
            teacher = self._build_teacher(self.sampler_distill_teacher_weights)
            self.set_teacher(teacher)

    def set_teacher(self, teacher: nn.Module) -> None:
        # Avoid registering the teacher as a submodule (checkpoint bloat).
        self.__dict__["teacher"] = teacher

    def _get_teacher(self, device: torch.device) -> nn.Module | None:
        teacher = self.__dict__.get("teacher", None)
        if teacher is None or not isinstance(teacher, nn.Module):
            return None
        try:
            p0 = next(teacher.parameters())
        except StopIteration:
            p0 = None
        if p0 is not None and p0.device != device:
            teacher.to(device=device)
        teacher.eval()
        return teacher

    def _build_teacher(self, weights: str) -> nn.Module:
        teacher = copy.deepcopy(self)
        teacher.__dict__.pop("teacher", None)
        teacher.sampler_distill = False
        teacher.sampler_distill_teacher_weights = ""
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        try:
            from mmengine.runner.checkpoint import load_checkpoint

            load_checkpoint(teacher, weights, map_location="cpu", strict=False)
        except Exception:
            ckpt = torch.load(weights, map_location="cpu")
            state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            if not isinstance(state, dict):
                raise TypeError(f"Unsupported checkpoint format: {type(ckpt)}")
            teacher.load_state_dict(state, strict=False)

        return teacher

    @property
    def with_neck(self) -> bool:  # keep mypy happy
        return getattr(self, "neck", None) is not None

    def extract_feat(self, batch_inputs: torch.Tensor):
        feats = self.backbone(batch_inputs)
        if self.with_neck:
            feats = self.neck(feats)
        if isinstance(feats, dict):
            feats_list = [feats[k] for k in sorted(feats.keys())]
        elif isinstance(feats, (list, tuple)):
            feats_list = list(feats)
        else:
            feats_list = [feats]

        # Match roi_extractor.featmap_strides length.
        num_levels = len(self.roi_featmap_strides)
        return feats_list[:num_levels]

    def loss(self, batch_inputs: torch.Tensor, batch_data_samples: List) -> dict:
        device = batch_inputs.device
        feats = self.extract_feat(batch_inputs)

        b = int(batch_inputs.shape[0])
        image_sizes = torch.stack(
            [torch.as_tensor(ds.graph_image_size, device=device, dtype=torch.int64) for ds in batch_data_samples], dim=0
        )
        graph_boxes = torch.stack([ds.graph_boxes.to(device=device) for ds in batch_data_samples], dim=0)
        graph_labels = torch.stack([ds.graph_labels.to(device=device) for ds in batch_data_samples], dim=0)
        graph_mask = torch.stack([ds.graph_mask.to(device=device) for ds in batch_data_samples], dim=0)
        if int(graph_boxes.shape[1]) != int(self.num_proposals):
            raise ValueError(
                f"graph_boxes has num_nodes={int(graph_boxes.shape[1])}, but model.num_proposals={int(self.num_proposals)}"
            )

        x0 = _boxes_abs_to_diffusion(graph_boxes, image_sizes, box_scale=self.box_scale)

        timesteps = torch.randint(0, self.diffusion_timesteps, (b,), device=device, dtype=torch.long)
        noise = torch.randn_like(x0)
        if self.aniso_noise:
            sigma_vec = self.aniso_noise_sigma.to(device=noise.device, dtype=noise.dtype).view(1, 1, 4)
            noise = noise * sigma_vec
        if self.box_scheduler is not None:
            x_t = self.box_scheduler.add_noise(x0, noise, timesteps)
        else:
            sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].view(b, 1, 1)
            sqrt_om = self.sqrt_one_minus_alphas_cumprod[timesteps].view(b, 1, 1)
            x_t = sqrt_alpha * x0 + sqrt_om * noise

        x_t = x_t.clamp(min=-float(self.box_scale), max=float(self.box_scale))

        # D3PM label corruption as categorical distributions (B,N,K+1).
        label_state = self.label_scheduler.q_probs(graph_labels, timesteps)

        boxes_t = _boxes_diffusion_to_abs(x_t, image_sizes, box_scale=self.box_scale)
        boxes_t = _clip_xyxy_to_image_min_size(boxes_t, image_sizes, min_size=1.0)

        outputs_class, outputs_coord, outputs_quality = self.head(
            feats=feats,
            init_bboxes=boxes_t,
            timesteps=timesteps,
            label_state=label_state,
            image_sizes=image_sizes,
        )

        num_layers = int(outputs_class.shape[0])
        loss_cls_total = boxes_t.new_zeros(())
        loss_bbox_total = boxes_t.new_zeros(())
        loss_giou_total = boxes_t.new_zeros(())
        loss_quality_total = boxes_t.new_zeros(())

        for layer_idx in range(num_layers):
            pred_logits = outputs_class[layer_idx]  # (B,N,K+1)
            pred_boxes_abs = _clip_xyxy_to_image_min_size_ste(outputs_coord[layer_idx], image_sizes, min_size=1.0)

            # classification targets: background (unk) by default
            unk_id = int(self.num_classes)
            target_labels = torch.full((b, self.num_proposals), unk_id, device=device, dtype=torch.long)
            target_boxes = torch.zeros((b, self.num_proposals, 4), device=device, dtype=torch.float32)
            pos_mask = torch.zeros((b, self.num_proposals), device=device, dtype=torch.bool)

            for i in range(b):
                gt_mask = graph_mask[i]
                gt_boxes = graph_boxes[i][gt_mask]
                gt_labels = graph_labels[i][gt_mask]
                if gt_boxes.numel() == 0:
                    continue
                if self.use_dynamic_k_matching:
                    match = _dynamic_k_match(
                        pred_boxes=pred_boxes_abs[i],
                        pred_logits=pred_logits[i],
                        gt_boxes=gt_boxes,
                        gt_labels=gt_labels,
                        image_size=(int(image_sizes[i, 0].item()), int(image_sizes[i, 1].item())),
                        num_classes=self.num_classes,
                        ota_k=int(self.ota_k),
                        use_sigmoid_cls=self.cls_loss_type in {"focal", "qfl"},
                        focal_alpha=float(self.focal_alpha),
                        focal_gamma=float(self.focal_gamma),
                    )
                else:
                    match = _hungarian_match(
                        pred_boxes=pred_boxes_abs[i],
                        pred_logits=pred_logits[i],
                        gt_boxes=gt_boxes,
                        gt_labels=gt_labels,
                        image_size=(int(image_sizes[i, 0].item()), int(image_sizes[i, 1].item())),
                        num_classes=self.num_classes,
                        use_sigmoid_cls=self.cls_loss_type in {"focal", "qfl"},
                        focal_cost=self.cls_loss_type == "focal",
                        focal_alpha=float(self.focal_alpha),
                        focal_gamma=float(self.focal_gamma),
                    )
                if match.pred_indices.numel() == 0:
                    continue
                target_labels[i, match.pred_indices] = gt_labels[match.gt_indices]
                target_boxes[i, match.pred_indices] = gt_boxes[match.gt_indices]
                pos_mask[i, match.pred_indices] = True

            num_pos = int(pos_mask.sum().item())
            denom = pred_boxes_abs.new_tensor(float(max(num_pos, 1)))

            if self.cls_loss_type == "ce":
                weight = None
                if float(self.no_object_weight) != 1.0:
                    w = torch.ones((self.num_classes + 1,), device=device, dtype=torch.float32)
                    w[-1] = float(self.no_object_weight)
                    weight = w
                loss_cls = F.cross_entropy(
                    pred_logits.view(-1, self.num_classes + 1),
                    target_labels.view(-1),
                    weight=weight,
                    reduction="mean",
                )
            elif self.cls_loss_type == "focal":
                src_logits = pred_logits[..., : self.num_classes]
                target_onehot = torch.zeros(
                    (b, self.num_proposals, self.num_classes),
                    device=device,
                    dtype=src_logits.dtype,
                )
                pos_flat = pos_mask.view(-1)
                pos_idx = pos_flat.nonzero(as_tuple=False).squeeze(1)
                if pos_idx.numel() > 0:
                    cls_pos = target_labels.view(-1)[pos_idx].to(device=device, dtype=torch.long)
                    target_onehot_flat = target_onehot.view(-1, self.num_classes)
                    target_onehot_flat[pos_idx, cls_pos] = 1.0
                cls_loss = _sigmoid_focal_loss(
                    src_logits,
                    target_onehot,
                    alpha=float(self.focal_alpha),
                    gamma=float(self.focal_gamma),
                )
                # Normalize by number of positives.
                loss_cls = cls_loss.sum() / denom
            else:
                src_logits = pred_logits[..., : self.num_classes]
                target_scores = torch.zeros(
                    (b, self.num_proposals, self.num_classes),
                    device=device,
                    dtype=src_logits.dtype,
                )
                pos_flat = pos_mask.view(-1)
                pos_idx = pos_flat.nonzero(as_tuple=False).squeeze(1)
                if pos_idx.numel() > 0:
                    pred_pos = pred_boxes_abs.view(-1, 4)[pos_idx]
                    tgt_pos = target_boxes.view(-1, 4)[pos_idx]
                    iou = _pairwise_iou_diag(pred_pos, tgt_pos).clamp(0.0, 1.0).detach()
                    cls_pos = target_labels.view(-1)[pos_idx].to(device=device, dtype=torch.long)
                    target_scores_flat = target_scores.view(-1, self.num_classes)
                    target_scores_flat[pos_idx, cls_pos] = iou.to(dtype=target_scores_flat.dtype)

                pred_sigmoid = torch.sigmoid(src_logits)
                weight = torch.abs(target_scores - pred_sigmoid).pow(float(self.qfl_beta))
                cls_loss = F.binary_cross_entropy_with_logits(src_logits, target_scores, reduction="none") * weight
                loss_cls = cls_loss.sum() / denom

            if num_pos > 0:
                pred_pos = pred_boxes_abs[pos_mask]
                tgt_pos = target_boxes[pos_mask]

                whwh = torch.stack(
                    [image_sizes[:, 1], image_sizes[:, 0], image_sizes[:, 1], image_sizes[:, 0]],
                    dim=-1,
                ).to(device=device, dtype=pred_pos.dtype)
                pred_norm = (pred_boxes_abs / whwh[:, None, :])[pos_mask]
                tgt_norm = (target_boxes / whwh[:, None, :])[pos_mask]

                loss_bbox_raw = F.l1_loss(pred_norm, tgt_norm, reduction="none")  # (P,4)
                loss_giou_raw = 1.0 - _generalized_iou(pred_pos, tgt_pos)  # (P,)
                if self.box_loss_iou_weight_power > 0.0:
                    iou = _pairwise_iou_diag(pred_pos, tgt_pos).clamp(0.0, 1.0).detach()
                    w = (iou + 1e-3).pow(float(self.box_loss_iou_weight_power)).to(dtype=loss_bbox_raw.dtype)
                    loss_bbox_raw = loss_bbox_raw * w[:, None]
                    loss_giou_raw = loss_giou_raw * w

                loss_bbox = loss_bbox_raw.sum() / denom
                loss_giou = loss_giou_raw.sum() / denom
            else:
                loss_bbox = pred_boxes_abs.new_zeros(())
                loss_giou = pred_boxes_abs.new_zeros(())

            loss_cls_total = loss_cls_total + loss_cls
            loss_bbox_total = loss_bbox_total + loss_bbox
            loss_giou_total = loss_giou_total + loss_giou
            if outputs_quality is not None and self.use_quality_head and self.quality_loss_weight > 0.0:
                pred_q = outputs_quality[layer_idx]  # (B,N)
                if num_pos > 0:
                    iou = _pairwise_iou_diag(pred_pos, tgt_pos).clamp(0.0, 1.0).detach()
                    q_pos = pred_q[pos_mask]
                    if self.quality_loss_type == "bce":
                        loss_q = F.binary_cross_entropy_with_logits(q_pos, iou.to(dtype=q_pos.dtype), reduction="sum")
                    elif self.quality_loss_type == "l1":
                        loss_q = F.l1_loss(torch.sigmoid(q_pos), iou.to(dtype=torch.float32), reduction="sum")
                    elif self.quality_loss_type == "mse":
                        loss_q = F.mse_loss(torch.sigmoid(q_pos), iou.to(dtype=torch.float32), reduction="sum")
                    else:
                        raise ValueError(f"Unsupported quality_loss_type={self.quality_loss_type!r}")
                    loss_quality_total = loss_quality_total + (loss_q / denom)
                else:
                    loss_quality_total = loss_quality_total + pred_q.sum() * 0

        denom_layers = float(max(num_layers, 1))
        losses = {
            "loss_cls": (loss_cls_total / denom_layers) * float(self.loss_cls_weight),
            "loss_bbox": (loss_bbox_total / denom_layers) * float(self.loss_bbox_weight),
            "loss_giou": (loss_giou_total / denom_layers) * float(self.loss_giou_weight),
        }
        if self.use_quality_head and self.quality_loss_weight > 0.0:
            losses["loss_quality"] = (loss_quality_total / denom_layers) * float(self.quality_loss_weight)

        if self.graph_topo_loss_weight > 0.0:
            attn = getattr(self.head, "last_attn_weights", None)
            if attn is None:
                loss_graph = losses["loss_cls"] * 0
                losses["loss_graph"] = loss_graph
            else:
                pred_logits = outputs_class[-1]
                pred_boxes_abs = _clip_xyxy_to_image_min_size_ste(outputs_coord[-1], image_sizes, min_size=1.0)

                loss_graph = attn.sum() * 0
                pairs = 0

                for i in range(b):
                    gt_mask = graph_mask[i]
                    gt_boxes = graph_boxes[i][gt_mask]
                    gt_labels = graph_labels[i][gt_mask]
                    num_gt = int(gt_boxes.shape[0])
                    if num_gt <= 1:
                        continue

                    match = _hungarian_match(
                        pred_boxes=pred_boxes_abs[i],
                        pred_logits=pred_logits[i],
                        gt_boxes=gt_boxes,
                        gt_labels=gt_labels,
                        image_size=(int(image_sizes[i, 0].item()), int(image_sizes[i, 1].item())),
                        num_classes=self.num_classes,
                        use_sigmoid_cls=self.cls_loss_type in {"focal", "qfl"},
                        focal_cost=self.cls_loss_type == "focal",
                        focal_alpha=float(self.focal_alpha),
                        focal_gamma=float(self.focal_gamma),
                    )
                    if match.pred_indices.numel() == 0:
                        continue

                    qids = torch.full((num_gt,), -1, device=device, dtype=torch.long)
                    qids[match.gt_indices] = match.pred_indices
                    valid = qids >= 0
                    qids = qids[valid]
                    if qids.numel() <= 1:
                        continue

                    gt_boxes_sel = gt_boxes[valid].to(device=attn.device, dtype=torch.float32)
                    a_pred = attn[i].to(dtype=torch.float32)[qids][:, qids]  # (M,M)
                    iou = _pairwise_iou(gt_boxes_sel, gt_boxes_sel).to(dtype=torch.float32)
                    a_tgt = (iou > float(self.graph_topo_iou_thresh)).to(dtype=torch.float32)

                    m = int(qids.numel())
                    eye = torch.eye(m, device=attn.device, dtype=torch.float32)
                    a_pred = a_pred * (1.0 - eye)
                    a_tgt = a_tgt * (1.0 - eye)

                    if self.graph_topo_target_norm == "row":
                        denom_t = a_tgt.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                        a_tgt = a_tgt / denom_t
                        denom_p = a_pred.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                        a_pred = a_pred / denom_p

                    loss_graph = loss_graph + F.mse_loss(a_pred, a_tgt, reduction="sum")
                    pairs += m * (m - 1)

                if pairs > 0:
                    loss_graph = loss_graph / float(pairs)
                losses["loss_graph"] = loss_graph

            losses["loss_graph"] = losses["loss_graph"] * float(self.graph_topo_loss_weight)

        teacher = self._get_teacher(device=device)
        if (
            teacher is not None
            and self.sampler_distill
            and (self.sampler_distill_box_weight > 0.0 or self.sampler_distill_cls_weight > 0.0)
        ):
            init_noise = torch.randn((b, self.num_proposals, 4), device=device, dtype=torch.float32)
            if self.aniso_noise:
                sigma_vec = self.aniso_noise_sigma.to(device=init_noise.device, dtype=init_noise.dtype).view(1, 1, 4)
                init_noise = init_noise * sigma_vec

            init_label_state = self.label_scheduler.prior(b, self.num_proposals, device=device) if self.use_label_state else None

            # Student one-step prediction from max-noise boxes (t=T-1).
            t_noise = torch.full((b,), self.diffusion_timesteps - 1, device=device, dtype=torch.long)
            boxes_noise = _boxes_diffusion_to_abs(init_noise, image_sizes, box_scale=self.box_scale)
            boxes_noise = _clip_xyxy_to_image_min_size(boxes_noise, image_sizes, min_size=1.0)

            s_cls, s_coord, _s_q = self.head(
                feats=feats,
                init_bboxes=boxes_noise,
                timesteps=t_noise,
                label_state=init_label_state,
                image_sizes=image_sizes,
            )
            student_logits = s_cls[-1]
            student_boxes = _clip_xyxy_to_image_min_size_ste(s_coord[-1], image_sizes, min_size=1.0)

            with torch.no_grad():
                tea_out = teacher.sample(
                    batch_inputs,
                    batch_data_samples,
                    init_noise=init_noise,
                    init_label_state=init_label_state,
                    num_inference_steps=int(self.sampler_distill_teacher_sample_step),
                    eta=float(self.sampler_distill_teacher_eta),
                    do_postprocess=False,
                    disable_guidance=True,
                )
                teacher_logits = tea_out["pred_logits"].detach()
                teacher_boxes = tea_out["pred_boxes"].detach()

            topk_idx = None
            if self.sampler_distill_topk > 0:
                if self.cls_loss_type in {"focal", "qfl"}:
                    scores_t = teacher_logits[..., : self.num_classes].sigmoid().amax(dim=-1)
                else:
                    scores_t = teacher_logits.softmax(dim=-1)[..., : self.num_classes].amax(dim=-1)
                n_props = int(scores_t.shape[1])
                if self.sampler_distill_topk < n_props:
                    _, topk_idx = torch.topk(scores_t, k=int(self.sampler_distill_topk), dim=1, sorted=False)

            if topk_idx is not None:
                idx_box = topk_idx.unsqueeze(-1).expand(-1, -1, 4)
                student_boxes = student_boxes.gather(1, idx_box)
                teacher_boxes = teacher_boxes.gather(1, idx_box)
                idx_logit = topk_idx.unsqueeze(-1).expand(-1, -1, int(student_logits.shape[-1]))
                student_logits = student_logits.gather(1, idx_logit)
                teacher_logits = teacher_logits.gather(1, idx_logit)

            whwh = torch.stack(
                [image_sizes[:, 1], image_sizes[:, 0], image_sizes[:, 1], image_sizes[:, 0]],
                dim=-1,
            ).to(device=device, dtype=student_boxes.dtype)
            whwh = whwh.clamp(min=1.0)

            if self.sampler_distill_box_weight > 0.0:
                stu = student_boxes / whwh[:, None, :]
                tea = teacher_boxes / whwh[:, None, :]
                loss_box = F.l1_loss(stu, tea, reduction="mean")
                losses["loss_sampler_distill_box"] = loss_box * float(self.sampler_distill_box_weight)

            if self.sampler_distill_cls_weight > 0.0:
                temp = max(float(self.sampler_distill_cls_temperature), 1e-6)
                if self.cls_loss_type in {"focal", "qfl"}:
                    p_s = (student_logits[..., : self.num_classes] / temp).sigmoid()
                    p_t = (teacher_logits[..., : self.num_classes] / temp).sigmoid()
                else:
                    p_s = (student_logits / temp).softmax(dim=-1)[..., : self.num_classes]
                    p_t = (teacher_logits / temp).softmax(dim=-1)[..., : self.num_classes]
                loss_cls = F.mse_loss(p_s, p_t, reduction="mean")
                losses["loss_sampler_distill_cls"] = loss_cls * float(self.sampler_distill_cls_weight)

        for k, v in list(losses.items()):
            if not torch.isfinite(v).all():
                raise FloatingPointError(f"Non-finite loss detected: {k}={v}")

        return losses

    @torch.no_grad()
    def sample(
        self,
        batch_inputs: torch.Tensor,
        batch_data_samples: List,
        *,
        init_boxes: Optional[torch.Tensor] = None,
        init_noise: Optional[torch.Tensor] = None,
        init_label_state: Optional[torch.Tensor] = None,
        num_inference_steps: int,
        eta: float,
        do_postprocess: bool = True,
        disable_guidance: bool = False,
    ) -> dict:
        device = batch_inputs.device
        feats = self.extract_feat(batch_inputs)

        b = int(batch_inputs.shape[0])
        image_sizes = torch.stack(
            [torch.as_tensor(ds.metainfo.get("img_shape"), device=device, dtype=torch.int64) for ds in batch_data_samples],
            dim=0,
        )

        if init_boxes is not None and init_noise is not None:
            raise ValueError("Only one of init_boxes or init_noise can be set.")

        if init_noise is not None:
            if init_noise.dim() != 3 or init_noise.shape[-1] != 4:
                raise ValueError(f"init_noise must be (B,N,4), got {tuple(init_noise.shape)}")
            if int(init_noise.shape[0]) != b or int(init_noise.shape[1]) != int(self.num_proposals):
                raise ValueError(f"init_noise must be (B={b},N={self.num_proposals},4), got {tuple(init_noise.shape)}")
            x = init_noise.to(device=device, dtype=torch.float32)
        elif init_boxes is not None:
            if init_boxes.dim() != 3 or init_boxes.shape[-1] != 4:
                raise ValueError(f"init_boxes must be (B,N,4), got {tuple(init_boxes.shape)}")
            if int(init_boxes.shape[0]) != b or int(init_boxes.shape[1]) != int(self.num_proposals):
                raise ValueError(
                    f"init_boxes must be (B={b},N={self.num_proposals},4), got {tuple(init_boxes.shape)}"
                )
            x = _boxes_abs_to_diffusion(init_boxes.to(device=device), image_sizes, box_scale=self.box_scale)
        else:
            x = torch.randn((b, self.num_proposals, 4), device=device, dtype=torch.float32)
            if self.aniso_noise:
                sigma = self.aniso_noise_sigma.to(device=x.device, dtype=x.dtype).view(1, 1, 4)
                x = x * sigma

        if init_label_state is not None:
            if init_label_state.dim() == 2:
                label_state = init_label_state.to(device=device, dtype=torch.long)
            elif init_label_state.dim() == 3:
                label_state = init_label_state.to(device=device, dtype=torch.float32)
            else:
                raise ValueError(f"init_label_state must be (B,N) or (B,N,K+1), got {tuple(init_label_state.shape)}")
        else:
            label_state = self.label_scheduler.prior(b, self.num_proposals, device=device)

        num_inference_steps = max(int(num_inference_steps), 1)
        if self.box_scheduler is not None:
            self.box_scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.box_scheduler.timesteps
        else:
            timesteps = torch.linspace(
                self.diffusion_timesteps - 1, 0, steps=num_inference_steps, device=device, dtype=torch.long
            )
            timesteps = timesteps.clamp(min=0, max=self.diffusion_timesteps - 1)

        pred_logits = None
        pred_boxes_abs = None
        pred_x0 = None
        pred_quality = None

        for idx, t in enumerate(timesteps):
            time = int(t.item())
            time_cond = torch.full((b,), time, device=device, dtype=torch.long)

            boxes_t = _boxes_diffusion_to_abs(x, image_sizes, box_scale=self.box_scale)
            boxes_t = _clip_xyxy_to_image_min_size(boxes_t, image_sizes, min_size=1.0)

            outputs_class, outputs_coord, outputs_quality = self.head(
                feats=feats,
                init_bboxes=boxes_t,
                timesteps=time_cond,
                label_state=label_state,
                image_sizes=image_sizes,
            )
            pred_logits = outputs_class[-1]
            pred_boxes_abs = _clip_xyxy_to_image_min_size(outputs_coord[-1], image_sizes, min_size=1.0)
            pred_quality = outputs_quality[-1] if outputs_quality is not None else None

            use_guidance = (
                (not disable_guidance)
                and self.use_quality_head
                and pred_quality is not None
                and self.quality_guidance_scale > 0.0
                and getattr(self.head, "last_quality_feat", None) is not None
            )
            if use_guidance:
                mode = self.quality_guidance_mode
                if mode == "final":
                    use_guidance = (idx + 1) >= len(timesteps)
                elif mode == "all":
                    use_guidance = True
                elif mode == "threshold":
                    use_guidance = time <= int(self.quality_guidance_t_threshold)
                else:
                    raise ValueError(f"Unsupported quality_guidance_mode={mode!r}")

            if use_guidance:
                q_feat = self.head.last_quality_feat.detach()
                q_feat = q_feat.reshape(b * int(self.num_proposals), -1).to(device=device, dtype=torch.float32)

                if self.cls_loss_type in {"focal", "qfl"}:
                    score = pred_logits[..., : self.num_classes].sigmoid().amax(dim=-1).detach()
                else:
                    score = pred_logits.softmax(dim=-1)[..., : self.num_classes].amax(dim=-1).detach()
                topk_idx = None
                if self.quality_guidance_topk > 0 and self.quality_guidance_topk < score.shape[1]:
                    _, topk_idx = torch.topk(score, k=self.quality_guidance_topk, dim=1)

                step_scale = None
                if self.quality_guidance_step_schedule == "constant":
                    step_scale = None
                elif self.quality_guidance_step_schedule == "linear":
                    denom = float(max(self.diffusion_timesteps - 1, 1))
                    frac = (time_cond.to(dtype=torch.float32) / denom).clamp(0.0, 1.0)
                    ts = (1.0 - frac).clamp(0.0, 1.0)
                    if self.quality_guidance_time_power != 1.0:
                        ts = ts ** float(self.quality_guidance_time_power)
                    step_scale = (float(self.quality_guidance_scale) * ts).view(b, 1, 1)
                else:
                    raise ValueError(
                        f"Unsupported quality_guidance_step_schedule={self.quality_guidance_step_schedule!r}"
                    )

                def _sanitize_boxes(boxes: torch.Tensor) -> torch.Tensor:
                    x1, y1, x2, y2 = boxes.unbind(dim=-1)
                    x1n = torch.minimum(x1, x2)
                    x2n = torch.maximum(x1, x2)
                    y1n = torch.minimum(y1, y2)
                    y2n = torch.maximum(y1, y2)
                    boxes = torch.stack([x1n, y1n, x2n, y2n], dim=-1)
                    boxes = boxes.clamp(min=0.0)
                    boxes = torch.minimum(
                        boxes, image_sizes[:, None, [1, 0, 1, 0]].to(device=boxes.device, dtype=boxes.dtype)
                    )
                    x1, y1, x2, y2 = boxes.unbind(dim=-1)
                    x2 = torch.maximum(x2, x1 + 1e-6)
                    y2 = torch.maximum(y2, y1 + 1e-6)
                    return torch.stack([x1, y1, x2, y2], dim=-1)

                rcnn_last = self.head.head_series[-1]
                boxes = _sanitize_boxes(pred_boxes_abs.detach())
                with torch.enable_grad():
                    for _ in range(int(self.quality_guidance_langevin_steps)):
                        boxes_var = boxes.reshape(-1, 4).detach().requires_grad_(True)
                        boxes_view = boxes_var.view(b, int(self.num_proposals), 4)
                        boxes_sorted = _sanitize_boxes(boxes_view).reshape(-1, 4)

                        q_geo = rcnn_last._build_quality_box_features(
                            boxes_sorted, dtype=q_feat.dtype, diffusion_t=time_cond, nr_boxes=int(self.num_proposals)
                        )
                        q_in2 = torch.cat([q_feat, q_geo.to(device=q_feat.device, dtype=q_feat.dtype)], dim=1)
                        q_logits = rcnn_last.quality_head(q_in2).squeeze(-1).view(b, int(self.num_proposals))
                        q = torch.sigmoid(q_logits)

                        if topk_idx is not None:
                            q_sel = q.gather(1, topk_idx)
                            score_sel = score.gather(1, topk_idx)
                        else:
                            q_sel = q
                            score_sel = score

                        obj = q_sel
                        if self.quality_guidance_score_weight:
                            obj = obj * score_sel
                        obj = obj.mean()

                        grad = torch.autograd.grad(obj, boxes_var, retain_graph=False, create_graph=False)[0]
                        grad = grad.view(b, int(self.num_proposals), 4)
                        if self.quality_guidance_grad_norm == "proposal":
                            grad = grad / (grad.norm(dim=-1, keepdim=True) + 1e-6)
                        else:
                            denom_g = grad.reshape(grad.shape[0], -1).norm(dim=-1).view(-1, 1, 1)
                            grad = grad / (denom_g + 1e-6)

                        if step_scale is None:
                            boxes = (boxes_view + float(self.quality_guidance_scale) * grad).detach()
                        else:
                            boxes = (
                                boxes_view + step_scale.to(device=grad.device, dtype=grad.dtype) * grad
                            ).detach()

                        if self.quality_guidance_langevin_noise > 0.0:
                            boxes = boxes + float(self.quality_guidance_langevin_noise) * torch.randn_like(boxes)
                        boxes = _sanitize_boxes(boxes)

                pred_boxes_abs = boxes

            pred_x0 = _boxes_abs_to_diffusion(pred_boxes_abs, image_sizes, box_scale=self.box_scale)

            if self.box_scheduler is not None:
                step_out = self.box_scheduler.step(model_output=pred_x0, timestep=time, sample=x, eta=float(eta))
                x = step_out.prev_sample
                t_next = int(timesteps[idx + 1].item()) if (idx + 1) < len(timesteps) else -1
                label_state = self.label_scheduler.infer_update(
                    pred_logits[..., : self.num_classes], t_next=t_next, device=device
                )
            else:
                # Final-step DDIM fallback: no stochasticity.
                if (idx + 1) < len(timesteps):
                    time_next = int(timesteps[idx + 1].item())
                    sqrt_alpha_t = self.sqrt_alphas_cumprod[time].to(device=device, dtype=torch.float32)
                    sqrt_beta_t = self.sqrt_one_minus_alphas_cumprod[time].to(device=device, dtype=torch.float32)
                    sqrt_alpha_prev = self.sqrt_alphas_cumprod[time_next].to(device=device, dtype=torch.float32)
                    sqrt_beta_prev = self.sqrt_one_minus_alphas_cumprod[time_next].to(device=device, dtype=torch.float32)
                    eps = (x - sqrt_alpha_t * pred_x0) / sqrt_beta_t.clamp(min=1e-6)
                    x = sqrt_alpha_prev * pred_x0 + sqrt_beta_prev * eps
                    label_state = self.label_scheduler.infer_update(
                        pred_logits[..., : self.num_classes], t_next=time_next, device=device
                    )
                else:
                    x = pred_x0
                    label_state = self.label_scheduler.infer_update(
                        pred_logits[..., : self.num_classes], t_next=-1, device=device
                    )

        if pred_logits is None or pred_boxes_abs is None:
            raise RuntimeError("Sampling produced no predictions.")

        data_samples = None
        if do_postprocess:
            results_list = []
            for i in range(b):
                h, w = int(image_sizes[i, 0].item()), int(image_sizes[i, 1].item())
                boxes_all = _clamp_xyxy_single(pred_boxes_abs[i].clone(), height=h, width=w)

                if self.cls_loss_type in {"focal", "qfl"}:
                    # Sigmoid-style multi-label scores: take top-N across (proposal, class).
                    probs_i = pred_logits[i, :, : self.num_classes].sigmoid()  # (N,K)
                    scores_flat = probs_i.flatten(0, 1)
                    topk = min(int(self.num_proposals), int(scores_flat.numel()))
                    scores_i, topk_idx = torch.topk(scores_flat, k=topk, sorted=False)
                    labels_i = (topk_idx % int(self.num_classes)).to(dtype=torch.long)
                    box_idx = (topk_idx // int(self.num_classes)).to(dtype=torch.long)
                    boxes_i = boxes_all[box_idx]
                else:
                    probs_i = pred_logits[i].softmax(dim=-1)
                    scores_i, labels_i = probs_i[..., : self.num_classes].max(dim=-1)
                    boxes_i = boxes_all

                keep = scores_i > self.score_thr
                boxes_i = boxes_i[keep]
                scores_i = scores_i[keep]
                labels_i = labels_i[keep]

                if boxes_i.numel() == 0:
                    inst = InstanceData(bboxes=boxes_i, scores=scores_i, labels=labels_i)
                    results_list.append(inst)
                    continue

                kept_boxes = []
                kept_scores = []
                kept_labels = []
                for c in range(self.num_classes):
                    cls_mask = labels_i == c
                    if not torch.any(cls_mask):
                        continue
                    b_c = boxes_i[cls_mask]
                    s_c = scores_i[cls_mask]
                    keep_idx = nms(b_c, s_c, self.nms_iou_thr)
                    kept_boxes.append(b_c[keep_idx])
                    kept_scores.append(s_c[keep_idx])
                    kept_labels.append(torch.full((keep_idx.numel(),), c, device=device, dtype=torch.long))

                if kept_boxes:
                    b_out = torch.cat(kept_boxes, dim=0)
                    s_out = torch.cat(kept_scores, dim=0)
                    l_out = torch.cat(kept_labels, dim=0)
                    if b_out.shape[0] > self.max_per_img:
                        topk = torch.topk(s_out, k=self.max_per_img).indices
                        b_out = b_out[topk]
                        s_out = s_out[topk]
                        l_out = l_out[topk]
                else:
                    b_out = boxes_i[:0]
                    s_out = scores_i[:0]
                    l_out = labels_i[:0]

                scale_factor = batch_data_samples[i].metainfo.get("scale_factor", None)
                if scale_factor is not None:
                    sf = torch.as_tensor(scale_factor, device=device, dtype=b_out.dtype).flatten()
                    if sf.numel() == 2:
                        sf = sf.repeat(2)
                    if sf.numel() == 4 and torch.all(sf > 0):
                        b_out = b_out / sf
                        ori_shape = batch_data_samples[i].metainfo.get("ori_shape", None)
                        if ori_shape is not None:
                            oh, ow = int(ori_shape[0]), int(ori_shape[1])
                            b_out = _clamp_xyxy_single(b_out, height=oh, width=ow)

                inst = InstanceData(bboxes=b_out, scores=s_out, labels=l_out)
                results_list.append(inst)

            data_samples = self.add_pred_to_datasample(batch_data_samples, results_list)
        return {
            "pred_boxes": pred_boxes_abs,
            "pred_logits": pred_logits,
            "pred_quality": pred_quality,
            "data_samples": data_samples,
        }

    @torch.no_grad()
    def predict(self, batch_inputs: torch.Tensor, batch_data_samples: List) -> List:
        out = self.sample(
            batch_inputs,
            batch_data_samples,
            num_inference_steps=int(self.sampling_timesteps),
            eta=float(self.ddim_sampling_eta),
        )
        return out["data_samples"]

    def _forward(self, batch_inputs: torch.Tensor, batch_data_samples: Optional[List] = None):
        feats = self.extract_feat(batch_inputs)
        return feats
