from __future__ import annotations

import copy
import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from mmdet.registry import MODELS


_DEFAULT_SCALE_CLAMP = math.log(100000.0 / 16)
_DEFAULT_MIN_BOX_SIZE = 1.0


def _sanitize_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    x1n = torch.minimum(x1, x2)
    x2n = torch.maximum(x1, x2)
    y1n = torch.minimum(y1, y2)
    y2n = torch.maximum(y1, y2)
    return torch.stack([x1n, y1n, x2n, y2n], dim=-1)


def _clip_boxes_to_image(
    boxes: torch.Tensor,
    image_sizes: torch.Tensor,
    *,
    min_size: float = _DEFAULT_MIN_BOX_SIZE,
) -> torch.Tensor:
    """
    boxes: (B,N,4) abs xyxy
    image_sizes: (B,2) int64 [H,W]
    """
    if boxes.dim() != 3 or boxes.shape[-1] != 4:
        raise ValueError(f"Expected boxes (B,N,4), got {tuple(boxes.shape)}")
    if image_sizes.dim() != 2 or image_sizes.shape[-1] != 2 or image_sizes.shape[0] != boxes.shape[0]:
        raise ValueError(f"Expected image_sizes (B,2), got {tuple(image_sizes.shape)} for boxes {tuple(boxes.shape)}")

    boxes = _sanitize_xyxy(boxes)
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

    eps = torch.as_tensor(float(min_size), device=boxes.device, dtype=dtype).clamp(min=0.0)
    min_w = torch.minimum(w, eps)
    min_h = torch.minimum(h, eps)

    x1 = torch.minimum(x1, w - min_w)
    y1 = torch.minimum(y1, h - min_h)
    x1 = x1.clamp(min=0.0)
    y1 = y1.clamp(min=0.0)
    x2 = torch.maximum(x2, x1 + min_w)
    y2 = torch.maximum(y2, y1 + min_h)
    x2 = torch.minimum(x2, w)
    y2 = torch.minimum(y2, h)
    return torch.stack([x1, y1, x2, y2], dim=-1)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        # time: (B,) float/long
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None].to(dtype=torch.float32) * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        if self.dim % 2 == 1:
            embeddings = torch.cat([embeddings, torch.zeros((embeddings.shape[0], 1), device=device)], dim=-1)
        return embeddings


class DynamicConv(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        dim_dynamic: int,
        num_dynamic: int,
        pooler_resolution: int,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.dim_dynamic = int(dim_dynamic)
        self.num_dynamic = int(num_dynamic)
        self.num_params = self.hidden_dim * self.dim_dynamic

        self.dynamic_layer = nn.Linear(self.hidden_dim, self.num_dynamic * self.num_params)
        self.norm1 = nn.LayerNorm(self.dim_dynamic)
        self.norm2 = nn.LayerNorm(self.hidden_dim)
        self.activation = nn.ReLU(inplace=True)

        num_output = self.hidden_dim * int(pooler_resolution) ** 2
        self.out_layer = nn.Linear(num_output, self.hidden_dim)
        self.norm3 = nn.LayerNorm(self.hidden_dim)

    def forward(self, pro_features: torch.Tensor, roi_features: torch.Tensor) -> torch.Tensor:
        """
        pro_features: (1, B*N, D)
        roi_features: (R*R, B*N, D)
        returns: (B*N, D)
        """
        features = roi_features.permute(1, 0, 2)  # (B*N, R*R, D)
        parameters = self.dynamic_layer(pro_features).permute(1, 0, 2)  # (B*N, 1, P)

        param1 = parameters[:, :, : self.num_params].view(-1, self.hidden_dim, self.dim_dynamic)
        param2 = parameters[:, :, self.num_params :].view(-1, self.dim_dynamic, self.hidden_dim)

        features = torch.bmm(features, param1)
        features = self.norm1(features)
        features = self.activation(features)

        features = torch.bmm(features, param2)
        features = self.norm2(features)
        features = self.activation(features)

        features = features.flatten(1)
        features = self.out_layer(features)
        features = self.norm3(features)
        features = self.activation(features)
        return features


def _get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(int(n))])


def _get_activation_fn(name: str):
    name = str(name).lower()
    if name == "relu":
        return F.relu
    if name == "gelu":
        return F.gelu
    if name == "glu":
        return F.glu
    raise ValueError(f"Unsupported activation={name!r}")


class RCNNHead(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int,
        d_model: int,
        dim_feedforward: int,
        nhead: int,
        dropout: float,
        activation: str,
        pooler_resolution: int,
        dim_dynamic: int = 64,
        num_dynamic: int = 2,
        num_cls_layers: int = 1,
        num_reg_layers: int = 1,
        bbox_weights: tuple[float, float, float, float] = (2.0, 2.0, 1.0, 1.0),
        scale_clamp: float = _DEFAULT_SCALE_CLAMP,
        # graph transformer / geo bias
        use_geo_bias: bool = True,
        geo_bias_type: str = "mlp",
        geo_bias_scale: float = 1.0,
        geo_bias_norm: float = 1000.0,
        geo_bias_use_log_wh: bool = True,
        geo_bias_input_clip: float = 0.0,
        geo_bias_rel_include_iou: bool = False,
        geo_bias_mlp_hidden_dim: int = 64,
        geo_bias_out_tanh: bool = True,
        geo_bias_topk: int = 0,
        geo_bias_learnable_scale: bool = False,
        diffusion_timesteps: int = 1000,
        geo_bias_schedule: str = "constant",
        geo_bias_time_power: float = 1.0,
        geo_bias_t_threshold: int = 0,
        geo_bias_sigma: float = 2.0,
        geo_bias_min_norm: float = 0.0,
        # label state
        use_label_state: bool = True,
        label_state_scale: float = 0.1,
        label_state_alpha_init: float = 0.1,
        label_state_relative_to_unk: bool = True,
        label_state_proj_zero_init: bool = True,
        # self-attn capture (for graph consistency loss)
        capture_graph_attn: bool = False,
        # quality head (IoU prediction / energy guidance)
        use_quality_head: bool = False,
        quality_head_hidden_dim: int = 256,
        quality_head_include_t: bool = True,
        quality_head_use_log_wh: bool = True,
        quality_box_norm: float = 1000.0,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.pooler_resolution = int(pooler_resolution)

        self.use_geo_bias = bool(use_geo_bias)
        self.geo_bias_type = str(geo_bias_type).lower()
        self.geo_bias_scale = float(geo_bias_scale)
        self.geo_bias_norm = float(geo_bias_norm)
        self.geo_bias_use_log_wh = bool(geo_bias_use_log_wh)
        self.geo_bias_input_clip = float(geo_bias_input_clip)
        self.geo_bias_rel_include_iou = bool(geo_bias_rel_include_iou)
        self.geo_bias_mlp_hidden_dim = int(geo_bias_mlp_hidden_dim)
        self.geo_bias_out_tanh = bool(geo_bias_out_tanh)
        self.geo_bias_topk = int(geo_bias_topk)
        self.geo_bias_learnable_scale = bool(geo_bias_learnable_scale)
        self.diffusion_timesteps = int(diffusion_timesteps)
        self.geo_bias_schedule = str(geo_bias_schedule).lower()
        self.geo_bias_time_power = float(geo_bias_time_power)
        self.geo_bias_t_threshold = int(geo_bias_t_threshold)
        self.geo_bias_sigma = float(geo_bias_sigma)
        self.geo_bias_min_norm = float(geo_bias_min_norm)

        self.capture_graph_attn = bool(capture_graph_attn)
        self.last_attn_weights: torch.Tensor | None = None
        self.last_quality_feat: torch.Tensor | None = None

        if self.use_geo_bias and self.geo_bias_learnable_scale:
            self.geo_bias_scale_param = nn.Parameter(torch.zeros(self.nhead))

        if self.use_geo_bias and self.geo_bias_type == "mlp":
            rel_in_dim = 4 + (1 if self.geo_bias_rel_include_iou else 0)
            hidden = max(self.geo_bias_mlp_hidden_dim, 1)
            self.geo_bias_mlp = nn.Sequential(
                nn.Linear(rel_in_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, 1),
            )

        self.use_label_state = bool(use_label_state)
        self.label_state_scale = float(label_state_scale)
        self.label_state_alpha_init = float(label_state_alpha_init)
        self.label_state_relative_to_unk = bool(label_state_relative_to_unk)
        self.label_state_proj_zero_init = bool(label_state_proj_zero_init)
        if self.use_label_state:
            alpha = min(max(self.label_state_alpha_init, 1e-4), 1.0 - 1e-4)
            self.label_state_alpha_param = nn.Parameter(torch.tensor(math.log(alpha / (1.0 - alpha))))
            self.label_state_embed = nn.Embedding(int(num_classes) + 1, self.d_model)
            self.label_state_proj = nn.Linear(self.d_model, self.d_model)
            if self.label_state_proj_zero_init:
                nn.init.constant_(self.label_state_proj.weight, 0.0)
                if self.label_state_proj.bias is not None:
                    nn.init.constant_(self.label_state_proj.bias, 0.0)

        self.self_attn = nn.MultiheadAttention(self.d_model, self.nhead, dropout=float(dropout))
        self.inst_interact = DynamicConv(
            hidden_dim=self.d_model,
            dim_dynamic=int(dim_dynamic),
            num_dynamic=int(num_dynamic),
            pooler_resolution=self.pooler_resolution,
        )

        self.linear1 = nn.Linear(self.d_model, int(dim_feedforward))
        self.dropout = nn.Dropout(float(dropout))
        self.linear2 = nn.Linear(int(dim_feedforward), self.d_model)

        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.norm3 = nn.LayerNorm(self.d_model)
        self.dropout1 = nn.Dropout(float(dropout))
        self.dropout2 = nn.Dropout(float(dropout))
        self.dropout3 = nn.Dropout(float(dropout))

        self.activation = _get_activation_fn(activation)

        # time conditioning
        self.block_time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(self.d_model * 4, self.d_model * 2))

        # cls/reg towers
        cls_module = []
        for _ in range(max(int(num_cls_layers), 0)):
            cls_module.append(nn.Linear(self.d_model, self.d_model, bias=False))
            cls_module.append(nn.LayerNorm(self.d_model))
            cls_module.append(nn.ReLU(inplace=True))
        self.cls_module = nn.ModuleList(cls_module)

        reg_module = []
        for _ in range(max(int(num_reg_layers), 0)):
            reg_module.append(nn.Linear(self.d_model, self.d_model, bias=False))
            reg_module.append(nn.LayerNorm(self.d_model))
            reg_module.append(nn.ReLU(inplace=True))
        self.reg_module = nn.ModuleList(reg_module)

        self.class_logits = nn.Linear(self.d_model, int(num_classes) + 1)
        self.bboxes_delta = nn.Linear(self.d_model, 4)

        self.scale_clamp = float(scale_clamp)
        self.bbox_weights = bbox_weights

        self.use_quality_head = bool(use_quality_head)
        self.quality_head_hidden_dim = int(quality_head_hidden_dim)
        self.quality_head_include_t = bool(quality_head_include_t)
        self.quality_head_use_log_wh = bool(quality_head_use_log_wh)
        self.quality_box_norm = float(quality_box_norm)
        if self.use_quality_head:
            quality_in_dim = self.d_model + 4 + (1 if self.quality_head_include_t else 0)
            hidden = max(int(self.quality_head_hidden_dim), 1)
            self.quality_head = nn.Sequential(
                nn.Linear(quality_in_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, 1),
            )

    def reset_quality_head_parameters(self) -> None:
        if not hasattr(self, "quality_head"):
            return
        last = self.quality_head[-1]
        if isinstance(last, nn.Linear):
            nn.init.constant_(last.weight, 0.0)
            if last.bias is not None:
                nn.init.constant_(last.bias, 0.0)

    def reset_geo_bias_parameters(self) -> None:
        if not hasattr(self, "geo_bias_mlp"):
            return
        last = self.geo_bias_mlp[-1]
        if isinstance(last, nn.Linear):
            nn.init.constant_(last.weight, 0.0)
            if last.bias is not None:
                nn.init.constant_(last.bias, 0.0)

    def reset_label_state_parameters(self) -> None:
        if not getattr(self, "label_state_proj_zero_init", False):
            return
        if hasattr(self, "label_state_proj") and isinstance(self.label_state_proj, nn.Linear):
            nn.init.constant_(self.label_state_proj.weight, 0.0)
            if self.label_state_proj.bias is not None:
                nn.init.constant_(self.label_state_proj.bias, 0.0)

    def _label_state_alpha(self, dtype: torch.dtype) -> torch.Tensor:
        if not hasattr(self, "label_state_alpha_param"):
            return torch.tensor(1.0, dtype=dtype)
        return torch.sigmoid(self.label_state_alpha_param).to(dtype=dtype)

    def _build_quality_box_features(
        self,
        boxes_xyxy: torch.Tensor,
        *,
        dtype: torch.dtype,
        diffusion_t: Optional[torch.Tensor],
        nr_boxes: int,
    ) -> torch.Tensor:
        if boxes_xyxy.dim() != 2 or boxes_xyxy.shape[-1] != 4:
            raise ValueError(f"Expected boxes_xyxy (B*N,4), got {tuple(boxes_xyxy.shape)}")

        x1, y1, x2, y2 = boxes_xyxy.unbind(dim=-1)
        x1n = torch.minimum(x1, x2)
        x2n = torch.maximum(x1, x2)
        y1n = torch.minimum(y1, y2)
        y2n = torch.maximum(y1, y2)

        cx = (x1n + x2n) * 0.5
        cy = (y1n + y2n) * 0.5
        w = (x2n - x1n).clamp(min=1e-6)
        h = (y2n - y1n).clamp(min=1e-6)

        norm = max(float(self.quality_box_norm), 1e-6)
        cx = cx / norm
        cy = cy / norm
        w = w / norm
        h = h / norm
        if self.quality_head_use_log_wh:
            w = torch.log(w + 1e-6)
            h = torch.log(h + 1e-6)

        feats = [cx, cy, w, h]
        if self.quality_head_include_t:
            if diffusion_t is None:
                raise ValueError("diffusion_t is required when quality_head_include_t is True")
            denom = float(max(self.diffusion_timesteps - 1, 1))
            t = diffusion_t.to(dtype=torch.float32) / denom
            t = t.clamp(0.0, 1.0)
            t = torch.repeat_interleave(t, int(nr_boxes), dim=0).to(dtype=cx.dtype)
            feats.append(t)

        return torch.stack(feats, dim=-1).to(dtype=dtype)

    def _build_geometry_attn_bias(
        self,
        bboxes: torch.Tensor,  # (B,N,4) abs xyxy
        *,
        dtype: torch.dtype,
        diffusion_t: Optional[torch.Tensor],
    ) -> torch.Tensor | None:
        if not self.use_geo_bias:
            return None
        if not self.geo_bias_learnable_scale and self.geo_bias_scale == 0.0:
            return None

        time_scale = None
        if self.geo_bias_schedule == "constant":
            time_scale = None
        elif self.geo_bias_schedule == "linear":
            if diffusion_t is None:
                raise ValueError("diffusion_t is required when GEO_BIAS_SCHEDULE == 'linear'")
            denom = float(max(self.diffusion_timesteps - 1, 1))
            t = diffusion_t.detach().to(dtype=torch.float32)
            frac = (t / denom).clamp(0.0, 1.0)
            time_scale = (1.0 - frac).clamp(0.0, 1.0)
            if self.geo_bias_time_power != 1.0:
                time_scale = time_scale ** float(self.geo_bias_time_power)
        elif self.geo_bias_schedule == "threshold":
            if diffusion_t is None:
                raise ValueError("diffusion_t is required when GEO_BIAS_SCHEDULE == 'threshold'")
            t = diffusion_t.detach()
            keep = t <= int(self.geo_bias_t_threshold)
            if not torch.any(keep):
                return None
            time_scale = keep.to(dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported GEO_BIAS_SCHEDULE={self.geo_bias_schedule!r}")

        boxes = bboxes.detach()
        x1, y1, x2, y2 = boxes.unbind(dim=-1)

        if self.geo_bias_type == "distance":
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5
            w = (x2 - x1).clamp(min=1e-6)
            h = (y2 - y1).clamp(min=1e-6)
            size = torch.sqrt(w * h)
            dx = cx[:, :, None] - cx[:, None, :]
            dy = cy[:, :, None] - cy[:, None, :]
            dist = torch.sqrt(dx * dx + dy * dy + 1e-6)
            norm = (size[:, :, None] + size[:, None, :]) * 0.5
            if self.geo_bias_min_norm > 0.0:
                norm = norm.clamp(min=self.geo_bias_min_norm)
            dist = dist / (norm + 1e-6)
            sigma = max(self.geo_bias_sigma, 1e-6)
            bias = -((dist / sigma) ** 2)
            bias = bias.clamp(min=-10.0, max=0.0)
        elif self.geo_bias_type == "mlp":
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5
            w = (x2 - x1).clamp(min=1e-6)
            h = (y2 - y1).clamp(min=1e-6)

            norm = max(self.geo_bias_norm, 1e-6)
            cx = cx / norm
            cy = cy / norm
            w = w / norm
            h = h / norm
            if self.geo_bias_use_log_wh:
                w = torch.log(w + 1e-6)
                h = torch.log(h + 1e-6)

            dx = (cx[:, :, None] - cx[:, None, :]).to(dtype=torch.float32)
            dy = (cy[:, :, None] - cy[:, None, :]).to(dtype=torch.float32)
            dw = (w[:, :, None] - w[:, None, :]).to(dtype=torch.float32)
            dh = (h[:, :, None] - h[:, None, :]).to(dtype=torch.float32)
            rel_feats = [dx, dy, dw, dh]

            if self.geo_bias_rel_include_iou:
                xx1 = torch.maximum(x1[:, :, None], x1[:, None, :])
                yy1 = torch.maximum(y1[:, :, None], y1[:, None, :])
                xx2 = torch.minimum(x2[:, :, None], x2[:, None, :])
                yy2 = torch.minimum(y2[:, :, None], y2[:, None, :])
                inter_w = (xx2 - xx1).clamp(min=0.0)
                inter_h = (yy2 - yy1).clamp(min=0.0)
                inter = inter_w * inter_h
                area = (x2 - x1).clamp(min=0.0) * (y2 - y1).clamp(min=0.0)
                union = area[:, :, None] + area[:, None, :] - inter
                iou = inter / (union + 1e-6)
                rel_feats.append(iou.to(dtype=torch.float32))

            rel = torch.stack(rel_feats, dim=-1)
            if self.geo_bias_input_clip > 0.0:
                clip = float(self.geo_bias_input_clip)
                rel = rel.clamp(min=-clip, max=clip)
            bias = self.geo_bias_mlp(rel).squeeze(-1)
            if self.geo_bias_out_tanh:
                bias = torch.tanh(bias)

            nr_boxes = bias.shape[-1]
            eye = torch.eye(nr_boxes, device=bias.device, dtype=torch.bool)
            bias = bias.masked_fill(eye[None, :, :], 0.0)
        else:
            raise ValueError(f"Unsupported GEO_BIAS_TYPE={self.geo_bias_type!r}")

        if time_scale is not None:
            bias = bias * time_scale[:, None, None]

        keep = None
        topk = int(self.geo_bias_topk)
        n, nr_boxes = bias.shape[:2]
        if topk > 0 and topk < nr_boxes:
            eye = torch.eye(nr_boxes, device=bias.device, dtype=torch.bool)
            bias_for_topk = bias.masked_fill(eye[None, :, :], -1.0e9)
            _, idx = torch.topk(bias_for_topk, k=topk, dim=-1, sorted=False)
            keep = torch.zeros_like(bias_for_topk, dtype=torch.bool)
            keep.scatter_(-1, idx, True)
            keep = keep | eye[None, :, :]

        bias = bias.to(dtype=dtype)
        if self.geo_bias_learnable_scale:
            scale = self.geo_bias_scale * torch.tanh(self.geo_bias_scale_param)
            scale = scale.to(dtype=dtype)
            bias_h = bias.unsqueeze(1) * scale[None, :, None, None]
        else:
            bias_h = (bias * self.geo_bias_scale).unsqueeze(1).repeat(1, self.nhead, 1, 1)

        if keep is not None:
            bias_h = bias_h.masked_fill(~keep[:, None, :, :], -1.0e4)
        return bias_h.reshape(n * self.nhead, nr_boxes, nr_boxes)

    def apply_deltas(self, deltas: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        boxes = _sanitize_xyxy(boxes).to(deltas.dtype)
        widths = (boxes[:, 2] - boxes[:, 0]).clamp(min=1e-6)
        heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=1e-6)
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.bbox_weights
        dx = deltas[:, 0::4] / wx
        dy = deltas[:, 1::4] / wy
        dw = deltas[:, 2::4] / ww
        dh = deltas[:, 3::4] / wh

        dw = torch.clamp(dw, min=-self.scale_clamp, max=self.scale_clamp)
        dh = torch.clamp(dh, min=-self.scale_clamp, max=self.scale_clamp)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h
        return pred_boxes

    def forward(
        self,
        *,
        feats: list[torch.Tensor],
        bboxes: torch.Tensor,  # (B,N,4) abs xyxy
        pro_features: Optional[torch.Tensor],  # (B,N,D) or None
        roi_extractor: nn.Module,
        time_emb: torch.Tensor,  # (B, D*4)
        diffusion_t: Optional[torch.Tensor],
        label_state: Optional[torch.Tensor],
        image_sizes: Optional[torch.Tensor] = None,  # (B,2) int64 [H,W]
    ):
        b, nr_boxes = bboxes.shape[:2]
        if image_sizes is not None:
            bboxes = _clip_boxes_to_image(bboxes, image_sizes, min_size=_DEFAULT_MIN_BOX_SIZE)

        # rois: (B*N, 5) with batch-ind
        rois = torch.cat(
            [
                torch.arange(b, device=bboxes.device, dtype=torch.float32).view(b, 1, 1).expand(b, nr_boxes, 1),
                bboxes.to(dtype=torch.float32),
            ],
            dim=-1,
        ).reshape(-1, 5)
        roi_features = roi_extractor(feats, rois)  # (B*N, C, R, R)

        if pro_features is None:
            pro_features = roi_features.view(b, nr_boxes, self.d_model, -1).mean(-1)
        else:
            pro_features = pro_features.view(b, nr_boxes, self.d_model)

        if self.use_label_state and label_state is not None:
            if label_state.dim() == 2:
                emb = self.label_state_embed(label_state.to(device=pro_features.device, dtype=torch.long))
            elif label_state.dim() == 3:
                probs = label_state.to(device=pro_features.device, dtype=torch.float32)
                weight = self.label_state_embed.weight.to(dtype=torch.float32)
                if probs.shape[-1] != weight.shape[0]:
                    raise ValueError(
                        f"label_state last dim {probs.shape[-1]} must match embedding states {weight.shape[0]}"
                    )
                if self.training:
                    flat = probs.reshape(-1, probs.shape[-1])
                    flat = flat / flat.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                    sampled = torch.multinomial(flat, num_samples=1).squeeze(1)
                    sampled = sampled.view(probs.shape[0], probs.shape[1]).to(dtype=torch.long)
                    emb = self.label_state_embed(sampled)
                else:
                    emb = torch.matmul(probs, weight).to(dtype=pro_features.dtype)
            else:
                raise ValueError(f"Unsupported label_state shape={tuple(label_state.shape)}")

            proj = self.label_state_proj(emb.to(dtype=pro_features.dtype))
            if self.label_state_relative_to_unk:
                unk_id = int(self.label_state_embed.weight.shape[0] - 1)
                unk = torch.tensor([unk_id], device=pro_features.device, dtype=torch.long)
                unk_emb = self.label_state_embed(unk).to(dtype=pro_features.dtype)
                unk_proj = self.label_state_proj(unk_emb).view(1, 1, -1)
                delta = proj - unk_proj
            else:
                delta = proj
            scale = float(self.label_state_scale) * self._label_state_alpha(dtype=pro_features.dtype)
            pro_features = pro_features + delta * scale

        roi_features = roi_features.view(b * nr_boxes, self.d_model, -1).permute(2, 0, 1)  # (R*R, B*N, D)

        # self-attn on proposals (graph transformer)
        pro_features = pro_features.view(b, nr_boxes, self.d_model).permute(1, 0, 2)  # (N, B, D)
        attn_bias = self._build_geometry_attn_bias(bboxes, dtype=pro_features.dtype, diffusion_t=diffusion_t)

        if self.capture_graph_attn:
            pro_features2, attn_weights = self.self_attn(
                pro_features, pro_features, pro_features, attn_mask=attn_bias, need_weights=True, average_attn_weights=True
            )
            self.last_attn_weights = attn_weights  # (B, N, N)
        else:
            pro_features2 = self.self_attn(pro_features, pro_features, pro_features, attn_mask=attn_bias, need_weights=False)[0]
            self.last_attn_weights = None

        pro_features = pro_features + self.dropout1(pro_features2)
        pro_features = self.norm1(pro_features)

        # inst interact (dynamic conv)
        pro_features = pro_features.view(nr_boxes, b, self.d_model).permute(1, 0, 2).reshape(1, b * nr_boxes, self.d_model)
        pro_features2 = self.inst_interact(pro_features, roi_features)
        pro_features = pro_features + self.dropout2(pro_features2)
        obj_features = self.norm2(pro_features)

        # ffn
        obj_features2 = self.linear2(self.dropout(self.activation(self.linear1(obj_features))))
        obj_features = obj_features + self.dropout3(obj_features2)
        obj_features = self.norm3(obj_features)

        fc_feature = obj_features.transpose(0, 1).reshape(b * nr_boxes, -1)

        scale_shift = self.block_time_mlp(time_emb)
        scale_shift = torch.repeat_interleave(scale_shift, nr_boxes, dim=0)
        scale, shift = scale_shift.chunk(2, dim=1)
        fc_feature = fc_feature * (scale + 1) + shift

        if self.use_quality_head:
            self.last_quality_feat = fc_feature.detach()
        else:
            self.last_quality_feat = None

        cls_feature = fc_feature
        reg_feature = fc_feature
        for layer in self.cls_module:
            cls_feature = layer(cls_feature)
        for layer in self.reg_module:
            reg_feature = layer(reg_feature)

        class_logits = self.class_logits(cls_feature)  # (B*N, K+1)
        bboxes_deltas = self.bboxes_delta(reg_feature)  # (B*N, 4)
        pred_bboxes = self.apply_deltas(bboxes_deltas, bboxes.view(-1, 4))

        pred_quality = None
        if self.use_quality_head:
            q_geo = self._build_quality_box_features(
                boxes_xyxy=bboxes.reshape(-1, 4),
                dtype=fc_feature.dtype,
                diffusion_t=diffusion_t,
                nr_boxes=nr_boxes,
            ).to(device=fc_feature.device)
            q_in = torch.cat([fc_feature, q_geo], dim=1)
            pred_quality = self.quality_head(q_in).squeeze(-1).view(b, nr_boxes)  # (B,N)

        return (
            class_logits.view(b, nr_boxes, -1),
            pred_bboxes.view(b, nr_boxes, -1),
            obj_features,  # (1, B*N, D)
            pred_quality,
        )


class DynamicHead(nn.Module):
    def __init__(
        self,
        *,
        num_proposals: int,
        num_classes: int,
        hidden_dim: int,
        dim_feedforward: int,
        nhead: int,
        dropout: float,
        activation: str,
        num_heads: int,
        deep_supervision: bool,
        roi_extractor: dict,
        pooler_resolution: int,
        diffusion_timesteps: int = 1000,
        dim_dynamic: int = 64,
        num_dynamic: int = 2,
        num_cls_layers: int = 1,
        num_reg_layers: int = 1,
        # geo bias / graph transformer
        use_geo_bias: bool = True,
        geo_bias_type: str = "mlp",
        geo_bias_scale: float = 1.0,
        capture_graph_attn: bool = False,
        # label state
        use_label_state: bool = True,
        label_state_scale: float = 0.1,
        # quality head (IoU prediction / energy guidance)
        use_quality_head: bool = False,
        quality_head_hidden_dim: int = 256,
        quality_head_include_t: bool = True,
        quality_head_use_log_wh: bool = True,
        quality_box_norm: float = 1000.0,
    ):
        super().__init__()
        self.num_proposals = int(num_proposals)
        self.num_heads = int(num_heads)
        self.return_intermediate = bool(deep_supervision)

        self.roi_extractor = MODELS.build(roi_extractor)

        rcnn_head = RCNNHead(
            num_classes=int(num_classes),
            d_model=int(hidden_dim),
            dim_feedforward=int(dim_feedforward),
            nhead=int(nhead),
            dropout=float(dropout),
            activation=str(activation),
            pooler_resolution=int(pooler_resolution),
            dim_dynamic=int(dim_dynamic),
            num_dynamic=int(num_dynamic),
            num_cls_layers=int(num_cls_layers),
            num_reg_layers=int(num_reg_layers),
            use_geo_bias=bool(use_geo_bias),
            geo_bias_type=str(geo_bias_type),
            geo_bias_scale=float(geo_bias_scale),
            diffusion_timesteps=int(diffusion_timesteps),
            use_label_state=bool(use_label_state),
            label_state_scale=float(label_state_scale),
            capture_graph_attn=bool(capture_graph_attn),
            use_quality_head=bool(use_quality_head),
            quality_head_hidden_dim=int(quality_head_hidden_dim),
            quality_head_include_t=bool(quality_head_include_t),
            quality_head_use_log_wh=bool(quality_head_use_log_wh),
            quality_box_norm=float(quality_box_norm),
        )
        self.head_series = _get_clones(rcnn_head, self.num_heads)

        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        time_dim = int(hidden_dim) * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.init_proposal_features = nn.Embedding(self.num_proposals, self.hidden_dim)
        nn.init.normal_(self.init_proposal_features.weight, std=0.02)

        self.last_attn_weights: torch.Tensor | None = None
        self.last_quality_feat: torch.Tensor | None = None

    def reset_parameters(self, *, use_sigmoid_cls: bool, prior_prob: float = 0.01) -> None:
        """
        Roughly aligns initialization with Detectron2 DiffusionDet:
        - Xavier init for main parameters
        - Optional: set classification bias to logit(prior_prob) for sigmoid-style classification
        - Force optional modules (geo bias / quality head / label state proj) to be strict no-ops at init
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        if use_sigmoid_cls:
            prior = float(prior_prob)
            prior = min(max(prior, 1e-6), 1.0 - 1e-6)
            bias_value = -math.log((1.0 - prior) / prior)
            # Only set classification bias. Do NOT apply this to bbox-regression bias
            # (important for small-class datasets where num_classes+1 may equal 4).
            for head in getattr(self, "head_series", []):
                if hasattr(head, "class_logits") and isinstance(head.class_logits, nn.Linear):
                    nn.init.constant_(head.class_logits.bias, bias_value)

        for head in getattr(self, "head_series", []):
            if hasattr(head, "reset_geo_bias_parameters"):
                head.reset_geo_bias_parameters()
            if hasattr(head, "reset_quality_head_parameters"):
                head.reset_quality_head_parameters()
            if hasattr(head, "reset_label_state_parameters"):
                head.reset_label_state_parameters()

    def forward(
        self,
        *,
        feats: list[torch.Tensor],
        init_bboxes: torch.Tensor,  # (B,N,4) abs xyxy
        timesteps: torch.Tensor,  # (B,) long
        label_state: Optional[torch.Tensor],
        image_sizes: Optional[torch.Tensor] = None,  # (B,2) int64 [H,W]
    ):
        bs = int(init_bboxes.shape[0])
        nr_boxes = int(init_bboxes.shape[1])
        if nr_boxes != self.num_proposals:
            raise ValueError(f"Expected init_bboxes to have N={self.num_proposals}, got {nr_boxes}")
        if image_sizes is not None and (
            image_sizes.dim() != 2 or image_sizes.shape[0] != bs or image_sizes.shape[1] != 2
        ):
            raise ValueError(f"Expected image_sizes (B,2), got {tuple(image_sizes.shape)} for B={bs}")

        t = timesteps.to(dtype=torch.float32)
        time_emb = self.time_mlp(t)

        proposal_features = self.init_proposal_features.weight[None, :, :].repeat(bs, 1, 1)
        bboxes = init_bboxes
        if image_sizes is not None:
            bboxes = _clip_boxes_to_image(bboxes, image_sizes, min_size=_DEFAULT_MIN_BOX_SIZE)

        inter_class_logits = []
        inter_pred_bboxes = []
        inter_pred_quality = []

        last_attn = None
        last_q_feat = None
        for rcnn_head in self.head_series:
            class_logits, pred_bboxes, proposal_features, pred_quality = rcnn_head(
                feats=feats,
                bboxes=bboxes,
                pro_features=proposal_features,
                roi_extractor=self.roi_extractor,
                time_emb=time_emb,
                diffusion_t=timesteps,
                label_state=label_state,
                image_sizes=image_sizes,
            )
            inter_class_logits.append(class_logits)
            inter_pred_bboxes.append(pred_bboxes)
            if pred_quality is not None:
                inter_pred_quality.append(pred_quality)
            bboxes = pred_bboxes.detach()
            if image_sizes is not None:
                bboxes = _clip_boxes_to_image(bboxes, image_sizes, min_size=_DEFAULT_MIN_BOX_SIZE)
            if rcnn_head.last_attn_weights is not None:
                last_attn = rcnn_head.last_attn_weights
            if getattr(rcnn_head, "last_quality_feat", None) is not None:
                last_q_feat = rcnn_head.last_quality_feat

        self.last_attn_weights = last_attn
        self.last_quality_feat = last_q_feat

        if self.return_intermediate:
            q = torch.stack(inter_pred_quality) if inter_pred_quality else None
            return torch.stack(inter_class_logits), torch.stack(inter_pred_bboxes), q
        q_last = inter_pred_quality[-1][None] if inter_pred_quality else None
        return inter_class_logits[-1][None], inter_pred_bboxes[-1][None], q_last


class GraphTransformerBlock(RCNNHead):
    """
    Alias for `RCNNHead`, named to match `check.md:4.3` "Graph Transformer Block".
    """


class GraphDenoisingNetwork(DynamicHead):
    """
    Alias for `DynamicHead`, named to match `check.md:4.*` "Graph Denoising Network (GDN)".
    """
