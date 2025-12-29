# ========================================
# Modified by Shoufa Chen
# ========================================
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DiffusionDet Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
import math

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes

from .util.sdpa_attention import SDPAMultiheadAttention


_DEFAULT_SCALE_CLAMP = math.log(100000.0 / 16)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)


class DynamicHead(nn.Module):

    def __init__(self, cfg, roi_input_shape):
        super().__init__()

        # Build RoI.
        box_pooler = self._init_box_pooler(cfg, roi_input_shape)
        self.box_pooler = box_pooler
        
        # Build heads.
        num_classes = cfg.MODEL.DiffusionDet.NUM_CLASSES
        d_model = cfg.MODEL.DiffusionDet.HIDDEN_DIM
        dim_feedforward = cfg.MODEL.DiffusionDet.DIM_FEEDFORWARD
        nhead = cfg.MODEL.DiffusionDet.NHEADS
        dropout = cfg.MODEL.DiffusionDet.DROPOUT
        activation = cfg.MODEL.DiffusionDet.ACTIVATION
        num_heads = cfg.MODEL.DiffusionDet.NUM_HEADS
        rcnn_head = RCNNHead(cfg, d_model, num_classes, dim_feedforward, nhead, dropout, activation)
        self.head_series = _get_clones(rcnn_head, num_heads)
        self.num_heads = num_heads
        self.return_intermediate = cfg.MODEL.DiffusionDet.DEEP_SUPERVISION
        # Phase 2: capture proposal self-attention for optional topology loss.
        self.last_attn_weights: torch.Tensor | None = None

        # Gaussian random feature embedding layer for time
        self.d_model = d_model
        time_dim = d_model * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(d_model),
            nn.Linear(d_model, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Init parameters.
        self.use_focal = cfg.MODEL.DiffusionDet.USE_FOCAL
        self.use_fed_loss = cfg.MODEL.DiffusionDet.USE_FED_LOSS
        self.num_classes = num_classes
        if self.use_focal or self.use_fed_loss:
            prior_prob = cfg.MODEL.DiffusionDet.PRIOR_PROB
            self.bias_value = -math.log((1 - prior_prob) / prior_prob)
        self._reset_parameters()

    def _reset_parameters(self):
        # Init parameters.
        #
        # Important: when optional modules (e.g. quality head / geo feat) are enabled, we want
        # the baseline detection parameters to keep the same initialization as much as possible.
        # Therefore, we first initialize "main" parameters, then initialize optional-module
        # parameters in a second pass so they don't shift the RNG consumption of the main params.
        has_optional_modules = False
        for head in getattr(self, "head_series", []):
            if (
                hasattr(head, "quality_head")
                or hasattr(head, "geo_bias_mlp")
                or hasattr(head, "geo_feat_abs")
                or hasattr(head, "geo_feat_rel_mlp")
                or hasattr(head, "label_state_embed")
            ):
                has_optional_modules = True
                break

        if not has_optional_modules:
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

                # initialize the bias for focal loss and fed loss.
                if self.use_focal or self.use_fed_loss:
                    if p.dim() == 1 and (p.numel() == self.num_classes or p.numel() == self.num_classes + 1):
                        nn.init.constant_(p, self.bias_value)
        else:
            # Initialize main parameters first, skipping optional-module weights so they don't
            # shift RNG consumption of the main params.
            for name, p in self.named_parameters(recurse=True):
                if p.dim() > 1:
                    if "quality_" in name or "geo_feat_" in name or "geo_bias_" in name or "label_state_" in name:
                        continue
                    nn.init.xavier_uniform_(p)

                # Initialize the bias for focal loss and fed loss.
                if self.use_focal or self.use_fed_loss:
                    if p.dim() == 1 and (p.numel() == self.num_classes or p.numel() == self.num_classes + 1):
                        nn.init.constant_(p, self.bias_value)

            # Initialize optional-module parameters without changing global RNG state,
            # so training-time randomness stays comparable to the baseline.
            _rng_state = torch.get_rng_state()
            try:
                for name, p in self.named_parameters(recurse=True):
                    if p.dim() > 1 and ("quality_" in name or "geo_feat_" in name or "geo_bias_" in name or "label_state_" in name):
                        nn.init.xavier_uniform_(p)
            finally:
                torch.set_rng_state(_rng_state)

        # Keep geometry feature injection as a strict no-op at initialization.
        for head in getattr(self, "head_series", []):
            if hasattr(head, "reset_geo_feat_parameters"):
                head.reset_geo_feat_parameters()
            if hasattr(head, "reset_geo_bias_parameters"):
                head.reset_geo_bias_parameters()
            if hasattr(head, "reset_quality_head_parameters"):
                head.reset_quality_head_parameters()
            if hasattr(head, "reset_label_state_parameters"):
                head.reset_label_state_parameters()

    @staticmethod
    def _init_box_pooler(cfg, input_shape):

        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        return box_pooler

    def forward(self, features, init_bboxes, t, init_features, label_state=None):
        # assert t shape (batch_size)
        time = self.time_mlp(t)

        inter_class_logits = []
        inter_pred_bboxes = []
        inter_quality_logits = []

        bs = len(features[0])
        bboxes = init_bboxes
        num_boxes = bboxes.shape[1]

        if init_features is not None:
            init_features = init_features[None].repeat(1, bs, 1)
            proposal_features = init_features.clone()
        else:
            proposal_features = None

        last_attn_weights = None
        for head_idx, rcnn_head in enumerate(self.head_series):
            class_logits, pred_bboxes, proposal_features, quality_logits = rcnn_head(
                features,
                bboxes,
                proposal_features,
                self.box_pooler,
                time,
                t,
                label_state=label_state,
            )
            last_attn_weights = getattr(rcnn_head, "last_attn_weights", None)
            if self.return_intermediate:
                inter_class_logits.append(class_logits)
                inter_pred_bboxes.append(pred_bboxes)
                if quality_logits is not None:
                    inter_quality_logits.append(quality_logits)
            bboxes = pred_bboxes.detach()

        self.last_attn_weights = last_attn_weights
        if self.return_intermediate:
            quality = torch.stack(inter_quality_logits) if len(inter_quality_logits) > 0 else None
            return torch.stack(inter_class_logits), torch.stack(inter_pred_bboxes), quality

        return class_logits[None], pred_bboxes[None], quality_logits[None] if quality_logits is not None else None


class RCNNHead(nn.Module):

    def __init__(self, cfg, d_model, num_classes, dim_feedforward=2048, nhead=8, dropout=0.1, activation="relu",
                 scale_clamp: float = _DEFAULT_SCALE_CLAMP, bbox_weights=(2.0, 2.0, 1.0, 1.0)):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.use_geo_bias = bool(getattr(cfg.MODEL.DiffusionDet, "GEO_BIAS", False))
        self.geo_bias_type = str(getattr(cfg.MODEL.DiffusionDet, "GEO_BIAS_TYPE", "distance")).lower()
        self.geo_bias_schedule = str(getattr(cfg.MODEL.DiffusionDet, "GEO_BIAS_SCHEDULE", "constant")).lower()
        self.geo_bias_time_power = float(getattr(cfg.MODEL.DiffusionDet, "GEO_BIAS_TIME_POWER", 1.0))
        self.geo_bias_t_threshold = int(getattr(cfg.MODEL.DiffusionDet, "GEO_BIAS_T_THRESHOLD", 0))
        self.diffusion_timesteps = int(getattr(cfg.MODEL.DiffusionDet, "DIFFUSION_TIMESTEPS", 1000))
        self.geo_bias_learnable_scale = bool(getattr(cfg.MODEL.DiffusionDet, "GEO_BIAS_LEARNABLE_SCALE", False))
        self.geo_bias_scale = float(getattr(cfg.MODEL.DiffusionDet, "GEO_BIAS_SCALE", 1.0))
        self.geo_bias_sigma = float(getattr(cfg.MODEL.DiffusionDet, "GEO_BIAS_SIGMA", 2.0))
        self.geo_bias_min_norm = float(getattr(cfg.MODEL.DiffusionDet, "GEO_BIAS_MIN_NORM", 0.0))
        self.geo_bias_norm = float(getattr(cfg.MODEL.DiffusionDet, "GEO_BIAS_NORM", 1000.0))
        self.geo_bias_use_log_wh = bool(getattr(cfg.MODEL.DiffusionDet, "GEO_BIAS_USE_LOG_WH", True))
        self.geo_bias_input_clip = float(getattr(cfg.MODEL.DiffusionDet, "GEO_BIAS_INPUT_CLIP", 0.0))
        self.geo_bias_rel_include_iou = bool(getattr(cfg.MODEL.DiffusionDet, "GEO_BIAS_REL_INCLUDE_IOU", False))
        self.geo_bias_mlp_hidden_dim = int(getattr(cfg.MODEL.DiffusionDet, "GEO_BIAS_MLP_HIDDEN_DIM", 64))
        self.geo_bias_out_tanh = bool(getattr(cfg.MODEL.DiffusionDet, "GEO_BIAS_OUT_TANH", True))
        self.geo_bias_topk = int(getattr(cfg.MODEL.DiffusionDet, "GEO_BIAS_TOPK", 0))
        if self.use_geo_bias and self.geo_bias_learnable_scale:
            self.geo_bias_scale_param = nn.Parameter(torch.zeros(self.nhead))

        if self.use_geo_bias and self.geo_bias_type == "mlp":
            rel_in_dim = 4 + (1 if self.geo_bias_rel_include_iou else 0)
            hidden = max(self.geo_bias_mlp_hidden_dim, 1)
            _rng_state = torch.get_rng_state()
            try:
                self.geo_bias_mlp = nn.Sequential(
                    nn.Linear(rel_in_dim, hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, 1),
                )
            finally:
                torch.set_rng_state(_rng_state)

        self.use_geo_feat = bool(getattr(cfg.MODEL.DiffusionDet, "GEO_FEAT", False))
        self.geo_feat_encoder = str(getattr(cfg.MODEL.DiffusionDet, "GEO_FEAT_ENCODER", "mlp")).lower()
        self.geo_feat_scale = float(getattr(cfg.MODEL.DiffusionDet, "GEO_FEAT_SCALE", 0.1))
        self.geo_feat_alpha_init = float(getattr(cfg.MODEL.DiffusionDet, "GEO_FEAT_ALPHA_INIT", 0.1))
        self.geo_feat_norm = float(getattr(cfg.MODEL.DiffusionDet, "GEO_FEAT_NORM", 1000.0))
        self.geo_feat_use_log_wh = bool(getattr(cfg.MODEL.DiffusionDet, "GEO_FEAT_USE_LOG_WH", True))
        self.geo_feat_include_t = bool(getattr(cfg.MODEL.DiffusionDet, "GEO_FEAT_INCLUDE_T", True))
        self.geo_feat_target = str(getattr(cfg.MODEL.DiffusionDet, "GEO_FEAT_TARGET", "proposal")).lower()
        self.geo_feat_proposal_mode = str(
            getattr(cfg.MODEL.DiffusionDet, "GEO_FEAT_PROPOSAL_MODE", "qk_pos")
        ).lower()
        self.geo_feat_train_start_iter = int(getattr(cfg.MODEL.DiffusionDet, "GEO_FEAT_TRAIN_START_ITER", 0))
        self.geo_feat_train_warmup_iters = int(getattr(cfg.MODEL.DiffusionDet, "GEO_FEAT_TRAIN_WARMUP_ITERS", 0))
        self.geo_feat_mlp_hidden_dim = int(getattr(cfg.MODEL.DiffusionDet, "GEO_FEAT_MLP_HIDDEN_DIM", 128))
        self.geo_feat_rel = bool(getattr(cfg.MODEL.DiffusionDet, "GEO_FEAT_REL", False))
        self.geo_feat_rel_mlp_hidden_dim = int(getattr(cfg.MODEL.DiffusionDet, "GEO_FEAT_REL_MLP_HIDDEN_DIM", 64))
        self.geo_feat_rel_include_iou = bool(getattr(cfg.MODEL.DiffusionDet, "GEO_FEAT_REL_INCLUDE_IOU", False))
        self.geo_feat_rel_tanh = bool(getattr(cfg.MODEL.DiffusionDet, "GEO_FEAT_REL_TANH", True))
        self.geo_feat_input_clip = float(getattr(cfg.MODEL.DiffusionDet, "GEO_FEAT_INPUT_CLIP", 0.0))
        self.geo_feat_out_tanh = bool(getattr(cfg.MODEL.DiffusionDet, "GEO_FEAT_OUT_TANH", False))
        self.geo_feat_rel_norm = str(getattr(cfg.MODEL.DiffusionDet, "GEO_FEAT_REL_NORM", "none")).lower()
        self.geo_feat_schedule = str(getattr(cfg.MODEL.DiffusionDet, "GEO_FEAT_SCHEDULE", "constant")).lower()
        self.geo_feat_time_power = float(getattr(cfg.MODEL.DiffusionDet, "GEO_FEAT_TIME_POWER", 1.0))
        self.geo_feat_t_threshold = int(getattr(cfg.MODEL.DiffusionDet, "GEO_FEAT_T_THRESHOLD", 0))
        if self.use_geo_feat and self.geo_feat_target not in {"proposal", "reg"}:
            raise ValueError(f"Unsupported GEO_FEAT_TARGET={self.geo_feat_target!r}")
        if self.use_geo_feat and self.geo_feat_target == "proposal" and self.geo_feat_proposal_mode not in {"add", "qk_pos"}:
            raise ValueError(f"Unsupported GEO_FEAT_PROPOSAL_MODE={self.geo_feat_proposal_mode!r}")
        if self.use_geo_feat and self.geo_feat_encoder not in {"linear", "mlp"}:
            raise ValueError(f"Unsupported GEO_FEAT_ENCODER={self.geo_feat_encoder!r}")
        if self.use_geo_feat and self.geo_feat_rel and self.geo_feat_rel_norm not in {"softmax", "mean", "none"}:
            raise ValueError(f"Unsupported GEO_FEAT_REL_NORM={self.geo_feat_rel_norm!r}")

        # Phase 2(A): label-state injection (discrete state in {0..K-1, unk}).
        self.use_label_state = bool(getattr(cfg.MODEL.DiffusionDet, "LABEL_STATE", False))
        self.label_state_scale = float(getattr(cfg.MODEL.DiffusionDet, "LABEL_STATE_SCALE", 0.1))
        self.label_state_alpha_init = float(getattr(cfg.MODEL.DiffusionDet, "LABEL_STATE_ALPHA_INIT", 0.1))
        self.label_state_proj_zero_init = bool(getattr(cfg.MODEL.DiffusionDet, "LABEL_STATE_PROJ_ZERO_INIT", True))
        self.label_state_relative_to_unk = bool(
            getattr(cfg.MODEL.DiffusionDet, "LABEL_STATE_RELATIVE_TO_UNK", False)
        )

        # Phase 2: optional topology supervision on the proposal graph.
        self.capture_graph_attn = float(getattr(cfg.MODEL.DiffusionDet, "GRAPH_TOPO_LOSS_WEIGHT", 0.0)) > 0.0
        self.last_attn_weights: torch.Tensor | None = None
        self.disable_self_attn = bool(getattr(cfg.MODEL.DiffusionDet, "DISABLE_SELF_ATTN", False))

        self.use_quality_head = bool(getattr(cfg.MODEL.DiffusionDet, "QUALITY_HEAD", False))
        self.quality_head_hidden_dim = int(getattr(cfg.MODEL.DiffusionDet, "QUALITY_HEAD_HIDDEN_DIM", 256))
        self.quality_head_include_t = bool(getattr(cfg.MODEL.DiffusionDet, "QUALITY_HEAD_INCLUDE_T", True))
        self.quality_head_use_log_wh = bool(getattr(cfg.MODEL.DiffusionDet, "QUALITY_HEAD_USE_LOG_WH", True))
        self.quality_box_norm = float(getattr(cfg.MODEL.DiffusionDet, "QUALITY_BOX_NORM", 1000.0))

        # dynamic.
        self.self_attn_impl = str(getattr(cfg.MODEL.DiffusionDet, "SELF_ATTN_IMPL", "torch")).lower()
        self.self_attn_sdpa_backend = str(
            getattr(cfg.MODEL.DiffusionDet, "SELF_ATTN_SDPA_BACKEND", "auto")
        ).lower()
        if self.self_attn_impl not in {"torch", "sdpa"}:
            raise ValueError(f"Unsupported SELF_ATTN_IMPL={self.self_attn_impl!r}")
        if self.self_attn_sdpa_backend not in {"auto", "flash", "mem_efficient", "math"}:
            raise ValueError(f"Unsupported SELF_ATTN_SDPA_BACKEND={self.self_attn_sdpa_backend!r}")

        # NOTE: graph topology loss needs attention weights; SDPA path does not return weights.
        if self.self_attn_impl == "sdpa" and not self.capture_graph_attn:
            self.self_attn = SDPAMultiheadAttention(
                d_model,
                nhead,
                dropout=dropout,
                backend=self.self_attn_sdpa_backend,  # type: ignore[arg-type]
            )
        else:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.self_attn_impl = "torch"
        self.inst_interact = DynamicConv(cfg)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        # block time mlp
        self.block_time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(d_model * 4, d_model * 2))

        # cls.
        num_cls = cfg.MODEL.DiffusionDet.NUM_CLS
        cls_module = list()
        for _ in range(num_cls):
            cls_module.append(nn.Linear(d_model, d_model, False))
            cls_module.append(nn.LayerNorm(d_model))
            cls_module.append(nn.ReLU(inplace=True))
        self.cls_module = nn.ModuleList(cls_module)

        # reg.
        num_reg = cfg.MODEL.DiffusionDet.NUM_REG
        reg_module = list()
        for _ in range(num_reg):
            reg_module.append(nn.Linear(d_model, d_model, False))
            reg_module.append(nn.LayerNorm(d_model))
            reg_module.append(nn.ReLU(inplace=True))
        self.reg_module = nn.ModuleList(reg_module)
        
        # pred.
        self.use_focal = cfg.MODEL.DiffusionDet.USE_FOCAL
        self.use_fed_loss = cfg.MODEL.DiffusionDet.USE_FED_LOSS
        if self.use_focal or self.use_fed_loss:
            self.class_logits = nn.Linear(d_model, num_classes)
        else:
            self.class_logits = nn.Linear(d_model, num_classes + 1)
        self.bboxes_delta = nn.Linear(d_model, 4)
        self.scale_clamp = scale_clamp
        self.bbox_weights = bbox_weights

        if self.use_quality_head:
            q_geo_dim = 4 + (1 if self.quality_head_include_t else 0)
            q_hidden = max(self.quality_head_hidden_dim, 1)
            # Avoid changing global RNG state when enabling optional modules.
            # This helps keep baseline training randomness comparable.
            _rng_state = torch.get_rng_state()
            try:
                self.quality_head = nn.Sequential(
                    nn.Linear(d_model + q_geo_dim, q_hidden),
                    nn.SiLU(),
                    nn.Linear(q_hidden, 1),
                )
            finally:
                torch.set_rng_state(_rng_state)

        if self.use_geo_feat:
            alpha = min(max(self.geo_feat_alpha_init, 1e-4), 1.0 - 1e-4)
            self.geo_feat_alpha_param = nn.Parameter(torch.tensor(math.log(alpha / (1.0 - alpha))))

            geo_in_dim = 4 + (1 if self.geo_feat_include_t else 0)
            if self.geo_feat_encoder == "linear":
                self.geo_feat_abs = nn.Linear(geo_in_dim, d_model)
            else:
                hidden = max(self.geo_feat_mlp_hidden_dim, 1)
                self.geo_feat_abs = nn.Sequential(
                    nn.Linear(geo_in_dim, hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, d_model),
                )

            if self.geo_feat_rel:
                rel_in_dim = 4 + (1 if self.geo_feat_rel_include_iou else 0)
                rel_hidden = max(self.geo_feat_rel_mlp_hidden_dim, 1)
                self.geo_feat_rel_mlp = nn.Sequential(
                    nn.Linear(rel_in_dim, rel_hidden),
                    nn.SiLU(),
                    nn.Linear(rel_hidden, 1),
                )

        if self.use_label_state:
            alpha = min(max(self.label_state_alpha_init, 1e-4), 1.0 - 1e-4)
            self.label_state_alpha_param = nn.Parameter(torch.tensor(math.log(alpha / (1.0 - alpha))))
            # Avoid changing global RNG state when enabling optional modules.
            _rng_state = torch.get_rng_state()
            try:
                self.label_state_embed = nn.Embedding(num_classes + 1, d_model)
                self.label_state_proj = nn.Linear(d_model, d_model)
            finally:
                torch.set_rng_state(_rng_state)

    def reset_quality_head_parameters(self) -> None:
        if not hasattr(self, "quality_head"):
            return
        last = self.quality_head[-1]
        if isinstance(last, nn.Linear):
            nn.init.constant_(last.weight, 0.0)
            if last.bias is not None:
                nn.init.constant_(last.bias, 0.0)

    def reset_geo_feat_parameters(self) -> None:
        if hasattr(self, "geo_feat_abs"):
            if isinstance(self.geo_feat_abs, nn.Linear):
                nn.init.constant_(self.geo_feat_abs.weight, 0.0)
                if self.geo_feat_abs.bias is not None:
                    nn.init.constant_(self.geo_feat_abs.bias, 0.0)
            else:
                last = self.geo_feat_abs[-1]
                if isinstance(last, nn.Linear):
                    nn.init.constant_(last.weight, 0.0)
                    if last.bias is not None:
                        nn.init.constant_(last.bias, 0.0)

        if hasattr(self, "geo_feat_rel_mlp"):
            last = self.geo_feat_rel_mlp[-1]
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

    def _geo_feat_alpha(self, dtype: torch.dtype) -> torch.Tensor:
        if not hasattr(self, "geo_feat_alpha_param"):
            return torch.tensor(1.0, dtype=dtype)
        return torch.sigmoid(self.geo_feat_alpha_param).to(dtype=dtype)

    def _label_state_alpha(self, dtype: torch.dtype) -> torch.Tensor:
        if not hasattr(self, "label_state_alpha_param"):
            return torch.tensor(1.0, dtype=dtype)
        return torch.sigmoid(self.label_state_alpha_param).to(dtype=dtype)

    def _build_quality_box_features(
        self,
        boxes_xyxy: torch.Tensor,
        dtype: torch.dtype,
        diffusion_t: torch.Tensor | None,
        nr_boxes: int,
    ) -> torch.Tensor:
        x1, y1, x2, y2 = boxes_xyxy.unbind(dim=-1)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        w = (x2 - x1).clamp(min=1e-6)
        h = (y2 - y1).clamp(min=1e-6)

        norm = max(self.quality_box_norm, 1e-6)
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
                raise ValueError("diffusion_t is required when QUALITY_HEAD_INCLUDE_T is True")
            denom = float(max(self.diffusion_timesteps - 1, 1))
            t = diffusion_t.to(dtype=torch.float32) / denom
            t = t.clamp(0.0, 1.0)
            t = torch.repeat_interleave(t, nr_boxes, dim=0).to(dtype=cx.dtype)
            feats.append(t)

        return torch.stack(feats, dim=-1).to(dtype=dtype)

    def _build_geometry_attn_bias(
        self,
        bboxes: torch.Tensor,
        dtype: torch.dtype,
        diffusion_t: torch.Tensor | None,
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

        if self.geo_bias_type == "iou":
            # IoU-based bias in [0, 1]. We set diagonal to 0 to avoid over-boosting self-attention.
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
            nr_boxes = iou.shape[-1]
            eye = torch.eye(nr_boxes, device=iou.device, dtype=torch.bool)
            iou = iou.masked_fill(eye[None, :, :], 0.0)

            bias = iou
        elif self.geo_bias_type == "distance":
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
        topk = int(self.geo_bias_topk) if hasattr(self, "geo_bias_topk") else 0
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
            bias_h = bias * self.geo_bias_scale
            bias_h = bias_h.unsqueeze(1).repeat(1, self.nhead, 1, 1)

        if keep is not None:
            bias_h = bias_h.masked_fill(~keep[:, None, :, :], -1.0e4)

        return bias_h.reshape(n * self.nhead, nr_boxes, nr_boxes)

    def _build_geometry_feature_delta(
        self,
        bboxes: torch.Tensor,
        dtype: torch.dtype,
        diffusion_t: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if not self.use_geo_feat:
            return None
        if self.geo_feat_scale == 0.0:
            return None
        if self.training and self.geo_feat_train_start_iter > 0:
            try:
                from detectron2.utils.events import get_event_storage

                current_iter = get_event_storage().iter
            except Exception:
                current_iter = None
            if current_iter is not None and current_iter < int(self.geo_feat_train_start_iter):
                return None

        time_scale = None
        if self.geo_feat_schedule == "constant":
            time_scale = None
        elif self.geo_feat_schedule == "linear":
            if diffusion_t is None:
                raise ValueError("diffusion_t is required when GEO_FEAT_SCHEDULE == 'linear'")
            denom = float(max(self.diffusion_timesteps - 1, 1))
            t = diffusion_t.detach().to(dtype=torch.float32)
            frac = (t / denom).clamp(0.0, 1.0)
            time_scale = (1.0 - frac).clamp(0.0, 1.0)
            if self.geo_feat_time_power != 1.0:
                time_scale = time_scale ** float(self.geo_feat_time_power)
        elif self.geo_feat_schedule == "threshold":
            if diffusion_t is None:
                raise ValueError("diffusion_t is required when GEO_FEAT_SCHEDULE == 'threshold'")
            t = diffusion_t.detach()
            keep = t <= int(self.geo_feat_t_threshold)
            if not torch.any(keep):
                return None
            time_scale = keep.to(dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported GEO_FEAT_SCHEDULE={self.geo_feat_schedule!r}")

        boxes = bboxes.detach()
        x1, y1, x2, y2 = boxes.unbind(dim=-1)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        w = (x2 - x1).clamp(min=1e-6)
        h = (y2 - y1).clamp(min=1e-6)

        norm = max(self.geo_feat_norm, 1e-6)
        cx = cx / norm
        cy = cy / norm
        w = w / norm
        h = h / norm
        if self.geo_feat_use_log_wh:
            w = torch.log(w + 1e-6)
            h = torch.log(h + 1e-6)

        feats = [cx, cy, w, h]
        if self.geo_feat_include_t:
            if diffusion_t is None:
                raise ValueError("diffusion_t is required when GEO_FEAT_INCLUDE_T is True")
            denom = float(max(self.diffusion_timesteps - 1, 1))
            t = diffusion_t.detach().to(dtype=torch.float32) / denom
            t = t.clamp(0.0, 1.0)
            t = t[:, None].expand_as(cx)
            feats.append(t)

        geo = torch.stack(feats, dim=-1).to(dtype=torch.float32)
        if self.geo_feat_input_clip > 0.0:
            clip = float(self.geo_feat_input_clip)
            geo = geo.clamp(min=-clip, max=clip)
        delta = self.geo_feat_abs(geo)
        if self.geo_feat_out_tanh:
            delta = torch.tanh(delta)
        scale = float(self.geo_feat_scale) * self._geo_feat_alpha(dtype=torch.float32)
        delta = (delta * scale).to(dtype=dtype)

        if time_scale is not None:
            delta = delta * time_scale.to(dtype=dtype)[:, None, None]

        if self.geo_feat_train_warmup_iters > 0:
            try:
                from detectron2.utils.events import get_event_storage
                current_iter = get_event_storage().iter
            except Exception:
                current_iter = None
            if current_iter is not None:
                start = int(self.geo_feat_train_start_iter) if self.training else 0
                if start > 0:
                    warm = float(current_iter - start) / float(self.geo_feat_train_warmup_iters)
                else:
                    warm = float(current_iter) / float(self.geo_feat_train_warmup_iters)
                warm = max(0.0, min(warm, 1.0))
                delta = delta * warm
        return delta

    def _build_relative_geometry_feature_delta(
        self,
        pro_features: torch.Tensor,
        bboxes: torch.Tensor,
        dtype: torch.dtype,
        diffusion_t: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if not (self.use_geo_feat and self.geo_feat_rel):
            return None
        if self.geo_feat_scale == 0.0:
            return None
        if self.training and self.geo_feat_train_start_iter > 0:
            try:
                from detectron2.utils.events import get_event_storage

                current_iter = get_event_storage().iter
            except Exception:
                current_iter = None
            if current_iter is not None and current_iter < int(self.geo_feat_train_start_iter):
                return None

        time_scale = None
        if self.geo_feat_schedule == "constant":
            time_scale = None
        elif self.geo_feat_schedule == "linear":
            if diffusion_t is None:
                raise ValueError("diffusion_t is required when GEO_FEAT_SCHEDULE == 'linear'")
            denom = float(max(self.diffusion_timesteps - 1, 1))
            t = diffusion_t.detach().to(dtype=torch.float32)
            frac = (t / denom).clamp(0.0, 1.0)
            time_scale = (1.0 - frac).clamp(0.0, 1.0)
            if self.geo_feat_time_power != 1.0:
                time_scale = time_scale ** float(self.geo_feat_time_power)
        elif self.geo_feat_schedule == "threshold":
            if diffusion_t is None:
                raise ValueError("diffusion_t is required when GEO_FEAT_SCHEDULE == 'threshold'")
            t = diffusion_t.detach()
            keep = t <= int(self.geo_feat_t_threshold)
            if not torch.any(keep):
                return None
            time_scale = keep.to(dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported GEO_FEAT_SCHEDULE={self.geo_feat_schedule!r}")

        boxes = bboxes.detach()
        x1, y1, x2, y2 = boxes.unbind(dim=-1)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        w = (x2 - x1).clamp(min=1e-6)
        h = (y2 - y1).clamp(min=1e-6)

        norm = max(self.geo_feat_norm, 1e-6)
        cx = cx / norm
        cy = cy / norm
        w = w / norm
        h = h / norm
        if self.geo_feat_use_log_wh:
            w = torch.log(w + 1e-6)
            h = torch.log(h + 1e-6)

        dx = (cx[:, :, None] - cx[:, None, :]).to(dtype=torch.float32)
        dy = (cy[:, :, None] - cy[:, None, :]).to(dtype=torch.float32)
        dw = (w[:, :, None] - w[:, None, :]).to(dtype=torch.float32)
        dh = (h[:, :, None] - h[:, None, :]).to(dtype=torch.float32)
        rel_feats = [dx, dy, dw, dh]

        if self.geo_feat_rel_include_iou:
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
        if self.geo_feat_input_clip > 0.0:
            clip = float(self.geo_feat_input_clip)
            rel = rel.clamp(min=-clip, max=clip)
        w_ij = self.geo_feat_rel_mlp(rel).squeeze(-1)
        if self.geo_feat_rel_tanh:
            w_ij = torch.tanh(w_ij)

        nr_boxes = w_ij.shape[-1]
        if nr_boxes <= 1:
            return torch.zeros_like(pro_features)
        eye = torch.eye(nr_boxes, device=w_ij.device, dtype=torch.bool)
        mask = eye[None, :, :]
        norm = self.geo_feat_rel_norm
        if norm == "softmax":
            # Use a normalized (convex) message to keep magnitude stable with NUM_PROPOSALS=500.
            w = w_ij.masked_fill(mask, -1.0e4)
            attn = torch.softmax(w.to(dtype=torch.float32), dim=-1)
            msg = torch.bmm(attn, pro_features.to(dtype=torch.float32))
        elif norm == "mean":
            w = w_ij.masked_fill(mask, 0.0)
            msg = torch.bmm(w.to(dtype=torch.float32), pro_features.to(dtype=torch.float32))
            msg = msg / float(max(nr_boxes - 1, 1))
        else:
            w = w_ij.masked_fill(mask, 0.0)
            msg = torch.bmm(w.to(dtype=torch.float32), pro_features.to(dtype=torch.float32))

        scale = float(self.geo_feat_scale) * self._geo_feat_alpha(dtype=torch.float32)
        delta = (msg * scale).to(dtype=dtype)

        if time_scale is not None:
            delta = delta * time_scale.to(dtype=dtype)[:, None, None]

        if self.geo_feat_train_warmup_iters > 0:
            try:
                from detectron2.utils.events import get_event_storage
                current_iter = get_event_storage().iter
            except Exception:
                current_iter = None
            if current_iter is not None:
                start = int(self.geo_feat_train_start_iter) if self.training else 0
                if start > 0:
                    warm = float(current_iter - start) / float(self.geo_feat_train_warmup_iters)
                else:
                    warm = float(current_iter) / float(self.geo_feat_train_warmup_iters)
                warm = max(0.0, min(warm, 1.0))
                delta = delta * warm
        return delta

    def forward(self, features, bboxes, pro_features, pooler, time_emb, diffusion_t=None, label_state=None):
        """
        :param bboxes: (N, nr_boxes, 4)
        :param pro_features: (N, nr_boxes, d_model)
        """

        N, nr_boxes = bboxes.shape[:2]
        
        # roi_feature.
        proposal_boxes = list()
        for b in range(N):
            proposal_boxes.append(Boxes(bboxes[b]))
        roi_features = pooler(features, proposal_boxes)

        if pro_features is None:
            pro_features = roi_features.view(N, nr_boxes, self.d_model, -1).mean(-1)
        else:
            pro_features = pro_features.view(N, nr_boxes, self.d_model)

        if self.use_label_state and label_state is not None:
            # label_state:
            # - hard ids: (N, nr_boxes) long in [0..K-1, unk(K)]
            # - soft distribution: (N, nr_boxes, K+1) float, rows sum to 1
            if label_state.dim() == 2:
                emb = self.label_state_embed(label_state.to(device=pro_features.device, dtype=torch.long))
            elif label_state.dim() == 3:
                probs = label_state.to(device=pro_features.device, dtype=torch.float32)
                weight = self.label_state_embed.weight.to(dtype=torch.float32)
                if probs.shape[-1] != weight.shape[0]:
                    raise ValueError(
                        f"label_state last dim {probs.shape[-1]} must match embedding states {weight.shape[0]}"
                    )
                # Discrete diffusion states are categorical. During training, sampling a discrete state
                # is closer to the true forward process and empirically avoids "soft label leakage".
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

        pro_pos = None
        if self.use_geo_feat and self.geo_feat_target == "proposal":
            pro_features = pro_features.view(N, nr_boxes, self.d_model)
            geo_delta = self._build_geometry_feature_delta(bboxes, dtype=pro_features.dtype, diffusion_t=diffusion_t)
            rel_delta = self._build_relative_geometry_feature_delta(
                pro_features,
                bboxes,
                dtype=pro_features.dtype,
                diffusion_t=diffusion_t,
            )
            if self.geo_feat_proposal_mode == "add":
                if geo_delta is not None:
                    pro_features = pro_features + geo_delta
                if rel_delta is not None:
                    pro_features = pro_features + rel_delta
            else:
                if geo_delta is not None:
                    pro_pos = geo_delta
                if rel_delta is not None:
                    pro_pos = rel_delta if pro_pos is None else (pro_pos + rel_delta)

        roi_features = roi_features.view(N * nr_boxes, self.d_model, -1).permute(2, 0, 1)

        # self_att.
        pro_features = pro_features.view(N, nr_boxes, self.d_model).permute(1, 0, 2)
        if pro_pos is not None:
            pro_pos = pro_pos.view(N, nr_boxes, self.d_model).permute(1, 0, 2).to(dtype=pro_features.dtype)
        attn_bias = self._build_geometry_attn_bias(bboxes, dtype=pro_features.dtype, diffusion_t=diffusion_t)
        q = pro_features if pro_pos is None else (pro_features + pro_pos)
        k = pro_features if pro_pos is None else (pro_features + pro_pos)
        if self.disable_self_attn:
            pro_features2 = torch.zeros_like(pro_features)
            self.last_attn_weights = None
        elif self.capture_graph_attn:
            pro_features2, attn_weights = self.self_attn(
                q,
                k,
                value=pro_features,
                attn_mask=attn_bias,
                need_weights=True,
            )
            # (batch, nr_boxes, nr_boxes) average-attention weights.
            self.last_attn_weights = attn_weights
        else:
            pro_features2 = self.self_attn(q, k, value=pro_features, attn_mask=attn_bias, need_weights=False)[0]
            self.last_attn_weights = None
        pro_features = pro_features + self.dropout1(pro_features2)
        pro_features = self.norm1(pro_features)

        # inst_interact.
        pro_features = pro_features.view(nr_boxes, N, self.d_model).permute(1, 0, 2).reshape(1, N * nr_boxes, self.d_model)
        pro_features2 = self.inst_interact(pro_features, roi_features)
        pro_features = pro_features + self.dropout2(pro_features2)
        obj_features = self.norm2(pro_features)

        # obj_feature.
        obj_features2 = self.linear2(self.dropout(self.activation(self.linear1(obj_features))))
        obj_features = obj_features + self.dropout3(obj_features2)
        obj_features = self.norm3(obj_features)
        
        fc_feature = obj_features.transpose(0, 1).reshape(N * nr_boxes, -1)

        scale_shift = self.block_time_mlp(time_emb)
        scale_shift = torch.repeat_interleave(scale_shift, nr_boxes, dim=0)
        scale, shift = scale_shift.chunk(2, dim=1)
        fc_feature = fc_feature * (scale + 1) + shift

        cls_feature = fc_feature.clone()
        reg_feature = fc_feature.clone()
        if self.use_geo_feat and self.geo_feat_target == "reg":
            geo_delta = self._build_geometry_feature_delta(bboxes, dtype=reg_feature.dtype, diffusion_t=diffusion_t)
            if geo_delta is not None:
                reg_feature = reg_feature + geo_delta.reshape(N * nr_boxes, self.d_model)
        for cls_layer in self.cls_module:
            cls_feature = cls_layer(cls_feature)
        for reg_layer in self.reg_module:
            reg_feature = reg_layer(reg_feature)
        class_logits = self.class_logits(cls_feature)
        bboxes_deltas = self.bboxes_delta(reg_feature)
        pred_bboxes = self.apply_deltas(bboxes_deltas, bboxes.view(-1, 4))

        quality_logits = None
        if self.use_quality_head:
            q_boxes = pred_bboxes.detach() if self.training else pred_bboxes
            q_feat = fc_feature.detach() if self.training else fc_feature
            q_geo = self._build_quality_box_features(
                q_boxes,
                dtype=fc_feature.dtype,
                diffusion_t=diffusion_t,
                nr_boxes=nr_boxes,
            )
            q_in = torch.cat([q_feat, q_geo], dim=1)
            quality_logits = self.quality_head(q_in).squeeze(-1).view(N, nr_boxes)
        
        return class_logits.view(N, nr_boxes, -1), pred_bboxes.view(N, nr_boxes, -1), obj_features, quality_logits

    def apply_deltas(self, deltas, boxes):
        """
        Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """
        boxes = boxes.to(deltas.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.bbox_weights
        dx = deltas[:, 0::4] / wx
        dy = deltas[:, 1::4] / wy
        dw = deltas[:, 2::4] / ww
        dh = deltas[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w  # x1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h  # y1
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w  # x2
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h  # y2

        return pred_boxes


class DynamicConv(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.hidden_dim = cfg.MODEL.DiffusionDet.HIDDEN_DIM
        self.dim_dynamic = cfg.MODEL.DiffusionDet.DIM_DYNAMIC
        self.num_dynamic = cfg.MODEL.DiffusionDet.NUM_DYNAMIC
        self.num_params = self.hidden_dim * self.dim_dynamic
        self.dynamic_layer = nn.Linear(self.hidden_dim, self.num_dynamic * self.num_params)

        self.norm1 = nn.LayerNorm(self.dim_dynamic)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        self.activation = nn.ReLU(inplace=True)

        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        num_output = self.hidden_dim * pooler_resolution ** 2
        self.out_layer = nn.Linear(num_output, self.hidden_dim)
        self.norm3 = nn.LayerNorm(self.hidden_dim)

    def forward(self, pro_features, roi_features):
        '''
        pro_features: (1,  N * nr_boxes, self.d_model)
        roi_features: (49, N * nr_boxes, self.d_model)
        '''
        features = roi_features.permute(1, 0, 2)
        parameters = self.dynamic_layer(pro_features).permute(1, 0, 2)

        param1 = parameters[:, :, :self.num_params].view(-1, self.hidden_dim, self.dim_dynamic)
        param2 = parameters[:, :, self.num_params:].view(-1, self.dim_dynamic, self.hidden_dim)

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


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
