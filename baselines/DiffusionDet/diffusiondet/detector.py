# ========================================
# Modified by Shoufa Chen
# ========================================
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
import random
from typing import List
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import batched_nms
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess

from detectron2.structures import Boxes, ImageList, Instances

from .loss import SetCriterionDynamicK, HungarianMatcherDynamicK
from .head import DynamicHead
from .util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from .util.misc import nested_tensor_from_tensor_list

__all__ = ["DiffusionDet"]

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


@META_ARCH_REGISTRY.register()
class DiffusionDet(nn.Module):
    """
    Implement DiffusionDet
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes = cfg.MODEL.DiffusionDet.NUM_CLASSES
        self.num_proposals = cfg.MODEL.DiffusionDet.NUM_PROPOSALS
        self.hidden_dim = cfg.MODEL.DiffusionDet.HIDDEN_DIM
        self.num_heads = cfg.MODEL.DiffusionDet.NUM_HEADS
        self.use_quality_head = bool(getattr(cfg.MODEL.DiffusionDet, "QUALITY_HEAD", False))
        self.quality_guidance_scale = float(getattr(cfg.MODEL.DiffusionDet, "QUALITY_GUIDANCE_SCALE", 0.0))
        self.quality_guidance_topk = int(getattr(cfg.MODEL.DiffusionDet, "QUALITY_GUIDANCE_TOPK", 50))
        self.quality_guidance_score_weight = bool(getattr(cfg.MODEL.DiffusionDet, "QUALITY_GUIDANCE_SCORE_WEIGHT", True))
        self.quality_guidance_grad_norm = str(
            getattr(cfg.MODEL.DiffusionDet, "QUALITY_GUIDANCE_GRAD_NORM", "proposal")
        ).lower()
        self.quality_guidance_mode = str(getattr(cfg.MODEL.DiffusionDet, "QUALITY_GUIDANCE_MODE", "final")).lower()
        self.quality_guidance_t_threshold = int(getattr(cfg.MODEL.DiffusionDet, "QUALITY_GUIDANCE_T_THRESHOLD", 0))
        self.quality_guidance_langevin_steps = int(
            getattr(cfg.MODEL.DiffusionDet, "QUALITY_GUIDANCE_LANGEVIN_STEPS", 1)
        )
        self.quality_guidance_langevin_steps = max(int(self.quality_guidance_langevin_steps), 1)
        self.quality_guidance_langevin_noise = float(
            getattr(cfg.MODEL.DiffusionDet, "QUALITY_GUIDANCE_LANGEVIN_NOISE", 0.0)
        )
        self.quality_guidance_step_schedule = str(
            getattr(cfg.MODEL.DiffusionDet, "QUALITY_GUIDANCE_STEP_SCHEDULE", "constant")
        ).lower()
        self.quality_guidance_time_power = float(getattr(cfg.MODEL.DiffusionDet, "QUALITY_GUIDANCE_TIME_POWER", 1.0))
        self.quality_score_reweight = bool(getattr(cfg.MODEL.DiffusionDet, "QUALITY_SCORE_REWEIGHT", False))
        self.quality_score_power = float(getattr(cfg.MODEL.DiffusionDet, "QUALITY_SCORE_POWER", 1.0))
        self.consistency_distill = bool(getattr(cfg.MODEL.DiffusionDet, "CONSISTENCY_DISTILL", False))
        self.consistency_distill_box_weight = float(
            getattr(cfg.MODEL.DiffusionDet, "CONSISTENCY_DISTILL_BOX_WEIGHT", 0.0)
        )
        self.consistency_distill_cls_weight = float(
            getattr(cfg.MODEL.DiffusionDet, "CONSISTENCY_DISTILL_CLS_WEIGHT", 0.0)
        )
        self.consistency_distill_cls_temperature = float(
            getattr(cfg.MODEL.DiffusionDet, "CONSISTENCY_DISTILL_CLS_TEMPERATURE", 1.0)
        )

        # Phase 3: sampler distillation (teacher multi-step -> student fewer-step).
        self.sampler_distill = bool(getattr(cfg.MODEL.DiffusionDet, "SAMPLER_DISTILL", False))
        self.sampler_distill_teacher_sample_step = int(
            getattr(cfg.MODEL.DiffusionDet, "SAMPLER_DISTILL_TEACHER_SAMPLE_STEP", 20)
        )
        self.sampler_distill_student_sample_step = int(
            getattr(cfg.MODEL.DiffusionDet, "SAMPLER_DISTILL_STUDENT_SAMPLE_STEP", 1)
        )
        self.sampler_distill_student_eta = float(
            getattr(cfg.MODEL.DiffusionDet, "SAMPLER_DISTILL_STUDENT_ETA", 0.0)
        )
        self.sampler_distill_box_weight = float(
            getattr(cfg.MODEL.DiffusionDet, "SAMPLER_DISTILL_BOX_WEIGHT", 0.0)
        )
        self.sampler_distill_cls_weight = float(
            getattr(cfg.MODEL.DiffusionDet, "SAMPLER_DISTILL_CLS_WEIGHT", 0.0)
        )
        self.sampler_distill_cls_temperature = float(
            getattr(cfg.MODEL.DiffusionDet, "SAMPLER_DISTILL_CLS_TEMPERATURE", 1.0)
        )
        self.sampler_distill_topk = max(int(getattr(cfg.MODEL.DiffusionDet, "SAMPLER_DISTILL_TOPK", 0)), 0)

        # Phase 2(A): Label-state (masked corruption / absorbing unk) - minimal skeleton.
        self.use_label_state = bool(getattr(cfg.MODEL.DiffusionDet, "LABEL_STATE", False))
        self.label_state_keep_prob_schedule = str(
            getattr(cfg.MODEL.DiffusionDet, "LABEL_STATE_KEEP_PROB_SCHEDULE", "sqrt_alphas_cumprod")
        ).lower()
        self.label_state_keep_prob_const = float(getattr(cfg.MODEL.DiffusionDet, "LABEL_STATE_KEEP_PROB_CONST", 0.0))
        self.label_state_keep_prob_power = float(getattr(cfg.MODEL.DiffusionDet, "LABEL_STATE_KEEP_PROB_POWER", 1.0))
        self.label_state_keep_prob_min = float(getattr(cfg.MODEL.DiffusionDet, "LABEL_STATE_KEEP_PROB_MIN", 0.0))
        self.label_state_force_unk_infer = bool(getattr(cfg.MODEL.DiffusionDet, "LABEL_STATE_FORCE_UNK_INFER", True))
        self.label_d3pm = bool(getattr(cfg.MODEL.DiffusionDet, "LABEL_D3PM", False))
        self.label_d3pm_kernel = str(getattr(cfg.MODEL.DiffusionDet, "LABEL_D3PM_KERNEL", "mask")).lower()
        self.label_d3pm_use_distribution = bool(
            getattr(cfg.MODEL.DiffusionDet, "LABEL_D3PM_USE_DISTRIBUTION", True)
        )
        self.label_d3pm_infer_update = bool(getattr(cfg.MODEL.DiffusionDet, "LABEL_D3PM_INFER_UPDATE", True))
        # Unknown absorbing state id is K (0..K-1 are real classes).
        self.label_state_unk_id = int(self.num_classes)

        # Build Backbone.
        self.backbone = build_backbone(cfg)
        self.size_divisibility = self.backbone.size_divisibility

        # build diffusion
        timesteps = int(getattr(cfg.MODEL.DiffusionDet, "DIFFUSION_TIMESTEPS", 1000))
        sampling_timesteps = cfg.MODEL.DiffusionDet.SAMPLE_STEP
        self.objective = 'pred_x0'
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 1.
        self.self_condition = False
        self.scale = cfg.MODEL.DiffusionDet.SNR_SCALE
        # Phase 2: anisotropic Gaussian noise in box diffusion space.
        self.aniso_noise = bool(getattr(cfg.MODEL.DiffusionDet, "ANISO_NOISE", False))
        sigma_xy = float(getattr(cfg.MODEL.DiffusionDet, "ANISO_NOISE_SIGMA_XY", 1.0))
        sigma_w = float(getattr(cfg.MODEL.DiffusionDet, "ANISO_NOISE_SIGMA_W", 1.0))
        sigma_h = float(getattr(cfg.MODEL.DiffusionDet, "ANISO_NOISE_SIGMA_H", 1.0))
        sigma_xy = max(sigma_xy, 1e-6)
        sigma_w = max(sigma_w, 1e-6)
        sigma_h = max(sigma_h, 1e-6)
        self.register_buffer(
            "aniso_noise_sigma",
            torch.tensor([sigma_xy, sigma_xy, sigma_w, sigma_h], dtype=torch.float32),
        )
        self.box_renewal = True
        self.use_ensemble = True

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        self.register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # Build Dynamic Head.
        self.head = DynamicHead(cfg=cfg, roi_input_shape=self.backbone.output_shape())
        # Loss parameters:
        class_weight = cfg.MODEL.DiffusionDet.CLASS_WEIGHT
        giou_weight = cfg.MODEL.DiffusionDet.GIOU_WEIGHT
        l1_weight = cfg.MODEL.DiffusionDet.L1_WEIGHT
        no_object_weight = cfg.MODEL.DiffusionDet.NO_OBJECT_WEIGHT
        self.deep_supervision = cfg.MODEL.DiffusionDet.DEEP_SUPERVISION
        self.use_focal = cfg.MODEL.DiffusionDet.USE_FOCAL
        self.use_fed_loss = cfg.MODEL.DiffusionDet.USE_FED_LOSS
        self.use_nms = cfg.MODEL.DiffusionDet.USE_NMS

        # Build Criterion.
        matcher = HungarianMatcherDynamicK(
            cfg=cfg, cost_class=class_weight, cost_bbox=l1_weight, cost_giou=giou_weight, use_focal=self.use_focal
        )
        weight_dict = {"loss_ce": class_weight, "loss_bbox": l1_weight, "loss_giou": giou_weight}
        quality_weight = float(getattr(cfg.MODEL.DiffusionDet, "QUALITY_LOSS_WEIGHT", 1.0))
        if self.use_quality_head:
            weight_dict["loss_quality"] = quality_weight
        graph_weight = float(getattr(cfg.MODEL.DiffusionDet, "GRAPH_TOPO_LOSS_WEIGHT", 0.0))
        if graph_weight > 0.0:
            weight_dict["loss_graph"] = graph_weight
        if self.deep_supervision:
            aux_weight_dict = {}
            for i in range(self.num_heads - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "boxes"]
        if self.use_quality_head:
            losses.append("quality")
        if graph_weight > 0.0:
            losses.append("graph")

        self.criterion = SetCriterionDynamicK(
            cfg=cfg, num_classes=self.num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=no_object_weight,
            losses=losses, use_focal=self.use_focal,)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def set_teacher(self, teacher) -> None:
        # NOTE: do not register teacher as a submodule; otherwise Detectron2 will save teacher
        # weights into every checkpoint and double the disk usage.
        self.__dict__["teacher"] = teacher

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def _label_state_keep_prob(self, t: torch.Tensor) -> torch.Tensor:
        """
        Returns keep_prob(t) in [0, 1], where label is kept with prob keep_prob(t),
        otherwise replaced by the absorbing unk state.
        """
        schedule = self.label_state_keep_prob_schedule
        if schedule == "sqrt_alphas_cumprod":
            kp = self.sqrt_alphas_cumprod[t].to(dtype=torch.float32)
        elif schedule == "alphas_cumprod":
            kp = self.alphas_cumprod[t].to(dtype=torch.float32)
        elif schedule == "linear":
            denom = float(max(self.num_timesteps - 1, 1))
            kp = 1.0 - (t.to(dtype=torch.float32) / denom)
        elif schedule == "constant":
            kp = torch.full_like(t, float(self.label_state_keep_prob_const), dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported LABEL_STATE_KEEP_PROB_SCHEDULE={schedule!r}")

        power = float(self.label_state_keep_prob_power)
        if power != 1.0:
            kp = kp.clamp(min=0.0, max=1.0) ** power
        kp = kp.clamp(min=float(self.label_state_keep_prob_min), max=1.0)
        return kp

    def _label_d3pm_forward(self, c0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Builds q(c_t | c_0) as a categorical distribution over {0..K-1, unk(K)}.

        Args:
            c0: (num_proposals,) long tensor in [0..K] where K is unk.
            t: (1,) long tensor, shared with box diffusion time step.

        Returns:
            probs: (num_proposals, K+1) float tensor, rows sum to 1.
        """
        if c0.dim() != 1:
            raise ValueError(f"Expected c0 to be 1D, got shape={tuple(c0.shape)}")

        num_states = int(self.num_classes) + 1
        unk = int(self.label_state_unk_id)

        keep = self._label_state_keep_prob(t).to(device=c0.device, dtype=torch.float32)
        keep = keep.reshape(1, 1)

        c0 = c0.to(dtype=torch.long)
        c0 = c0.clamp(min=0, max=num_states - 1)
        one_hot = F.one_hot(c0, num_classes=num_states).to(dtype=torch.float32)

        kernel = self.label_d3pm_kernel
        if kernel == "mask":
            probs = one_hot * keep
            # non-unk -> unk with (1-keep)
            is_unk = (c0 == unk).to(dtype=torch.float32).unsqueeze(-1)
            probs[:, unk] = probs[:, unk] + (1.0 - keep.squeeze(0)) * (1.0 - is_unk.squeeze(-1))
            # unk stays unk
            probs = probs * (1.0 - is_unk) + one_hot * is_unk
        elif kernel == "uniform":
            num_states_f = float(num_states)
            uniform = torch.full((c0.shape[0], num_states), 1.0 / num_states_f, device=c0.device)
            probs = one_hot * keep + uniform * (1.0 - keep)
        else:
            raise ValueError(f"Unsupported LABEL_D3PM_KERNEL={kernel!r}")

        probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return probs

    def model_predictions(self, backbone_feats, images_whwh, x, t, x_self_cond=None, label_state=None, clip_x_start=False):
        x_boxes = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x_boxes = ((x_boxes / self.scale) + 1) / 2
        x_boxes = box_cxcywh_to_xyxy(x_boxes)
        x_boxes = x_boxes * images_whwh[:, None, :]
        if label_state is None and self.use_label_state and self.label_state_force_unk_infer:
            if self.label_d3pm and self.label_d3pm_use_distribution:
                num_states = int(self.num_classes) + 1
                unk = int(self.label_state_unk_id)
                label_state = torch.zeros(
                    (x_boxes.shape[0], x_boxes.shape[1], num_states),
                    device=x_boxes.device,
                    dtype=torch.float32,
                )
                label_state[:, :, unk] = 1.0
            else:
                label_state = torch.full(
                    (x_boxes.shape[0], x_boxes.shape[1]),
                    int(self.label_state_unk_id),
                    device=x_boxes.device,
                    dtype=torch.long,
                )
        outputs_class, outputs_coord, outputs_quality = self.head(backbone_feats, x_boxes, t, None, label_state=label_state)

        x_start = outputs_coord[-1]  # (batch, num_proposals, 4) predict boxes: absolute coordinates (x1, y1, x2, y2)
        x_start = x_start / images_whwh[:, None, :]
        x_start = box_xyxy_to_cxcywh(x_start)
        x_start = (x_start * 2 - 1.) * self.scale
        x_start = torch.clamp(x_start, min=-1 * self.scale, max=self.scale)
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start), outputs_class, outputs_coord, outputs_quality

    def _ddim_sample_impl(
        self,
        batched_inputs,
        backbone_feats,
        images_whwh,
        images,
        clip_denoised=True,
        do_postprocess=True,
        init_noise=None,
        init_label_state=None,
        return_raw_predictions=False,
        sampling_timesteps_override: int | None = None,
        eta_override: float | None = None,
        disable_guidance: bool = False,
        disable_ensemble: bool = False,
    ):
        batch = images_whwh.shape[0]
        shape = (batch, self.num_proposals, 4)
        sampling_timesteps = self.sampling_timesteps
        if sampling_timesteps_override is not None:
            sampling_timesteps = int(sampling_timesteps_override)
        sampling_timesteps = max(int(sampling_timesteps), 1)

        eta = self.ddim_sampling_eta
        if eta_override is not None:
            eta = float(eta_override)
        eta = float(eta)

        total_timesteps = self.num_timesteps

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        if init_noise is None:
            img = torch.randn(shape, device=self.device)
            if self.aniso_noise:
                sigma = self.aniso_noise_sigma.to(device=img.device, dtype=img.dtype)
                img = img * sigma
        else:
            img = init_noise.to(device=self.device)
        label_state = init_label_state.to(device=self.device) if init_label_state is not None else None
        if (
            self.use_label_state
            and self.label_d3pm
            and self.label_d3pm_use_distribution
            and self.label_d3pm_infer_update
            and label_state is None
        ):
            num_states = int(self.num_classes) + 1
            unk = int(self.label_state_unk_id)
            label_state = torch.zeros(
                (batch, self.num_proposals, num_states),
                device=self.device,
                dtype=torch.float32,
            )
            if self.label_d3pm_kernel == "uniform":
                label_state.fill_(1.0 / float(num_states))
            else:
                label_state[:, :, unk] = 1.0

        ensemble_score, ensemble_label, ensemble_coord = [], [], []
        x_start = None
        final_box_pred = None
        final_logits = None
        final_quality = None
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=self.device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None

            def _sanitize_boxes(boxes: torch.Tensor) -> torch.Tensor:
                x1, y1, x2, y2 = boxes.unbind(dim=-1)
                x1n = torch.minimum(x1, x2)
                x2n = torch.maximum(x1, x2)
                y1n = torch.minimum(y1, y2)
                y2n = torch.maximum(y1, y2)
                boxes = torch.stack([x1n, y1n, x2n, y2n], dim=-1)
                boxes = boxes.clamp(min=0.0)
                boxes = torch.minimum(boxes, images_whwh[:, None, :])
                x1, y1, x2, y2 = boxes.unbind(dim=-1)
                x2 = torch.maximum(x2, x1 + 1e-6)
                y2 = torch.maximum(y2, y1 + 1e-6)
                return torch.stack([x1, y1, x2, y2], dim=-1)

            use_guidance = (not disable_guidance) and self.use_quality_head and self.quality_guidance_scale > 0.0
            if use_guidance:
                mode = self.quality_guidance_mode
                if mode == "final":
                    use_guidance = time_next < 0
                elif mode == "all":
                    use_guidance = True
                elif mode == "threshold":
                    use_guidance = time <= int(self.quality_guidance_t_threshold)
                else:
                    raise ValueError(f"Unsupported QUALITY_GUIDANCE_MODE={mode!r}")

            q_in = None
            if use_guidance:
                q_in_holder = {}

                def _capture_quality_in(mod, inp, out):
                    if inp and isinstance(inp[0], torch.Tensor):
                        q_in_holder["q_in"] = inp[0]

                rcnn_last = self.head.head_series[-1]
                hook_handle = None
                if hasattr(rcnn_last, "quality_head"):
                    hook_handle = rcnn_last.quality_head.register_forward_hook(_capture_quality_in)
                try:
                    preds, outputs_class, outputs_coord, outputs_quality = self.model_predictions(
                        backbone_feats,
                        images_whwh,
                        img,
                        time_cond,
                        self_cond,
                        label_state=label_state,
                        clip_x_start=clip_denoised,
                    )
                finally:
                    if hook_handle is not None:
                        hook_handle.remove()
                q_in = q_in_holder.get("q_in")
            else:
                preds, outputs_class, outputs_coord, outputs_quality = self.model_predictions(
                    backbone_feats,
                    images_whwh,
                    img,
                    time_cond,
                    self_cond,
                    label_state=label_state,
                    clip_x_start=clip_denoised,
                )

            pred_noise, x_start = preds.pred_noise, preds.pred_x_start
            logits = outputs_class[-1]
            box_pred = outputs_coord[-1]
            quality_pred = outputs_quality[-1] if outputs_quality is not None else None

            if use_guidance and q_in is not None and q_in.shape[1] >= self.hidden_dim + 4:
                score = torch.sigmoid(logits).amax(dim=-1).detach()
                topk_idx = None
                if self.quality_guidance_topk > 0 and self.quality_guidance_topk < score.shape[1]:
                    _, topk_idx = torch.topk(score, k=self.quality_guidance_topk, dim=1)

                # Optional schedule: stronger guidance at low-noise steps (small t).
                step_scale = None
                if self.quality_guidance_step_schedule == "constant":
                    step_scale = None
                elif self.quality_guidance_step_schedule == "linear":
                    denom = float(max(self.num_timesteps - 1, 1))
                    frac = (time_cond.to(dtype=torch.float32) / denom).clamp(0.0, 1.0)
                    ts = (1.0 - frac).clamp(0.0, 1.0)
                    if self.quality_guidance_time_power != 1.0:
                        ts = ts ** float(self.quality_guidance_time_power)
                    step_scale = (float(self.quality_guidance_scale) * ts).view(batch, 1, 1)
                else:
                    raise ValueError(f"Unsupported QUALITY_GUIDANCE_STEP_SCHEDULE={self.quality_guidance_step_schedule!r}")

                with torch.enable_grad():
                    nr_boxes = box_pred.shape[1]

                    # Use only the proposal features from the final head as a fixed condition,
                    # and make the guidance differentiable w.r.t. the output box coordinates.
                    q_feat = q_in[:, : self.hidden_dim].detach()

                    boxes = _sanitize_boxes(box_pred.detach())
                    for _ in range(int(self.quality_guidance_langevin_steps)):
                        boxes_var = boxes.reshape(-1, 4).detach().requires_grad_(True)
                        boxes_view = boxes_var.view(batch, nr_boxes, 4)
                        x1, y1, x2, y2 = boxes_view.unbind(dim=-1)
                        x1n = torch.minimum(x1, x2)
                        x2n = torch.maximum(x1, x2)
                        y1n = torch.minimum(y1, y2)
                        y2n = torch.maximum(y1, y2)
                        boxes_sorted = torch.stack([x1n, y1n, x2n, y2n], dim=-1).reshape(-1, 4)

                        q_geo = rcnn_last._build_quality_box_features(
                            boxes_sorted,
                            dtype=q_feat.dtype,
                            diffusion_t=time_cond,
                            nr_boxes=nr_boxes,
                        )
                        q_in2 = torch.cat([q_feat, q_geo], dim=1)
                        q_logits = rcnn_last.quality_head(q_in2).squeeze(-1).view(batch, nr_boxes)
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
                        grad = grad.view(batch, nr_boxes, 4)
                        if self.quality_guidance_grad_norm == "proposal":
                            grad = grad / (grad.norm(dim=-1, keepdim=True) + 1e-6)
                        elif self.quality_guidance_grad_norm == "global":
                            denom = grad.reshape(grad.shape[0], -1).norm(dim=-1).view(-1, 1, 1)
                            grad = grad / (denom + 1e-6)

                        if step_scale is None:
                            boxes = (boxes_view + float(self.quality_guidance_scale) * grad).detach()
                        else:
                            boxes = (boxes_view + step_scale.to(device=grad.device, dtype=grad.dtype) * grad).detach()

                        noise_scale = float(self.quality_guidance_langevin_noise)
                        if noise_scale > 0.0:
                            boxes = boxes + noise_scale * torch.randn_like(boxes)
                        boxes = _sanitize_boxes(boxes)

                    box_pred = boxes

                # Make guidance affect the diffusion trajectory (not only the final output):
                # update x_start and pred_noise consistently with the guided boxes.
                x0 = box_xyxy_to_cxcywh(box_pred / images_whwh[:, None, :])
                x0 = (x0 * 2.0 - 1.0) * self.scale
                x0 = torch.clamp(x0, min=-1 * self.scale, max=self.scale)
                x_start = x0
                pred_noise = self.predict_noise_from_start(img, time_cond, x_start)

            final_box_pred = box_pred
            final_logits = logits
            final_quality = quality_pred

            label_state_next = None
            if label_state is not None and self.label_d3pm_infer_update:
                p0 = torch.softmax(logits.to(dtype=torch.float32), dim=-1)
                if time_next < 0:
                    keep = torch.ones((batch,), device=self.device, dtype=torch.float32)
                else:
                    time_next_cond = torch.full((batch,), time_next, device=self.device, dtype=torch.long)
                    keep = self._label_state_keep_prob(time_next_cond).to(device=self.device, dtype=torch.float32)
                keep = keep.view(batch, 1, 1)

                num_states = int(self.num_classes) + 1
                unk = int(self.label_state_unk_id)
                label_state_next = torch.zeros(
                    (batch, p0.shape[1], num_states),
                    device=self.device,
                    dtype=torch.float32,
                )
                if self.label_d3pm_kernel == "uniform":
                    uniform_prob = (1.0 - keep) / float(num_states)
                    label_state_next[:, :, : self.num_classes] = p0 * keep + uniform_prob
                    label_state_next[:, :, unk] = uniform_prob.expand(batch, p0.shape[1], 1).squeeze(-1)
                elif self.label_d3pm_kernel == "mask":
                    label_state_next[:, :, : self.num_classes] = p0 * keep
                    label_state_next[:, :, unk] = (1.0 - keep).expand(batch, p0.shape[1], 1).squeeze(-1)
                else:
                    raise ValueError(f"Unsupported LABEL_D3PM_KERNEL={self.label_d3pm_kernel!r}")
                label_state_next = label_state_next / label_state_next.sum(dim=-1, keepdim=True).clamp(min=1e-6)

            if self.box_renewal:  # filter
                score_per_image, box_per_image = logits[0], box_pred[0]
                threshold = 0.5
                score_per_image = torch.sigmoid(score_per_image)
                value, _ = torch.max(score_per_image, -1, keepdim=False)
                keep_idx = value > threshold
                num_remain = torch.sum(keep_idx)

                pred_noise = pred_noise[:, keep_idx, :]
                x_start = x_start[:, keep_idx, :]
                img = img[:, keep_idx, :]
                if label_state_next is not None:
                    label_state_next = label_state_next[:, keep_idx, :]
            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)
            if self.aniso_noise:
                sigma_vec = self.aniso_noise_sigma.to(device=noise.device, dtype=noise.dtype)
                noise = noise * sigma_vec

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            if self.box_renewal:  # filter
                # replenish with randn boxes
                replenish = torch.randn(1, self.num_proposals - num_remain, 4, device=img.device)
                if self.aniso_noise:
                    sigma_vec = self.aniso_noise_sigma.to(device=replenish.device, dtype=replenish.dtype)
                    replenish = replenish * sigma_vec
                img = torch.cat((img, replenish), dim=1)
                if label_state_next is not None:
                    pad_count = int(self.num_proposals - num_remain.item())
                    pad = torch.zeros(
                        (batch, pad_count, int(self.num_classes) + 1),
                        device=self.device,
                        dtype=torch.float32,
                    )
                    pad[:, :, int(self.label_state_unk_id)] = 1.0
                    label_state_next = torch.cat((label_state_next, pad), dim=1)

            if label_state_next is not None:
                label_state = label_state_next
            if (not disable_ensemble) and self.use_ensemble and sampling_timesteps > 1:
                box_pred_per_image, scores_per_image, labels_per_image = self.inference(
                    logits,
                    box_pred,
                    images.image_sizes,
                    box_quality=quality_pred,
                )
                ensemble_score.append(scores_per_image)
                ensemble_label.append(labels_per_image)
                ensemble_coord.append(box_pred_per_image)

        output = {"pred_logits": final_logits, "pred_boxes": final_box_pred}
        if final_quality is not None:
            output["pred_quality"] = final_quality

        if return_raw_predictions:
            return output

        if self.use_ensemble and sampling_timesteps > 1:
            box_pred_per_image = torch.cat(ensemble_coord, dim=0)
            scores_per_image = torch.cat(ensemble_score, dim=0)
            labels_per_image = torch.cat(ensemble_label, dim=0)
            if self.use_nms:
                keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, 0.5)
                box_pred_per_image = box_pred_per_image[keep]
                scores_per_image = scores_per_image[keep]
                labels_per_image = labels_per_image[keep]

            result = Instances(images.image_sizes[0])
            result.pred_boxes = Boxes(box_pred_per_image)
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results = [result]
        else:
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            results = self.inference(box_cls, box_pred, images.image_sizes, box_quality=final_quality)

        if do_postprocess:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results
        return results

    @torch.no_grad()
    def ddim_sample(
        self,
        batched_inputs,
        backbone_feats,
        images_whwh,
        images,
        clip_denoised=True,
        do_postprocess=True,
        init_noise=None,
        init_label_state=None,
        return_raw_predictions=False,
        sampling_timesteps_override: int | None = None,
        eta_override: float | None = None,
        disable_guidance: bool = False,
        disable_ensemble: bool = False,
    ):
        return self._ddim_sample_impl(
            batched_inputs,
            backbone_feats,
            images_whwh,
            images,
            clip_denoised=clip_denoised,
            do_postprocess=do_postprocess,
            init_noise=init_noise,
            init_label_state=init_label_state,
            return_raw_predictions=return_raw_predictions,
            sampling_timesteps_override=sampling_timesteps_override,
            eta_override=eta_override,
            disable_guidance=disable_guidance,
            disable_ensemble=disable_ensemble,
        )

    def ddim_sample_with_grad(
        self,
        batched_inputs,
        backbone_feats,
        images_whwh,
        images,
        clip_denoised=True,
        do_postprocess=True,
        init_noise=None,
        init_label_state=None,
        return_raw_predictions=False,
        sampling_timesteps_override: int | None = None,
        eta_override: float | None = None,
        disable_guidance: bool = False,
        disable_ensemble: bool = False,
    ):
        return self._ddim_sample_impl(
            batched_inputs,
            backbone_feats,
            images_whwh,
            images,
            clip_denoised=clip_denoised,
            do_postprocess=do_postprocess,
            init_noise=init_noise,
            init_label_state=init_label_state,
            return_raw_predictions=return_raw_predictions,
            sampling_timesteps_override=sampling_timesteps_override,
            eta_override=eta_override,
            disable_guidance=disable_guidance,
            disable_ensemble=disable_ensemble,
        )

    # forward diffusion
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        if self.aniso_noise:
            sigma = self.aniso_noise_sigma.to(device=noise.device, dtype=noise.dtype)
            noise = noise * sigma

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def forward(self, batched_inputs, do_postprocess=True):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        """
        images, images_whwh = self.preprocess_image(batched_inputs)
        if isinstance(images, (list, torch.Tensor)):
            images = nested_tensor_from_tensor_list(images)

        # Feature Extraction.
        src = self.backbone(images.tensor)
        features = list()
        for f in self.in_features:
            feature = src[f]
            features.append(feature)

        # Prepare Proposals.
        if not self.training:
            results = self.ddim_sample(batched_inputs, features, images_whwh, images)
            return results

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets, x_boxes, noises, t, label_state = self.prepare_targets(gt_instances)
            t = t.squeeze(-1)
            x_boxes = x_boxes * images_whwh[:, None, :]

            outputs_class, outputs_coord, outputs_quality = self.head(features, x_boxes, t, None, label_state=label_state)
            output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
            if outputs_quality is not None:
                output['pred_quality'] = outputs_quality[-1]
            if getattr(self.head, "last_attn_weights", None) is not None:
                output["pred_attn"] = self.head.last_attn_weights

            if self.deep_supervision:
                if outputs_quality is None:
                    output['aux_outputs'] = [{'pred_logits': a, 'pred_boxes': b}
                                             for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
                else:
                    output['aux_outputs'] = [
                        {'pred_logits': a, 'pred_boxes': b, 'pred_quality': q}
                        for a, b, q in zip(outputs_class[:-1], outputs_coord[:-1], outputs_quality[:-1])
                    ]

            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]

            # Phase 3: optional teacher->student distillation loss (consistency distillation entry).
            teacher = self.__dict__.get("teacher", None)
            if (
                teacher is not None
                and self.consistency_distill
                and (self.consistency_distill_box_weight > 0.0 or self.consistency_distill_cls_weight > 0.0)
            ):
                with torch.no_grad():
                    src_t = teacher.backbone(images.tensor)
                    feats_t = [src_t[f] for f in teacher.in_features]
                    t_cls, t_coord, _t_q = teacher.head(feats_t, x_boxes, t, None, label_state=label_state)
                    teacher_logits = t_cls[-1].detach()
                    teacher_boxes = t_coord[-1].detach()

                if self.consistency_distill_box_weight > 0.0:
                    stu = outputs_coord[-1] / images_whwh[:, None, :]
                    tea = teacher_boxes / images_whwh[:, None, :]
                    loss_box = F.l1_loss(stu, tea, reduction="mean")
                    loss_dict["loss_consistency_box"] = loss_box * float(self.consistency_distill_box_weight)

                if self.consistency_distill_cls_weight > 0.0:
                    temp = max(float(self.consistency_distill_cls_temperature), 1e-6)
                    p_s = torch.sigmoid(outputs_class[-1] / temp)
                    p_t = torch.sigmoid(teacher_logits / temp)
                    loss_cls = F.mse_loss(p_s, p_t, reduction="mean")
                    loss_dict["loss_consistency_cls"] = loss_cls * float(self.consistency_distill_cls_weight)

            # Phase 3: sampler distillation (teacher multi-step -> student one-step).
            # Train student to predict teacher's final denoised boxes/logits from pure noise in one forward.
            if (
                teacher is not None
                and self.sampler_distill
                and (self.sampler_distill_box_weight > 0.0 or self.sampler_distill_cls_weight > 0.0)
            ):
                batch = images_whwh.shape[0]
                init_noise = torch.randn(
                    (batch, self.num_proposals, 4),
                    device=self.device,
                    dtype=outputs_coord[-1].dtype,
                )
                if self.aniso_noise:
                    sigma = self.aniso_noise_sigma.to(device=init_noise.device, dtype=init_noise.dtype)
                    init_noise = init_noise * sigma

                init_label_state = None
                if (
                    self.use_label_state
                    and self.label_d3pm
                    and self.label_d3pm_use_distribution
                    and self.label_d3pm_infer_update
                ):
                    num_states = int(self.num_classes) + 1
                    unk = int(self.label_state_unk_id)
                    init_label_state = torch.zeros(
                        (batch, self.num_proposals, num_states),
                        device=self.device,
                        dtype=torch.float32,
                    )
                    if self.label_d3pm_kernel == "uniform":
                        init_label_state.fill_(1.0 / float(num_states))
                    else:
                        init_label_state[:, :, unk] = 1.0

                student_logits = None
                student_boxes = None
                if int(self.sampler_distill_student_sample_step) <= 1:
                    t_noise = torch.full((batch,), self.num_timesteps - 1, device=self.device, dtype=torch.long)
                    _preds_s, s_cls, s_coord, _s_q = self.model_predictions(
                        features,
                        images_whwh,
                        init_noise,
                        t_noise,
                        None,
                        label_state=init_label_state,
                        clip_x_start=False,
                    )
                    student_logits = s_cls[-1]
                    student_boxes = s_coord[-1]
                else:
                    stu_out = self.ddim_sample_with_grad(
                        batched_inputs,
                        features,
                        images_whwh,
                        images,
                        clip_denoised=True,
                        do_postprocess=False,
                        init_noise=init_noise,
                        init_label_state=init_label_state,
                        return_raw_predictions=True,
                        sampling_timesteps_override=int(self.sampler_distill_student_sample_step),
                        eta_override=float(self.sampler_distill_student_eta),
                        disable_guidance=True,
                        disable_ensemble=True,
                    )
                    student_logits = stu_out["pred_logits"]
                    student_boxes = stu_out["pred_boxes"]

                with torch.no_grad():
                    src_t = teacher.backbone(images.tensor)
                    feats_t = [src_t[f] for f in teacher.in_features]
                    tea_out = teacher.ddim_sample(
                        batched_inputs,
                        feats_t,
                        images_whwh,
                        images,
                        do_postprocess=False,
                        init_noise=init_noise,
                        init_label_state=init_label_state,
                        return_raw_predictions=True,
                    )
                    teacher_logits = tea_out["pred_logits"].detach()
                    teacher_boxes = tea_out["pred_boxes"].detach()

                topk_idx = None
                if self.sampler_distill_topk > 0 and teacher_logits is not None:
                    n_props = int(teacher_logits.shape[1])
                    if self.sampler_distill_topk < n_props:
                        scores = torch.sigmoid(teacher_logits).amax(dim=-1)
                        _, topk_idx = torch.topk(scores, k=int(self.sampler_distill_topk), dim=1, sorted=False)

                if self.sampler_distill_box_weight > 0.0:
                    if topk_idx is not None:
                        idx_box = topk_idx.unsqueeze(-1).expand(-1, -1, 4)
                        student_boxes_sel = student_boxes.gather(1, idx_box)
                        teacher_boxes_sel = teacher_boxes.gather(1, idx_box)
                        stu = student_boxes_sel / images_whwh[:, None, :]
                        tea = teacher_boxes_sel / images_whwh[:, None, :]
                    else:
                        stu = student_boxes / images_whwh[:, None, :]
                        tea = teacher_boxes / images_whwh[:, None, :]
                    loss_box = F.l1_loss(stu, tea, reduction="mean")
                    loss_dict["loss_sampler_distill_box"] = loss_box * float(self.sampler_distill_box_weight)

                if self.sampler_distill_cls_weight > 0.0:
                    if topk_idx is not None:
                        idx_logit = topk_idx.unsqueeze(-1).expand(-1, -1, int(student_logits.shape[-1]))
                        student_logits = student_logits.gather(1, idx_logit)
                        teacher_logits = teacher_logits.gather(1, idx_logit)
                    temp = max(float(self.sampler_distill_cls_temperature), 1e-6)
                    p_s = torch.sigmoid(student_logits / temp)
                    p_t = torch.sigmoid(teacher_logits / temp)
                    loss_cls = F.mse_loss(p_s, p_t, reduction="mean")
                    loss_dict["loss_sampler_distill_cls"] = loss_cls * float(self.sampler_distill_cls_weight)
            return loss_dict

    def prepare_diffusion_repeat(self, gt_boxes, gt_classes):
        """
        :param gt_boxes: (cx, cy, w, h), normalized
        :param num_proposals:
        """
        t = torch.randint(0, self.num_timesteps, (1,), device=self.device).long()
        noise = torch.randn(self.num_proposals, 4, device=self.device)

        num_gt = gt_boxes.shape[0]
        num_gt_classes = int(gt_classes.shape[0]) if gt_classes is not None else 0
        if not num_gt:  # generate fake gt boxes if empty gt boxes
            gt_boxes = torch.as_tensor([[0.5, 0.5, 1., 1.]], dtype=torch.float, device=self.device)
            num_gt = 1

        num_repeat = self.num_proposals // num_gt  # number of repeat except the last gt box in one image
        repeat_tensor = [num_repeat] * (num_gt - self.num_proposals % num_gt) + [num_repeat + 1] * (
                self.num_proposals % num_gt)
        assert sum(repeat_tensor) == self.num_proposals
        random.shuffle(repeat_tensor)
        repeat_tensor = torch.tensor(repeat_tensor, device=self.device)

        gt_boxes = (gt_boxes * 2. - 1.) * self.scale
        x_start = torch.repeat_interleave(gt_boxes, repeat_tensor, dim=0)

        label_state = None
        if self.use_label_state:
            unk = int(self.label_state_unk_id)
            if self.label_d3pm:
                if num_gt_classes > 0:
                    c0 = torch.repeat_interleave(gt_classes.to(self.device), repeat_tensor, dim=0)
                else:
                    c0 = torch.full((self.num_proposals,), unk, device=self.device, dtype=torch.long)
                probs = self._label_d3pm_forward(c0, t)
                if self.label_d3pm_use_distribution:
                    label_state = probs
                else:
                    label_state = torch.multinomial(probs, num_samples=1).squeeze(1).to(dtype=torch.long)
            else:
                label_state = torch.full((self.num_proposals,), unk, device=self.device, dtype=torch.long)
                if num_gt_classes > 0:
                    labels = torch.repeat_interleave(gt_classes.to(self.device), repeat_tensor, dim=0)
                    keep_prob = self._label_state_keep_prob(t).to(device=labels.device)
                    keep = torch.rand_like(labels.to(dtype=torch.float32)) < keep_prob.to(dtype=torch.float32)
                    labels = labels.clone()
                    labels[~keep] = unk
                    label_state[:] = labels

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        x = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x = ((x / self.scale) + 1) / 2.

        diff_boxes = box_cxcywh_to_xyxy(x)

        return diff_boxes, noise, t, label_state

    def prepare_diffusion_concat(self, gt_boxes, gt_classes):
        """
        :param gt_boxes: (cx, cy, w, h), normalized
        :param num_proposals:
        """
        t = torch.randint(0, self.num_timesteps, (1,), device=self.device).long()
        noise = torch.randn(self.num_proposals, 4, device=self.device)

        num_gt = gt_boxes.shape[0]
        num_gt_classes = int(gt_classes.shape[0]) if gt_classes is not None else 0
        if not num_gt:  # generate fake gt boxes if empty gt boxes
            gt_boxes = torch.as_tensor([[0.5, 0.5, 1., 1.]], dtype=torch.float, device=self.device)
            num_gt = 1

        label_state = None
        if num_gt < self.num_proposals:
            box_placeholder = torch.randn(self.num_proposals - num_gt, 4,
                                          device=self.device) / 6. + 0.5  # 3sigma = 1/2 --> sigma: 1/6
            box_placeholder[:, 2:] = torch.clip(box_placeholder[:, 2:], min=1e-4)
            x_start = torch.cat((gt_boxes, box_placeholder), dim=0)
            if self.use_label_state:
                unk = int(self.label_state_unk_id)
                if self.label_d3pm:
                    c0 = torch.full((self.num_proposals,), unk, device=self.device, dtype=torch.long)
                    if num_gt_classes > 0:
                        c0[:num_gt_classes] = gt_classes.to(self.device)
                    probs = self._label_d3pm_forward(c0, t)
                    if self.label_d3pm_use_distribution:
                        label_state = probs
                    else:
                        label_state = torch.multinomial(probs, num_samples=1).squeeze(1).to(dtype=torch.long)
                else:
                    label_state = torch.full((self.num_proposals,), unk, device=self.device, dtype=torch.long)
                    if num_gt_classes > 0:
                        labels = gt_classes.to(self.device)
                        keep_prob = self._label_state_keep_prob(t).to(device=labels.device)
                        keep = torch.rand_like(labels.to(dtype=torch.float32)) < keep_prob.to(dtype=torch.float32)
                        labels = labels.clone()
                        labels[~keep] = unk
                        label_state[:num_gt_classes] = labels
        elif num_gt > self.num_proposals:
            select_mask = [True] * self.num_proposals + [False] * (num_gt - self.num_proposals)
            random.shuffle(select_mask)
            x_start = gt_boxes[select_mask]
            if self.use_label_state:
                unk = int(self.label_state_unk_id)
                c0 = gt_classes.to(self.device)[select_mask]
                if self.label_d3pm:
                    probs = self._label_d3pm_forward(c0, t)
                    if self.label_d3pm_use_distribution:
                        label_state = probs
                    else:
                        label_state = torch.multinomial(probs, num_samples=1).squeeze(1).to(dtype=torch.long)
                else:
                    labels = c0
                    keep_prob = self._label_state_keep_prob(t).to(device=labels.device)
                    keep = torch.rand_like(labels.to(dtype=torch.float32)) < keep_prob.to(dtype=torch.float32)
                    labels = labels.clone()
                    labels[~keep] = unk
                    label_state = labels
        else:
            x_start = gt_boxes
            if self.use_label_state:
                unk = int(self.label_state_unk_id)
                if self.label_d3pm:
                    c0 = torch.full((self.num_proposals,), unk, device=self.device, dtype=torch.long)
                    if num_gt_classes > 0:
                        c0[:num_gt_classes] = gt_classes.to(self.device)
                    probs = self._label_d3pm_forward(c0, t)
                    if self.label_d3pm_use_distribution:
                        label_state = probs
                    else:
                        label_state = torch.multinomial(probs, num_samples=1).squeeze(1).to(dtype=torch.long)
                else:
                    label_state = torch.full((self.num_proposals,), unk, device=self.device, dtype=torch.long)
                    if num_gt_classes > 0:
                        labels = gt_classes.to(self.device)
                        keep_prob = self._label_state_keep_prob(t).to(device=labels.device)
                        keep = torch.rand_like(labels.to(dtype=torch.float32)) < keep_prob.to(dtype=torch.float32)
                        labels = labels.clone()
                        labels[~keep] = unk
                        label_state[:num_gt_classes] = labels

        x_start = (x_start * 2. - 1.) * self.scale

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        x = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x = ((x / self.scale) + 1) / 2.

        diff_boxes = box_cxcywh_to_xyxy(x)

        return diff_boxes, noise, t, label_state

    def prepare_targets(self, targets):
        new_targets = []
        diffused_boxes = []
        noises = []
        ts = []
        label_states = [] if self.use_label_state else None
        for targets_per_image in targets:
            target = {}
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            d_boxes, d_noise, d_t, d_label_state = self.prepare_diffusion_concat(gt_boxes, gt_classes.to(self.device))
            diffused_boxes.append(d_boxes)
            noises.append(d_noise)
            ts.append(d_t)
            if label_states is not None:
                label_states.append(d_label_state)
            target["labels"] = gt_classes.to(self.device)
            target["boxes"] = gt_boxes.to(self.device)
            target["boxes_xyxy"] = targets_per_image.gt_boxes.tensor.to(self.device)
            target["image_size_xyxy"] = image_size_xyxy.to(self.device)
            image_size_xyxy_tgt = image_size_xyxy.unsqueeze(0).repeat(len(gt_boxes), 1)
            target["image_size_xyxy_tgt"] = image_size_xyxy_tgt.to(self.device)
            target["area"] = targets_per_image.gt_boxes.area().to(self.device)
            new_targets.append(target)

        label_states_tensor = torch.stack(label_states) if label_states is not None else None
        ts_tensor = torch.stack(ts)
        if self.use_label_state and label_states_tensor is not None:
            try:
                from detectron2.utils.events import get_event_storage

                storage = get_event_storage()
                t_batch = ts_tensor.squeeze(-1)
                kp = self._label_state_keep_prob(t_batch)
                storage.put_scalar("label_state_keep_prob", float(kp.mean().item()))
                unk = int(self.label_state_unk_id)
                if label_states_tensor.dim() == 3 and label_states_tensor.dtype.is_floating_point:
                    non_unk = 1.0 - label_states_tensor[:, :, unk].to(dtype=torch.float32).mean()
                else:
                    non_unk = (label_states_tensor != unk).to(dtype=torch.float32).mean()
                storage.put_scalar("label_state_non_unk_ratio", float(non_unk.item()))
            except Exception:
                pass

        return new_targets, torch.stack(diffused_boxes), torch.stack(noises), ts_tensor, label_states_tensor

    def inference(self, box_cls, box_pred, image_sizes, box_quality=None):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_proposals, K).
                The tensor predicts the classification probability for each proposal.
            box_pred (Tensor): tensors of shape (batch_size, num_proposals, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every proposal
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        if self.use_focal or self.use_fed_loss:
            scores = torch.sigmoid(box_cls)
            if self.quality_score_reweight and box_quality is not None:
                q = torch.sigmoid(box_quality)
                power = float(self.quality_score_power)
                if power != 1.0:
                    q = q.clamp(min=1e-6) ** power
                scores = scores * q.unsqueeze(-1)
            labels = torch.arange(self.num_classes, device=self.device). \
                unsqueeze(0).repeat(self.num_proposals, 1).flatten(0, 1)

            for i, (scores_per_image, box_pred_per_image, image_size) in enumerate(zip(
                    scores, box_pred, image_sizes
            )):
                result = Instances(image_size)
                scores_per_image, topk_indices = scores_per_image.flatten(0, 1).topk(self.num_proposals, sorted=False)
                labels_per_image = labels[topk_indices]
                box_pred_per_image = box_pred_per_image.view(-1, 1, 4).repeat(1, self.num_classes, 1).view(-1, 4)
                box_pred_per_image = box_pred_per_image[topk_indices]

                if self.use_ensemble and self.sampling_timesteps > 1:
                    return box_pred_per_image, scores_per_image, labels_per_image

                if self.use_nms:
                    keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, 0.5)
                    box_pred_per_image = box_pred_per_image[keep]
                    scores_per_image = scores_per_image[keep]
                    labels_per_image = labels_per_image[keep]

                result.pred_boxes = Boxes(box_pred_per_image)
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                results.append(result)

        else:
            # For each box we assign the best class or the second best if the best on is `no_object`.
            scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)
            if self.quality_score_reweight and box_quality is not None:
                q = torch.sigmoid(box_quality)
                power = float(self.quality_score_power)
                if power != 1.0:
                    q = q.clamp(min=1e-6) ** power
                scores = scores * q

            for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(
                    scores, labels, box_pred, image_sizes
            )):
                if self.use_ensemble and self.sampling_timesteps > 1:
                    return box_pred_per_image, scores_per_image, labels_per_image

                if self.use_nms:
                    keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, 0.5)
                    box_pred_per_image = box_pred_per_image[keep]
                    scores_per_image = scores_per_image[keep]
                    labels_per_image = labels_per_image[keep]
                result = Instances(image_size)
                result.pred_boxes = Boxes(box_pred_per_image)
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                results.append(result)

        return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images, self.size_divisibility)

        images_whwh = list()
        for bi in batched_inputs:
            h, w = bi["image"].shape[-2:]
            images_whwh.append(torch.tensor([w, h, w, h], dtype=torch.float32, device=self.device))
        images_whwh = torch.stack(images_whwh)

        return images, images_whwh
