from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from diffusers.utils import BaseOutput

try:
    from diffusers import DiffusionPipeline as _DiffusionPipeline
except Exception:
    # Some environments have a broken `flash_attn` / `transformers` chain that prevents importing
    # diffusers' pipeline base classes. We still keep the scheduler-based implementation usable.
    _DiffusionPipeline = None

from .schedulers import D3PMConfig, D3PMLabelScheduler, build_ddim_scheduler_from_betas


class HybridPipelineOutput(BaseOutput):
    """
    Output container aligned with Detectron2 inference usage.
    """

    instances: Any | None = None
    pred_boxes: torch.Tensor | None = None
    pred_logits: torch.Tensor | None = None
    pred_quality: torch.Tensor | None = None


class HybridDiffusionPipeline(_DiffusionPipeline if _DiffusionPipeline is not None else object):
    """
    Diffusers-style wrapper for this repo's detection diffusion model.

    - Box diffusion: DDIMScheduler(prediction_type="sample") in the same diffusion space as DiffusionDet (cxcywh in [-scale,scale]).
    - Label diffusion: custom D3PM-like scheduler with absorbing unk state (optional).

    This is intentionally inference-first; training in this repo remains Detectron2-based.
    """

    def __init__(self, *, detector: torch.nn.Module, box_scheduler, label_scheduler: D3PMLabelScheduler | None = None):
        if _DiffusionPipeline is not None:
            super().__init__()
            self.register_modules(detector=detector, box_scheduler=box_scheduler)
            self.label_scheduler = label_scheduler
        else:
            # Minimal fallback to keep this module functional without `diffusers.DiffusionPipeline`.
            self.detector = detector
            self.box_scheduler = box_scheduler
            self.label_scheduler = label_scheduler

    @classmethod
    def from_diffusiondet(cls, detector: torch.nn.Module) -> "HybridDiffusionPipeline":
        if not hasattr(detector, "betas"):
            raise ValueError("Expected a DiffusionDet-like model with .betas")
        box_scheduler = build_ddim_scheduler_from_betas(detector.betas)

        label_scheduler = None
        if bool(getattr(detector, "use_label_state", False)) and bool(getattr(detector, "label_d3pm", False)):
            cfg = D3PMConfig(
                num_classes=int(getattr(detector, "num_classes")),
                kernel=str(getattr(detector, "label_d3pm_kernel", "mask")).lower(),
                keep_prob_schedule=str(getattr(detector, "label_state_keep_prob_schedule", "sqrt_alphas_cumprod")).lower(),
                keep_prob_const=float(getattr(detector, "label_state_keep_prob_const", 0.0)),
                keep_prob_power=float(getattr(detector, "label_state_keep_prob_power", 1.0)),
                keep_prob_min=float(getattr(detector, "label_state_keep_prob_min", 0.0)),
            )
            label_scheduler = D3PMLabelScheduler(betas=detector.betas, config=cfg)

        return cls(detector=detector, box_scheduler=box_scheduler, label_scheduler=label_scheduler)

    @torch.no_grad()
    def __call__(
        self,
        *,
        batched_inputs: List[Dict[str, Any]],
        num_inference_steps: Optional[int] = None,
        eta: Optional[float] = None,
        clip_denoised: bool = True,
        return_raw_predictions: bool = False,
    ) -> HybridPipelineOutput:
        detector = self.detector
        device = next(detector.parameters()).device

        images, images_whwh = detector.preprocess_image(batched_inputs)

        src = detector.backbone(images.tensor)
        features = [src[f] for f in detector.in_features]

        batch = int(images_whwh.shape[0])
        num_props = int(getattr(detector, "num_proposals"))
        shape = (batch, num_props, 4)

        # DDIM init noise (DiffusionDet uses eta=1 by default; keep consistent unless overridden).
        if eta is None:
            eta = float(getattr(detector, "ddim_sampling_eta", 1.0))
        eta = float(eta)

        x = torch.randn(shape, device=device, dtype=torch.float32)
        if bool(getattr(detector, "aniso_noise", False)):
            sigma = detector.aniso_noise_sigma.to(device=device, dtype=x.dtype)
            x = x * sigma

        label_state = None
        if self.label_scheduler is not None:
            label_state = self.label_scheduler.prior(batch, num_props, device=device)

        if num_inference_steps is None:
            num_inference_steps = int(getattr(detector, "sampling_timesteps", 1))
        num_inference_steps = max(int(num_inference_steps), 1)

        self.box_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.box_scheduler.timesteps

        final_logits = None
        final_box_pred = None
        final_quality = None

        for i, t in enumerate(timesteps):
            time = int(t.item())
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)

            preds, outputs_class, outputs_coord, outputs_quality = detector.model_predictions(
                features,
                images_whwh,
                x,
                time_cond,
                label_state=label_state,
                clip_x_start=clip_denoised,
            )

            x_start = preds.pred_x_start
            final_logits = outputs_class[-1]
            final_box_pred = outputs_coord[-1]
            final_quality = outputs_quality[-1] if outputs_quality is not None else None

            step_out = self.box_scheduler.step(model_output=x_start, timestep=time, sample=x, eta=eta)
            x = step_out.prev_sample

            if (
                label_state is not None
                and bool(getattr(detector, "label_d3pm_infer_update", True))
                and self.label_scheduler is not None
            ):
                t_next = int(timesteps[i + 1].item()) if (i + 1) < len(timesteps) else -1
                label_state = self.label_scheduler.infer_update(final_logits, t_next=t_next, device=device)

        out = HybridPipelineOutput()
        out.pred_logits = final_logits
        out.pred_boxes = final_box_pred
        out.pred_quality = final_quality

        if not return_raw_predictions and final_logits is not None and final_box_pred is not None:
            results = detector.inference(final_logits, final_box_pred, images.image_sizes, box_quality=final_quality)
            out.instances = results

        return out
