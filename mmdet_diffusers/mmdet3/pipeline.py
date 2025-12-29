from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
try:
    from diffusers import DiffusionPipeline as _DiffusionPipeline
    from diffusers.utils import BaseOutput
except Exception:  # pragma: no cover
    _DiffusionPipeline = None

    class BaseOutput(dict):  # type: ignore[no-redef]
        pass
from mmengine.config import Config
from mmengine.runner.checkpoint import load_checkpoint

from mmdet.registry import MODELS
from mmdet.structures import DetDataSample

from mmdet_diffusers.schedulers import D3PMLabelScheduler, build_ddim_scheduler_from_betas


class MMDet3PipelineOutput(BaseOutput):
    data_samples: List[DetDataSample] | None = None
    pred_boxes: torch.Tensor | None = None  # (B,N,4) abs xyxy
    pred_logits: torch.Tensor | None = None  # (B,N,K+1)
    pred_quality: torch.Tensor | None = None  # (B,N) quality logits


@dataclass(frozen=True)
class _PreparedBatch:
    batch_inputs: torch.Tensor
    batch_data_samples: List[DetDataSample]


class MMDet3HybridDiffusionPipeline(_DiffusionPipeline if _DiffusionPipeline is not None else object):
    """
    Diffusers-style pipeline for MMDet3 diffusion detectors.

    The pipeline:
    - preprocesses input with the detector's `data_preprocessor`
    - runs DDIM sampling via `diffusers.DDIMScheduler` (prediction_type="sample")
    - updates label state with `D3PMLabelScheduler`
    """

    def __init__(
        self,
        *,
        detector: torch.nn.Module,
        box_scheduler,
        label_scheduler: D3PMLabelScheduler,
    ):
        if _DiffusionPipeline is not None:
            super().__init__()
            self.register_modules(detector=detector, box_scheduler=box_scheduler)
            self.label_scheduler = label_scheduler
        else:
            self.detector = detector
            self.box_scheduler = box_scheduler
            self.label_scheduler = label_scheduler

    @classmethod
    def from_detector(cls, detector: torch.nn.Module) -> "MMDet3HybridDiffusionPipeline":
        if not hasattr(detector, "betas"):
            raise ValueError("Expected detector to expose a `.betas` buffer.")
        box_scheduler = build_ddim_scheduler_from_betas(detector.betas)

        label_scheduler = getattr(detector, "label_scheduler", None)
        if not isinstance(label_scheduler, D3PMLabelScheduler):
            raise ValueError("Expected detector to expose a `.label_scheduler` (D3PMLabelScheduler).")
        return cls(detector=detector, box_scheduler=box_scheduler, label_scheduler=label_scheduler)

    @classmethod
    def from_mmdet3_config(
        cls,
        *,
        config: str,
        checkpoint: Optional[str] = None,
        device: str = "cuda",
    ) -> "MMDet3HybridDiffusionPipeline":
        cfg = Config.fromfile(config)
        model = MODELS.build(cfg.model)
        if checkpoint is not None:
            load_checkpoint(model, checkpoint, map_location="cpu", revise_keys=[(r"^module\\.", "")])
        model.eval()
        model.to(torch.device(device))
        return cls.from_detector(model)

    def _prepare_batch(
        self,
        *,
        data: Optional[Dict[str, Any]] = None,
        batch_inputs: Optional[torch.Tensor] = None,
        batch_data_samples: Optional[List[DetDataSample]] = None,
    ) -> _PreparedBatch:
        if data is not None:
            processed = self.detector.data_preprocessor(data, training=False)
            batch_inputs = processed["inputs"]
            batch_data_samples = processed.get("data_samples", None)
            if batch_data_samples is None:
                raise ValueError("data must contain `data_samples` after preprocessing.")
            return _PreparedBatch(batch_inputs=batch_inputs, batch_data_samples=batch_data_samples)
        if batch_inputs is None or batch_data_samples is None:
            raise ValueError("Provide either `data` or (`batch_inputs`, `batch_data_samples`).")
        return _PreparedBatch(batch_inputs=batch_inputs, batch_data_samples=batch_data_samples)

    @torch.no_grad()
    def __call__(
        self,
        *,
        data: Optional[Dict[str, Any]] = None,
        batch_inputs: Optional[torch.Tensor] = None,
        batch_data_samples: Optional[List[DetDataSample]] = None,
        init_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        num_inference_steps: Optional[int] = None,
        eta: float = 1.0,
        return_raw: bool = False,
    ) -> MMDet3PipelineOutput:
        prepared = self._prepare_batch(data=data, batch_inputs=batch_inputs, batch_data_samples=batch_data_samples)
        detector = self.detector

        if num_inference_steps is None:
            num_inference_steps = int(getattr(detector, "sampling_timesteps", 1))
        num_inference_steps = max(int(num_inference_steps), 1)

        # Delegate sampling to detector if it provides a compatible method.
        if hasattr(detector, "sample"):
            sample_kwargs: Dict[str, Any] = dict(num_inference_steps=num_inference_steps, eta=float(eta))
            if init_state is not None:
                if not isinstance(init_state, tuple) or len(init_state) != 2:
                    raise ValueError("init_state must be a tuple: (boxes, label_state).")
                init_boxes, init_label_state = init_state
                sample_kwargs["init_boxes"] = init_boxes
                sample_kwargs["init_label_state"] = init_label_state

            try:
                out = detector.sample(prepared.batch_inputs, prepared.batch_data_samples, **sample_kwargs)
            except TypeError:
                out = detector.sample(prepared.batch_inputs, prepared.batch_data_samples, num_inference_steps=num_inference_steps, eta=float(eta))
            if not isinstance(out, dict) or "pred_boxes" not in out or "pred_logits" not in out:
                raise RuntimeError("detector.sample must return a dict with keys: pred_boxes, pred_logits, data_samples")
            result = MMDet3PipelineOutput()
            result.pred_boxes = out["pred_boxes"]
            result.pred_logits = out["pred_logits"]
            result.pred_quality = out.get("pred_quality", None)
            result.data_samples = out.get("data_samples", None)
            return result

        # Fallback: call detector.predict (MMDet postprocess path).
        data_samples = detector.predict(prepared.batch_inputs, prepared.batch_data_samples)
        result = MMDet3PipelineOutput()
        result.data_samples = data_samples
        if return_raw:
            # No raw tensors available from predict path.
            result.pred_boxes = None
            result.pred_logits = None
        return result
