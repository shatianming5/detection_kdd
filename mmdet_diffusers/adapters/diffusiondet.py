from __future__ import annotations

from typing import Optional

import sys
from pathlib import Path

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling import build_model

from mmdet_diffusers.pipeline import HybridDiffusionPipeline


def _ensure_diffusiondet_importable() -> None:
    try:
        import diffusiondet  # noqa: F401
        return
    except Exception:
        repo_root = Path(__file__).resolve().parents[2]
        d2_path = repo_root / "baselines" / "DiffusionDet"
        sys.path.insert(0, str(d2_path))


def build_pipeline_from_detectron2_cfg(
    *,
    config_file: str,
    weights: str,
    device: Optional[str] = None,
) -> HybridDiffusionPipeline:
    """
    Minimal helper to build a DiffusionDet model and wrap it into HybridDiffusionPipeline.

    This is intentionally a thin convenience wrapper for check.md parity; training remains detectron2-based.
    """
    _ensure_diffusiondet_importable()
    from diffusiondet import add_diffusiondet_config  # local import after sys.path fix

    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = weights
    if device is not None:
        cfg.MODEL.DEVICE = str(device)
    cfg.freeze()

    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(weights)
    return HybridDiffusionPipeline.from_diffusiondet(model)
