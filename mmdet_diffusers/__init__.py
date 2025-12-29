from .collate import collate_coco_graph
from .schedulers import D3PMLabelScheduler, build_ddim_scheduler_from_betas

try:
    from .pipeline import HybridDiffusionPipeline  # noqa: F401
except Exception:  # pragma: no cover
    # Keep the package importable in minimal environments (e.g. a MMDet3 conda env)
    # where `diffusers` is not installed.
    HybridDiffusionPipeline = None  # type: ignore

__all__ = [
    "D3PMLabelScheduler",
    "build_ddim_scheduler_from_betas",
    "collate_coco_graph",
]

if HybridDiffusionPipeline is not None:
    __all__.append("HybridDiffusionPipeline")
