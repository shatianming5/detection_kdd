"""
MMDetection 3.x plugins.

This subpackage is only meant to be imported inside a working MMDet3
environment (e.g. a dedicated conda env). It registers custom components
into MMEngine/MMDet registries.
"""

from .collate import coco_graph_collate  # noqa: F401
from .diffusion_detector import GraphDiffusionDetector  # noqa: F401

try:
    from .pipeline import MMDet3HybridDiffusionPipeline  # noqa: F401
except Exception:  # pragma: no cover
    MMDet3HybridDiffusionPipeline = None  # type: ignore

__all__ = ["GraphDiffusionDetector", "coco_graph_collate"]

if MMDet3HybridDiffusionPipeline is not None:
    __all__.append("MMDet3HybridDiffusionPipeline")
