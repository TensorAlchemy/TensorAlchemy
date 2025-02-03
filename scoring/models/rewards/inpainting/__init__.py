"""
This package contains specialized reward models for evaluating image inpainting quality.

Models:
- BoundaryCoherenceModel: Evaluates seamlessness of inpainted boundaries
- MaskAdherenceModel: Checks if changes are properly confined to masked region
- StructureConsistencyModel: Analyzes structural continuity across inpainted region
- SemanticConsistencyModel: Verifies semantic consistency between original and inpainted images
"""

from .boundary_coherence import BoundaryCoherenceModel
from .mask_adherence import MaskAdherenceModel
from .structure_consistency import StructureConsistencyModel
from .semantic_consistency import SemanticConsistencyModel

__all__ = [
    'BoundaryCoherenceModel',
    'MaskAdherenceModel', 
    'StructureConsistencyModel',
    'SemanticConsistencyModel'
]