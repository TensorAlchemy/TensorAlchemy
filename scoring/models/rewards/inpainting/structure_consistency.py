import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16

from neurons.protocol import BaseTask
from neurons.utils.image import synapse_to_tensors
from scoring.models.base import BaseRewardModel
from scoring.models.rewards.inpainting.base import BaseInpaintingModel
from scoring.models.types import RewardModelType


class StructureConsistencyModel(BaseInpaintingModel):
    @property
    def name(self) -> RewardModelType:
        return RewardModelType.STRUCTURE_CONSISTENCY

    def __init__(self):
        super().__init__()
        # Load pre-trained VGG for feature extraction
        vgg = vgg16(pretrained=True)
        self.features = nn.Sequential(
            *list(vgg.features.children())[:16]
        ).eval()

        # Freeze parameters
        for param in self.features.parameters():
            param.requires_grad = False

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

    async def compute_score(
        self,
        input_image: torch.Tensor,
        mask_image: torch.Tensor,
        generated_image: torch.Tensor,
    ) -> float:

        # Extract deep features
        original_features = self.extract_features(input_image.unsqueeze(0))
        generated_features = self.extract_features(generated_image.unsqueeze(0))

        # Upsample mask to match feature dimensions
        mask_resized = F.interpolate(
            mask_image.unsqueeze(0),
            size=original_features.shape[2:],
            mode="nearest",
        )

        # Calculate feature consistency in and around masked region
        feature_diff = torch.abs(original_features - generated_features)

        # Weight differences more heavily near mask boundaries
        boundary_weight = F.conv2d(
            mask_resized,
            torch.ones(1, 1, 3, 3).to(mask_resized.device),
            padding=1,
        )
        boundary_weight = boundary_weight / boundary_weight.max()

        weighted_diff = feature_diff * boundary_weight
        consistency_score = 1.0 - torch.mean(weighted_diff).item()

        return consistency_score
