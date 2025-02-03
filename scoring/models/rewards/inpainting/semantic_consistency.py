import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

from neurons.protocol import BaseTask
from neurons.utils.image import synapse_to_images
from scoring.models.base import BaseRewardModel
from scoring.models.types import RewardModelType


from scoring.models.rewards.inpainting.base import BaseInpaintingModel

class SemanticConsistencyModel(BaseInpaintingModel):
    @property
    def name(self) -> RewardModelType:
        return RewardModelType.SEMANTIC_CONSISTENCY

    def __init__(self):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

    async def compute_score(
        self,
        input_image: torch.Tensor,
        mask_image: torch.Tensor, 
        generated_image: torch.Tensor
    ) -> float:

        try:
            # Process images through CLIP
            inputs = self.processor(
                images=[input_image, generated_image], 
                return_tensors="pt"
            ).to(input_image.device)

        # Get image embeddings
        image_features = self.clip.get_image_features(**inputs)

        # Calculate cosine similarity between embeddings
        similarity = F.cosine_similarity(
            image_features[0].unsqueeze(0), image_features[1].unsqueeze(0)
        )

        return similarity.item()
