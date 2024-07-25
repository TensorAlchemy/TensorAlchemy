import pytest
from unittest.mock import patch, MagicMock
import torch
import bittensor as bt
from neurons.validator.scoring.models.rewards.clip import ClipRewardModel
from neurons.protocol import ImageGeneration


@pytest.fixture
def clip_reward_model():
    with patch(
        "neurons.validator.scoring.models.rewards.clip.CLIPModel.from_pretrained"
    ), patch(
        "neurons.validator.scoring.models.rewards.clip.CLIPProcessor.from_pretrained"
    ):
        return ClipRewardModel()


@pytest.mark.asyncio
async def test_clip_reward_model(clip_reward_model):
    # Mock synapse
    synapse = MagicMock(spec=ImageGeneration)
    synapse.images = [torch.rand(3, 224, 224)]
    synapse.prompt = "A beautiful landscape"

    # Mock processor output
    processor_output = {
        "input_ids": torch.randint(0, 1000, (1, 10)),
        "attention_mask": torch.ones(1, 10),
        "pixel_values": torch.rand(1, 3, 224, 224),
    }

    # Mock model output
    model_output = MagicMock()
    model_output.logits_per_image = torch.tensor([[0.7]])

    # Patch methods
    with patch.object(
        clip_reward_model.processor, "__call__", return_value=processor_output
    ), patch.object(
        clip_reward_model.scoring_model, "__call__", return_value=model_output
    ):
        reward = clip_reward_model.get_reward(synapse)

    assert isinstance(reward, float)
    assert 0 <= reward <= 1


@pytest.mark.asyncio
async def test_clip_reward_model_no_images(clip_reward_model):
    synapse = MagicMock(spec=ImageGeneration)
    synapse.images = []

    reward = clip_reward_model.get_reward(synapse)

    assert reward == 0.0


@pytest.mark.asyncio
async def test_clip_reward_model_none_image(clip_reward_model):
    synapse = MagicMock(spec=ImageGeneration)
    synapse.images = [None]

    reward = clip_reward_model.get_reward(synapse)

    assert reward == 0.0


@pytest.mark.asyncio
async def test_clip_reward_model_exception(clip_reward_model):
    synapse = MagicMock(spec=ImageGeneration)
    synapse.images = [torch.rand(3, 224, 224)]
    synapse.prompt = "A beautiful landscape"

    with patch.object(
        clip_reward_model.processor,
        "__call__",
        side_effect=Exception("Test error"),
    ):
        reward = clip_reward_model.get_reward(synapse)

    assert reward == 0.0
