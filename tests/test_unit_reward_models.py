import pytest
from unittest.mock import patch, MagicMock
import torch
import bittensor as bt

from neurons.protocol import ImageGeneration, ModelType
from neurons.validator.rewards.models.blacklist import BlacklistFilter
from neurons.validator.rewards.models.nsfw import NSFWRewardModel


@pytest.fixture
def mock_tensor():
    return MagicMock(spec=bt.Tensor)


@pytest.fixture
def mock_image_generation(mock_tensor):
    def _create_image_generation(images, height=1024, width=1024, prompt=""):
        return ImageGeneration(
            generation_type="TEXT_TO_IMAGE",
            seed=-1,
            model_type=ModelType.ALCHEMY.value,
            images=[mock_tensor for _ in range(images)],
            height=height,
            width=width,
            prompt=prompt,
            num_images_per_prompt=images,
        )

    return _create_image_generation


@pytest.fixture
def blacklist_filter():
    return BlacklistFilter()


@pytest.fixture
def nsfw_reward_model():
    with patch("neurons.validator.rewards.models.nsfw.StableDiffusionSafetyChecker"):
        with patch("neurons.validator.rewards.models.nsfw.CLIPImageProcessor"):
            return NSFWRewardModel()


@pytest.mark.asyncio
async def test_black_image(blacklist_filter, mock_image_generation):
    with patch("bittensor.Tensor.deserialize") as mock_deserialize:
        mock_deserialize.side_effect = [
            torch.full([3, 1024, 1024], 254, dtype=torch.float),
            torch.full([3, 1024, 1024], 0, dtype=torch.float),
        ]

        responses = [mock_image_generation(1), mock_image_generation(1)]

        # Add dendrite attribute with hotkey
        responses[0].dendrite = MagicMock(hotkey="hotkey_1")
        responses[1].dendrite = MagicMock(hotkey="hotkey_2")

        rewards = await blacklist_filter.get_rewards(responses[0], responses)

        assert rewards["hotkey_1"] == 1
        assert rewards["hotkey_2"] == 0


@pytest.mark.asyncio
async def test_incorrect_image_size(blacklist_filter, mock_image_generation):
    with patch("bittensor.Tensor.deserialize") as mock_deserialize:
        mock_deserialize.side_effect = [
            torch.full([3, 1024, 1024], 254, dtype=torch.float),
            torch.full([3, 100, 1024], 254, dtype=torch.float),
        ]

        responses = [
            mock_image_generation(1, height=1024, width=1024),
            mock_image_generation(1, height=100, width=1024),
        ]

        # Add dendrite attribute with hotkey
        responses[0].dendrite = MagicMock(hotkey="hotkey_1")
        responses[1].dendrite = MagicMock(hotkey="hotkey_2")

        rewards = await blacklist_filter.get_rewards(responses[0], responses)

        assert rewards["hotkey_1"] == 1
        assert rewards["hotkey_2"] == 0


@pytest.mark.asyncio
async def test_nsfw_image(nsfw_reward_model, mock_image_generation):
    with patch(
        "neurons.validator.rewards.models.nsfw.StableDiffusionSafetyChecker.forward"
    ) as mock_forward:
        mock_forward.return_value = (None, [True, False])

        responses = [
            mock_image_generation(1, prompt="An nsfw woman."),
            mock_image_generation(
                1, prompt="A majestic lion jumping from a big stone at night"
            ),
        ]

        # Add dendrite attribute with hotkey
        responses[0].dendrite = MagicMock(hotkey="hotkey_1")
        responses[1].dendrite = MagicMock(hotkey="hotkey_2")

        rewards = await nsfw_reward_model.get_rewards(responses[1], responses)

        print(rewards)

        assert rewards["hotkey_1"] == 0
        assert rewards["hotkey_2"] == 1


if __name__ == "__main__":
    pytest.main()
