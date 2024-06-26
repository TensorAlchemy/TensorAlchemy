import pytest
from unittest.mock import patch, MagicMock
from io import BytesIO
import requests

import torch
import bittensor as bt
import torchvision.transforms as transforms
from PIL import Image


from neurons.protocol import ImageGeneration, ModelType
from neurons.validator.rewards.models.blacklist import BlacklistFilter
from neurons.validator.rewards.models.nsfw import NSFWRewardModel


@pytest.fixture
def nsfw_reward_model():
    return NSFWRewardModel()


@pytest.fixture
def mock_tensor():
    return MagicMock(spec=bt.Tensor)


@pytest.fixture
def blacklist_filter():
    return BlacklistFilter()


@pytest.fixture
def mock_image_generation():
    def _create_image_generation(images, height=1024, width=1024, prompt=""):
        synapse = ImageGeneration(
            generation_type="TEXT_TO_IMAGE",
            seed=-1,
            model_type=ModelType.ALCHEMY.value,
            images=[MagicMock(spec=bt.Tensor) for _ in range(images)],
            height=height,
            width=width,
            prompt=prompt,
            num_images_per_prompt=images,
        )
        synapse.dendrite = MagicMock()
        return synapse

    return _create_image_generation


def create_mock_synapse(images, height, width, hotkey):
    synapse = ImageGeneration(
        generation_type="TEXT_TO_IMAGE",
        seed=-1,
        model_type=ModelType.ALCHEMY.value,
        images=images,
        height=height,
        width=width,
        num_images_per_prompt=len(images),
    )
    synapse.dendrite = bt.TerminalInfo(hotkey=hotkey)
    return synapse


@pytest.mark.asyncio
async def test_black_image(blacklist_filter):
    normal_image = bt.Tensor.serialize(
        torch.full([3, 1024, 1024], 254, dtype=torch.float)
    )
    black_image = bt.Tensor.serialize(torch.full([3, 1024, 1024], 0, dtype=torch.float))

    responses = [
        create_mock_synapse([normal_image], 1024, 1024, "hotkey_1"),
        create_mock_synapse([black_image], 1024, 1024, "hotkey_2"),
    ]

    rewards = await blacklist_filter.get_rewards(responses[0], responses)

    assert rewards["hotkey_1"] == 1.0
    assert rewards["hotkey_2"] == 0.0


@pytest.mark.asyncio
async def test_incorrect_image_size(blacklist_filter):
    correct_size_image = bt.Tensor.serialize(
        torch.full([3, 1024, 1024], 254, dtype=torch.float)
    )
    incorrect_size_image = bt.Tensor.serialize(
        torch.full([3, 100, 1024], 254, dtype=torch.float)
    )

    responses = [
        create_mock_synapse([correct_size_image], 1024, 1024, "hotkey_1"),
        create_mock_synapse([incorrect_size_image], 100, 1024, "hotkey_2"),
    ]

    rewards = await blacklist_filter.get_rewards(responses[0], responses)

    assert rewards["hotkey_1"] == 1.0
    assert rewards["hotkey_2"] == 0.0


@pytest.mark.asyncio
async def test_nsfw_image(nsfw_reward_model):
    nsfw_image_url = "https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/a05eaa75-ac8c-4460-b6b0-b7eb47e06987/width=1024/00027-4120052916.jpeg"
    transform = transforms.Compose([transforms.PILToTensor()])

    response_nsfw = ImageGeneration(
        generation_type="TEXT_TO_IMAGE",
        seed=-1,
        model_type=ModelType.ALCHEMY.value,
        prompt="An nsfw woman.",
        images=[
            bt.Tensor.serialize(
                transform(
                    Image.open(
                        BytesIO(
                            requests.get(nsfw_image_url).content,
                        )
                    )
                )
            )
        ],
    )
    response_nsfw.dendrite = bt.TerminalInfo(hotkey="hotkey_1")

    response_no_nsfw = ImageGeneration(
        generation_type="TEXT_TO_IMAGE",
        seed=-1,
        model_type=ModelType.ALCHEMY.value,
        prompt="A majestic lion jumping from a big stone at night",
        images=[
            bt.Tensor.serialize(
                transform(
                    Image.open(
                        r"tests/non_nsfw.jpeg",
                    )
                )
            )
        ],
    )
    response_no_nsfw.dendrite = bt.TerminalInfo(hotkey="hotkey_2")

    responses = [response_nsfw, response_no_nsfw]

    rewards = await nsfw_reward_model.get_rewards(response_no_nsfw, responses)

    print(rewards)
    assert rewards["hotkey_1"] == 0.0  # NSFW image should get 0.0 reward
    assert rewards["hotkey_2"] == 1.0  # Non-NSFW image should get 1.0 reward

    # Additional checks
    assert len(rewards) == 2  # Ensure we have rewards for both responses
    assert all(
        isinstance(value, float) for value in rewards.values()
    )  # Ensure all rewards are floats
    assert all(
        0 <= value <= 1 for value in rewards.values()
    )  # Ensure all rewards are between 0 and 1
