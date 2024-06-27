import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from functools import wraps

import torch
import bittensor as bt

from neurons.protocol import ImageGeneration, ModelType
from neurons.validator.rewards.models.blacklist import BlacklistFilter
from neurons.validator.rewards.models.nsfw import NSFWRewardModel


# Mock functions and classes
def mock_metagraph():
    mock = MagicMock()
    mock.hotkeys = [f"hotkey_{i}" for i in range(5)]
    mock.n = 5
    return mock


# Create instances of our mocks
mock_meta = mock_metagraph()


@pytest.fixture
def nsfw_reward_model():
    return NSFWRewardModel()


@pytest.fixture
def blacklist_filter():
    return BlacklistFilter()


def create_mock_synapse(images, height, width, hotkey):
    synapse = ImageGeneration(
        seed=-1,
        width=width,
        images=images,
        height=height,
        generation_type="TEXT_TO_IMAGE",
        model_type=ModelType.ALCHEMY.value,
        num_images_per_prompt=len(images),
    )
    synapse.dendrite = bt.TerminalInfo(hotkey=hotkey)
    return synapse


@pytest.mark.asyncio
@patch(
    "neurons.validator.rewards.models.base.get_metagraph",
    return_value=mock_meta,
)
async def test_black_image(mock_meta, blacklist_filter):
    normal_image = bt.Tensor.serialize(
        torch.full([3, 1024, 1024], 255, dtype=torch.float)
    )
    black_image = bt.Tensor.serialize(torch.full([3, 1024, 1024], 0, dtype=torch.float))

    responses = [
        create_mock_synapse([normal_image], 1024, 1024, "hotkey_1"),
        create_mock_synapse([black_image], 1024, 1024, "hotkey_2"),
    ]

    rewards = await blacklist_filter.get_rewards(responses[0], responses)

    assert rewards[0].item() == 0.0  # Normal image should not be blacklisted
    assert rewards[1].item() == 1.0  # Black image should be blacklisted


@pytest.mark.asyncio
@patch(
    "neurons.validator.rewards.models.base.get_metagraph",
    return_value=mock_meta,
)
async def test_incorrect_image_size(mock_meta, blacklist_filter):
    correct_size_image = bt.Tensor.serialize(
        torch.full([3, 1024, 1024], 255, dtype=torch.float)
    )
    incorrect_size_image = bt.Tensor.serialize(
        torch.full([3, 100, 1024], 255, dtype=torch.float)
    )

    responses = [
        create_mock_synapse([correct_size_image], 1024, 1024, "hotkey_1"),
        create_mock_synapse([incorrect_size_image], 100, 1024, "hotkey_2"),
    ]

    rewards = await blacklist_filter.get_rewards(responses[0], responses)

    assert rewards[0].item() == 0.0  # Correct size image should not be blacklisted
    assert rewards[1].item() == 1.0  # Incorrect size image should be blacklisted


@pytest.mark.asyncio
@patch(
    "neurons.validator.rewards.models.base.get_metagraph",
    return_value=mock_meta,
)
async def test_nsfw_image(mock_meta, nsfw_reward_model):
    nsfw_image = bt.Tensor.serialize(torch.rand(3, 512, 512))
    safe_image = bt.Tensor.serialize(torch.rand(3, 512, 512))

    response_nsfw = create_mock_synapse([nsfw_image], 512, 512, "hotkey_1")
    response_safe = create_mock_synapse([safe_image], 512, 512, "hotkey_2")

    responses = [response_nsfw, response_safe]

    rewards = await nsfw_reward_model.get_rewards(responses[0], responses)

    assert rewards[0].item() == 1.0  # NSFW image should be flagged
    assert rewards[1].item() == 0.0  # Safe image should not be flagged

    assert (
        rewards.shape[0] == 5
    )  # Ensure we have rewards for all hotkeys in the mock metagraph
    assert torch.all(
        (rewards == 0) | (rewards == 1)
    )  # Ensure all rewards are either 0 or 1
