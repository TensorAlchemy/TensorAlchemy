import pytest
import torch
import bittensor as bt
from PIL import Image
import numpy as np
from unittest.mock import MagicMock, patch

from neurons.validator.scoring.models.masks.duplicate import DuplicateFilter
from neurons.validator.config import get_metagraph


# Mock the get_metagraph function
@pytest.fixture
def mock_metagraph():
    metagraph = MagicMock()
    metagraph.n = 5
    metagraph.hotkeys = ["hotkey1", "hotkey2", "hotkey3", "hotkey4", "hotkey5"]
    return metagraph


@pytest.fixture
def duplicate_filter():
    return DuplicateFilter(hash_size=8, threshold=5)


def create_image(color):
    return Image.new("RGB", (100, 100), color=color)


def create_synapse(images):
    synapse = MagicMock(spec=bt.Synapse)
    synapse.images = images
    synapse.axon = bt.TerminalInfo(hotkey=f"hotkey{id(synapse)}")
    return synapse


@pytest.mark.asyncio
async def test_no_duplicates(duplicate_filter, mock_metagraph):
    with patch(
        "neurons.validator.scoring.models.masks.duplicate.get_metagraph",
        return_value=mock_metagraph,
    ):
        images1 = [
            torch.tensor(np.array(create_image("red"))).permute(2, 0, 1).float()
            / 255
        ]
        images2 = [
            torch.tensor(np.array(create_image("blue")))
            .permute(2, 0, 1)
            .float()
            / 255
        ]

        synapse1 = create_synapse(images1)
        synapse2 = create_synapse(images2)

        rewards = await duplicate_filter.get_rewards(None, [synapse1, synapse2])

        assert torch.allclose(rewards, torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0]))


@pytest.mark.asyncio
async def test_with_duplicates(duplicate_filter, mock_metagraph):
    with patch(
        "neurons.validator.scoring.models.masks.duplicate.get_metagraph",
        return_value=mock_metagraph,
    ):
        images1 = [
            torch.tensor(np.array(create_image("red"))).permute(2, 0, 1).float()
            / 255
        ]
        images2 = [
            torch.tensor(np.array(create_image("red"))).permute(2, 0, 1).float()
            / 255
        ]
        images3 = [
            torch.tensor(np.array(create_image("blue")))
            .permute(2, 0, 1)
            .float()
            / 255
        ]

        synapse1 = create_synapse(images1)
        synapse2 = create_synapse(images2)
        synapse3 = create_synapse(images3)

        rewards = await duplicate_filter.get_rewards(
            None, [synapse1, synapse2, synapse3]
        )

        assert torch.allclose(rewards, torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0]))


@pytest.mark.asyncio
async def test_slight_modification(duplicate_filter, mock_metagraph):
    with patch(
        "neurons.validator.scoring.models.masks.duplicate.get_metagraph",
        return_value=mock_metagraph,
    ):
        image1 = create_image("red")
        image2 = image1.copy()
        # Slightly modify image2
        pixels = image2.load()
        for i in range(10):
            for j in range(10):
                pixels[i, j] = (
                    255,
                    0,
                    0,
                )  # Make a small red square slightly brighter

        images1 = [
            torch.tensor(np.array(image1)).permute(2, 0, 1).float() / 255
        ]
        images2 = [
            torch.tensor(np.array(image2)).permute(2, 0, 1).float() / 255
        ]

        synapse1 = create_synapse(images1)
        synapse2 = create_synapse(images2)

        rewards = await duplicate_filter.get_rewards(None, [synapse1, synapse2])

        assert torch.allclose(
            rewards, torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
        )  # Both should be considered duplicates


@pytest.mark.asyncio
async def test_empty_responses(duplicate_filter, mock_metagraph):
    with patch(
        "neurons.validator.scoring.models.masks.duplicate.get_metagraph",
        return_value=mock_metagraph,
    ):
        rewards = await duplicate_filter.get_rewards(None, [])

        assert torch.allclose(rewards, torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0]))


@pytest.mark.asyncio
async def test_invalid_responses(duplicate_filter, mock_metagraph):
    with patch(
        "neurons.validator.scoring.models.masks.duplicate.get_metagraph",
        return_value=mock_metagraph,
    ):
        synapse1 = create_synapse([])
        synapse2 = create_synapse([None])

        rewards = await duplicate_filter.get_rewards(None, [synapse1, synapse2])

        assert torch.allclose(rewards, torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0]))


@pytest.mark.asyncio
async def test_mixed_valid_invalid_responses(duplicate_filter, mock_metagraph):
    with patch(
        "neurons.validator.scoring.models.masks.duplicate.get_metagraph",
        return_value=mock_metagraph,
    ):
        images1 = [
            torch.tensor(np.array(create_image("red"))).permute(2, 0, 1).float()
            / 255
        ]
        images2 = [
            torch.tensor(np.array(create_image("blue")))
            .permute(2, 0, 1)
            .float()
            / 255
        ]

        synapse1 = create_synapse(images1)
        synapse2 = create_synapse([])
        synapse3 = create_synapse(images2)

        rewards = await duplicate_filter.get_rewards(
            None, [synapse1, synapse2, synapse3]
        )

        assert torch.allclose(rewards, torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0]))
