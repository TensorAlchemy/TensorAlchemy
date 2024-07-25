import pytest
import torch
import bittensor as bt
from PIL import Image
import numpy as np
from unittest.mock import MagicMock, patch

from neurons.validator.scoring.models.masks.duplicate import DuplicateFilter
from neurons.validator.config import get_metagraph


@pytest.fixture
def mock_metagraph():
    metagraph = MagicMock()
    metagraph.n = 5
    metagraph.hotkeys = ["hotkey1", "hotkey2", "hotkey3", "hotkey4", "hotkey5"]
    return metagraph


@pytest.fixture
def duplicate_filter():
    return DuplicateFilter()


def create_image(color):
    return Image.new("RGB", (64, 64), color=color)


def create_synapse(hotkey: str, images):
    synapse = MagicMock(spec=bt.Synapse)
    synapse.images = images
    synapse.axon = bt.TerminalInfo(hotkey=hotkey)
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

        synapse1 = create_synapse("hotkey1", images1)
        synapse2 = create_synapse("hotkey2", images2)

        mask = await duplicate_filter.get_rewards(None, [synapse1, synapse2])

        assert torch.allclose(mask, torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0]))


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

        synapse1 = create_synapse("hotkey1", images1)
        synapse2 = create_synapse("hotkey2", images2)
        synapse3 = create_synapse("hotkey3", images3)

        mask = await duplicate_filter.get_rewards(
            None, [synapse1, synapse2, synapse3]
        )

        assert torch.allclose(mask, torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0]))


@pytest.mark.asyncio
async def test_slight_modification(mock_metagraph):
    with patch(
        "neurons.validator.scoring.models.masks.duplicate.get_metagraph",
        return_value=mock_metagraph,
    ):
        duplicate_filter = DuplicateFilter()

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

        synapse1 = create_synapse("hotkey1", images1)
        synapse2 = create_synapse("hotkey2", images2)

        mask = await duplicate_filter.get_rewards(None, [synapse1, synapse2])

        assert torch.allclose(
            mask, torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0])
        )  # Both should be considered duplicates


@pytest.mark.asyncio
async def test_empty_responses(duplicate_filter, mock_metagraph):
    with patch(
        "neurons.validator.scoring.models.masks.duplicate.get_metagraph",
        return_value=mock_metagraph,
    ):
        mask = await duplicate_filter.get_rewards(None, [])

        assert torch.allclose(mask, torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0]))


@pytest.mark.asyncio
async def test_invalid_responses(duplicate_filter, mock_metagraph):
    with patch(
        "neurons.validator.scoring.models.masks.duplicate.get_metagraph",
        return_value=mock_metagraph,
    ):
        synapse1 = create_synapse("hotkey1", [])
        synapse2 = create_synapse("hotkey2", [None])

        mask = await duplicate_filter.get_rewards(None, [synapse1, synapse2])

        assert torch.allclose(mask, torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0]))


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

        synapse1 = create_synapse("hotkey1", images1)
        synapse2 = create_synapse("hotkey2", [])
        synapse3 = create_synapse("hotkey3", images2)

        mask = await duplicate_filter.get_rewards(
            None, [synapse1, synapse2, synapse3]
        )

        assert torch.allclose(mask, torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0]))
