import pytest
import torch
import bittensor as bt
from PIL import Image, ImageDraw
import numpy as np
from unittest.mock import MagicMock, patch
import random

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


def create_complex_image(size=(64, 64), num_blobs=3):
    image = Image.new(
        "RGB",
        size,
        color=(
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        ),
    )
    draw = ImageDraw.Draw(image)

    for _ in range(num_blobs):
        blob_color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )
        x = random.randint(0, size[0])
        y = random.randint(0, size[1])
        radius = random.randint(5, 20)
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius], fill=blob_color
        )

    return image


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
            torch.tensor(np.array(create_complex_image()))
            .permute(2, 0, 1)
            .float()
            / 255
        ]
        images2 = [
            torch.tensor(np.array(create_complex_image()))
            .permute(2, 0, 1)
            .float()
            / 255
        ]
        synapse1 = create_synapse("hotkey1", images1)
        synapse2 = create_synapse("hotkey2", images2)
        mask = await duplicate_filter.get_rewards(None, [synapse1, synapse2])
        print(f"Resulting mask: {mask}")
        print(f"Image1 hash: {duplicate_filter.compute_phash(images1[0])}")
        print(f"Image2 hash: {duplicate_filter.compute_phash(images2[0])}")
        print(
            f"Hash difference: {duplicate_filter.compute_phash(images1[0]) - duplicate_filter.compute_phash(images2[0])}"
        )
        print(
            f"Threshold: {int(duplicate_filter.hash_size * duplicate_filter.hash_size * duplicate_filter.threshold_ratio)}"
        )
        assert torch.allclose(mask, torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0]))


@pytest.mark.asyncio
async def test_with_duplicates(duplicate_filter, mock_metagraph):
    with patch(
        "neurons.validator.scoring.models.masks.duplicate.get_metagraph",
        return_value=mock_metagraph,
    ):
        image = create_complex_image()
        images1 = [torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255]
        images2 = [torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255]
        images3 = [
            torch.tensor(np.array(create_complex_image()))
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

        image1 = create_complex_image()
        image2 = image1.copy()
        # Slightly modify image2
        pixels = image2.load()
        for _ in range(10):
            x = random.randint(0, image2.width - 1)
            y = random.randint(0, image2.height - 1)
            pixels[x, y] = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )

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
