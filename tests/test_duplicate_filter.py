import pytest
import torch
import bittensor as bt
from PIL import Image, ImageDraw, ImageEnhance
import numpy as np
from unittest.mock import MagicMock, patch
import random

from loguru import logger

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


def apply_fixed_transform(image, seed: int = 10):
    random.seed(seed)  # Use a fixed seed for reproducibility
    transforms = [
        lambda img: img.rotate(random.uniform(-5, 5)),
        lambda img: ImageEnhance.Brightness(img).enhance(
            random.uniform(0.8, 1.2)
        ),
        lambda img: ImageEnhance.Contrast(img).enhance(
            random.uniform(0.8, 1.2)
        ),
    ]
    return random.choice(transforms)(image)


@pytest.mark.asyncio
async def test_multiple_images_per_synapse(duplicate_filter, mock_metagraph):
    with patch(
        "neurons.validator.scoring.models.masks.duplicate.get_metagraph",
        return_value=mock_metagraph,
    ):
        images1 = [
            torch.tensor(np.array(create_complex_image()))
            .permute(2, 0, 1)
            .float()
            / 255
            for _ in range(3)
        ]
        images2 = [
            torch.tensor(np.array(create_complex_image()))
            .permute(2, 0, 1)
            .float()
            / 255
            for _ in range(3)
        ]
        synapse1 = create_synapse("hotkey1", images1)
        synapse2 = create_synapse("hotkey2", images2)

        mask = await duplicate_filter.get_rewards(None, [synapse1, synapse2])

        assert torch.allclose(mask, torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0]))


@pytest.mark.asyncio
async def test_partial_duplicates(duplicate_filter, mock_metagraph):
    with patch(
        "neurons.validator.scoring.models.masks.duplicate.get_metagraph",
        return_value=mock_metagraph,
    ):
        shared_image = create_complex_image()
        unique_image1 = create_complex_image()
        unique_image2 = create_complex_image()

        images1 = [
            torch.tensor(np.array(shared_image)).permute(2, 0, 1).float() / 255,
            torch.tensor(np.array(unique_image1)).permute(2, 0, 1).float()
            / 255,
        ]
        images2 = [
            torch.tensor(np.array(shared_image)).permute(2, 0, 1).float() / 255,
            torch.tensor(np.array(unique_image2)).permute(2, 0, 1).float()
            / 255,
        ]
        synapse1 = create_synapse("hotkey1", images1)
        synapse2 = create_synapse("hotkey2", images2)

        mask = await duplicate_filter.get_rewards(None, [synapse1, synapse2])

        assert torch.allclose(mask, torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0]))


@pytest.mark.asyncio
async def test_transformed_duplicates(mock_metagraph):
    with patch(
        "neurons.validator.scoring.models.masks.duplicate.get_metagraph",
        return_value=mock_metagraph,
    ):
        duplicate_filter = DuplicateFilter(
            hash_size=8, threshold_ratio=0.1
        )  # Using default values

        original_image = create_complex_image()
        transformed_image = apply_fixed_transform(original_image)

        images1 = [
            torch.tensor(np.array(original_image)).permute(2, 0, 1).float()
            / 255
        ]
        images2 = [
            torch.tensor(np.array(transformed_image)).permute(2, 0, 1).float()
            / 255
        ]

        synapse1 = create_synapse("hotkey1", images1)
        synapse2 = create_synapse("hotkey2", images2)

        # Log hash information for debugging
        original_hash = duplicate_filter.compute_phash(images1[0])
        transformed_hash = duplicate_filter.compute_phash(images2[0])
        hash_difference = original_hash - transformed_hash
        threshold = int(
            duplicate_filter.hash_size
            * duplicate_filter.hash_size
            * duplicate_filter.threshold_ratio
        )

        logger.info(f"Original hash: {original_hash}")
        logger.info(f"Transformed hash: {transformed_hash}")
        logger.info(f"Hash difference: {hash_difference}")
        logger.info(f"Threshold: {threshold}")

        mask = await duplicate_filter.get_rewards(None, [synapse1, synapse2])

        logger.info(f"Resulting mask: {mask}")

        assert torch.allclose(mask, torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0]))


@pytest.mark.asyncio
async def test_different_resolutions(duplicate_filter, mock_metagraph):
    with patch(
        "neurons.validator.scoring.models.masks.duplicate.get_metagraph",
        return_value=mock_metagraph,
    ):
        original_image = create_complex_image(size=(64, 64))
        resized_image = original_image.resize((128, 128))

        images1 = [
            torch.tensor(np.array(original_image)).permute(2, 0, 1).float()
            / 255
        ]
        images2 = [
            torch.tensor(np.array(resized_image)).permute(2, 0, 1).float() / 255
        ]

        synapse1 = create_synapse("hotkey1", images1)
        synapse2 = create_synapse("hotkey2", images2)

        mask = await duplicate_filter.get_rewards(None, [synapse1, synapse2])

        assert torch.allclose(mask, torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0]))


@pytest.mark.asyncio
async def test_multiple_duplicates(duplicate_filter, mock_metagraph):
    with patch(
        "neurons.validator.scoring.models.masks.duplicate.get_metagraph",
        return_value=mock_metagraph,
    ):
        image1 = create_complex_image()
        image2 = create_complex_image()

        synapse1 = create_synapse(
            "hotkey1",
            [torch.tensor(np.array(image1)).permute(2, 0, 1).float() / 255],
        )
        synapse2 = create_synapse(
            "hotkey2",
            [torch.tensor(np.array(image1)).permute(2, 0, 1).float() / 255],
        )
        synapse3 = create_synapse(
            "hotkey3",
            [torch.tensor(np.array(image2)).permute(2, 0, 1).float() / 255],
        )
        synapse4 = create_synapse(
            "hotkey4",
            [torch.tensor(np.array(image2)).permute(2, 0, 1).float() / 255],
        )
        synapse5 = create_synapse(
            "hotkey5",
            [
                torch.tensor(np.array(create_complex_image()))
                .permute(2, 0, 1)
                .float()
                / 255
            ],
        )

        mask = await duplicate_filter.get_rewards(
            None, [synapse1, synapse2, synapse3, synapse4, synapse5]
        )

        assert torch.allclose(mask, torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0]))


def load_and_preprocess_image(file_path):
    image = Image.open(file_path).convert("RGB")
    return torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255


@pytest.mark.asyncio
async def test_specific_different_images(duplicate_filter, mock_metagraph):
    with patch(
        "neurons.validator.scoring.models.masks.duplicate.get_metagraph",
        return_value=mock_metagraph,
    ):
        # Load the specific images
        image_a = load_and_preprocess_image("tests/images/diff_a.png")
        image_b = load_and_preprocess_image("tests/images/diff_b.png")

        synapse1 = create_synapse("hotkey1", [image_a])
        synapse2 = create_synapse("hotkey2", [image_b])

        mask = await duplicate_filter.get_rewards(None, [synapse1, synapse2])

        # Assert that the images are considered different (mask should be all zeros)
        assert torch.allclose(mask, torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0]))

        # Additional assertion to ensure the hashes are different
        assert duplicate_filter.compute_phash(
            image_a
        ) != duplicate_filter.compute_phash(image_b)
