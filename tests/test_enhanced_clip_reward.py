from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
from loguru import logger

from scoring.models.rewards.enhanced_clip import EnhancedClipRewardModel

# Import the mock functions and fixtures
from tests.fixtures import TEST_IMAGES, generate_synapse, mock_get_metagraph

# Define the mock configurations
mock_configs = {
    "neurons.config": {
        "get_metagraph": mock_get_metagraph,
        "get_openai_client": MagicMock(),
        "get_corcel_api_key": MagicMock(return_value="mock_api_key"),
    },
    "scoring.models.base": {"get_metagraph": mock_get_metagraph},
    "scoring.models.rewards.enhanced_clip.utils": {
        "openai_breakdown": AsyncMock(),
    },
}


# Define the patch decorator
def apply_patches(func):
    for module, mocks in mock_configs.items():
        func = patch.multiple(module, **mocks)(func)
    return func


@pytest.fixture
def mock_openai_response():
    return {
        "elements": [
            {"description": "Something"},
        ]
    }


@pytest.fixture
def mock_openai_response_sparrow():
    return {
        "elements": [
            {"description": "Sparrow"},
            {"description": "Fish"},
            {"description": "Soaring"},
            {"description": "Ocean"},
            {"description": "Bird"},
        ]
    }


@pytest.fixture
def mock_openai_response_elephant():
    return {
        "elements": [
            {"description": "Basket"},
            {"description": "Elephant"},
            {"description": "Desert"},
            {"description": "Elephant"},
        ]
    }


@pytest.fixture
@apply_patches
def patched_model(mock_openai_response):
    mock_configs["scoring.models.rewards.enhanced_clip.utils"][
        "openai_breakdown"
    ].return_value = mock_openai_response
    model = EnhancedClipRewardModel()
    model.device = "cpu"
    return model


@pytest.fixture
def mock_synapse():
    return generate_synapse("hotkey_0", TEST_IMAGES["COMPLEX_A"])


@apply_patches
class TestEnhancedClipRewardModel:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "image_a_key, image_b_key, prompt, mock_response_fixture",
        [
            (
                "SPARROW_FISH",
                "ELEPHANT_BASKET",
                "A tiny sparrow, carrying a fish in its beak, soars over a wide ocean.",
                "mock_openai_response_sparrow",
            ),
            (
                "ELEPHANT_BASKET",
                "SPARROW_FISH",
                "A huge elephant, carrying a basket in its trunk, walks the a wide desert.",
                "mock_openai_response_elephant",
            ),
        ],
    )
    async def test_correct_incorrect(
        self,
        patched_model,
        image_a_key,
        image_b_key,
        prompt,
        mock_response_fixture,
        request,
    ):
        # Get the appropriate mock response
        mock_openai_response = request.getfixturevalue(mock_response_fixture)

        mock_configs["scoring.models.rewards.enhanced_clip.utils"][
            "openai_breakdown"
        ].return_value = mock_openai_response

        right_synapse = generate_synapse(
            "hotkey_0",
            TEST_IMAGES[image_a_key],
            prompt=prompt,
        )

        wrong_synapse = generate_synapse(
            "hotkey_1",
            TEST_IMAGES[image_b_key],
            prompt=prompt,
        )

        rewards = await patched_model.get_rewards(
            right_synapse,
            [right_synapse, wrong_synapse],
        )

        logger.info(f"Right reward: {rewards[0].item()}")
        logger.info(f"Wrong reward: {rewards[1].item()}")

        assert isinstance(rewards, torch.Tensor)
        assert rewards.shape == (10,)
        assert rewards[0] > rewards[1], (
            f"Correct image reward ({rewards[0].item()}) "
            + f"should be higher than Incorrect image reward ({rewards[1].item()})"
        )

    @pytest.mark.asyncio
    async def test_valid_image(self, patched_model, mock_synapse):
        with patch.object(
            patched_model, "compute_clip_score", return_value=0.7
        ):
            reward = await patched_model.get_rewards(
                mock_synapse, [mock_synapse]
            )
        assert isinstance(reward, torch.Tensor)
        assert reward.shape == (10,)
        assert all(0 <= r <= 1 for r in reward)

    @pytest.mark.asyncio
    async def test_no_images(self, patched_model, mock_synapse):
        mock_synapse.images = []
        reward = await patched_model.get_rewards(mock_synapse, [mock_synapse])
        assert isinstance(reward, torch.Tensor)
        assert reward.shape == (10,)
        assert all(r == 0.0 for r in reward)

    @pytest.mark.asyncio
    async def test_multiple_images(self, patched_model, mock_synapse):
        synapse1 = generate_synapse("hotkey_1", TEST_IMAGES["COMPLEX_B"])
        synapse2 = generate_synapse("hotkey_2", TEST_IMAGES["COMPLEX_C"])
        with patch.object(
            patched_model, "compute_clip_score", side_effect=[0.7, 0.8]
        ):
            reward = await patched_model.get_rewards(
                mock_synapse, [synapse1, synapse2]
            )
        assert isinstance(reward, torch.Tensor)
        assert reward.shape == (10,)
        assert all(0 <= r <= 1 for r in reward)

    @pytest.mark.asyncio
    async def test_empty_prompt(self, patched_model, mock_synapse):
        mock_synapse.prompt = ""
        with patch.object(
            patched_model, "compute_clip_score", return_value=0.5
        ):
            reward = await patched_model.get_rewards(
                mock_synapse, [mock_synapse]
            )
        assert isinstance(reward, torch.Tensor)
        assert reward.shape == (10,)
        assert all(0 <= r <= 1 for r in reward)

    @pytest.mark.asyncio
    async def test_different_image_sizes(self, patched_model, mock_synapse):
        synapse1 = generate_synapse("hotkey_1", TEST_IMAGES["COMPLEX_D"])
        synapse2 = generate_synapse("hotkey_2", TEST_IMAGES["COMPLEX_E"])
        with patch.object(
            patched_model, "compute_clip_score", side_effect=[0.7, 0.9]
        ):
            reward = await patched_model.get_rewards(
                mock_synapse, [synapse1, synapse2]
            )
        assert isinstance(reward, torch.Tensor)
        assert reward.shape == (10,)
        assert all(0 <= r <= 1 for r in reward)
