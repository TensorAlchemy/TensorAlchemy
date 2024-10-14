import pytest
from unittest.mock import patch, MagicMock
import torch
from loguru import logger
from functools import wraps
from typing import Dict

from neurons.validator.forward import (
    update_moving_averages,
)
from scoring.types import ScoringResults
from neurons.constants import MOVING_AVERAGE_ALPHA
from tests.fixtures import mock_get_config


def mock_metagraph():
    mock = MagicMock()
    mock.hotkeys = [f"hotkey_{i}" for i in range(256)]
    mock.coldkeys = [f"coldkey_{i}" for i in range(256)]
    mock.n = 256
    return mock


def mock_backend_client():
    class FakeBackendClient:
        async def post_moving_averages(self, *args, **kwargs):
            logger.info("[fake] posting moving averages...")

    return FakeBackendClient()


# Patch configuration
mock_configs = {
    "neurons.validator.forward": {
        "get_config": mock_get_config,
        "get_metagraph": mock_metagraph,
        "get_backend_client": mock_backend_client,
    },
    "neurons.validator.averages": {
        "get_config": mock_get_config,
        "get_metagraph": mock_metagraph,
        "get_backend_client": mock_backend_client,
    },
}


# Custom decorator to apply all patches
def patch_all_dependencies(func):
    for module, mocks in mock_configs.items():
        func = patch.multiple(module, **mocks)(func)

    return func


def dict_to_tensor(rewards_dict: Dict[str, float], n: int) -> torch.FloatTensor:
    rewards_tensor = torch.zeros(n)
    for key, value in rewards_dict.items():
        index = int(key.split("_")[1])
        if index < n:
            rewards_tensor[index] = value
    return rewards_tensor


@pytest.mark.asyncio
@patch_all_dependencies
async def test_alpha_respected(*args):
    moving_average_scores = torch.zeros(256)
    rewards = {"hotkey_39": 1.0}  # Set a high reward for one hotkey
    rewards_tensor = dict_to_tensor(rewards, 256)
    uids = torch.tensor([39])  # Only update the score for hotkey_39

    # Run update_moving_averages multiple times
    for _ in range(5):
        scoring_results = ScoringResults(
            combined_scores=rewards_tensor, combined_uids=uids
        )
        moving_average_scores = await update_moving_averages(
            moving_average_scores, scoring_results, alpha=MOVING_AVERAGE_ALPHA
        )

    # Check if the change respects the alpha value
    expected_value = (
        1 - (1 - MOVING_AVERAGE_ALPHA) ** 5
    )  # Theoretical value after 5 updates
    actual_value = moving_average_scores[39].item()

    assert (
        abs(actual_value - expected_value) < 1e-2
    ), f"Expected ~{expected_value}, but got {actual_value}"


@pytest.mark.asyncio
@patch_all_dependencies
async def test_moving_average_calculation(*args):
    previous_ma_scores = torch.zeros(256)
    rewards = {"hotkey_0": 1.0}
    rewards_tensor = dict_to_tensor(rewards, 256)
    uids = torch.tensor([0])
    scoring_results = ScoringResults(
        combined_scores=rewards_tensor, combined_uids=uids
    )

    updated_ma_scores = await update_moving_averages(
        previous_ma_scores, scoring_results
    )

    expected_score = MOVING_AVERAGE_ALPHA * 1.0  # Since previous score was 0
    assert torch.isclose(
        updated_ma_scores[0], torch.tensor(expected_score)
    ), f"Expected score to be close to {expected_score}, but got {updated_ma_scores[0].item()}"


@pytest.mark.asyncio
@patch_all_dependencies
async def test_non_zero_moving_averages(*args):
    moving_average_scores = torch.zeros(256)
    rewards = {
        "hotkey_39": 0.6522690057754517,
        "hotkey_34": 0.7715857625007629,
        "hotkey_37": 0.7447815537452698,
        "hotkey_35": 0.7694319486618042,
        "hotkey_40": 0.03637188673019409,
        "hotkey_38": 0.7205913066864014,
        "hotkey_36": 0.0890098512172699,
        "hotkey_33": 0.7766138315200806,
        "hotkey_22": 0.0,
        "hotkey_58": 0.0,
    }
    rewards_tensor = dict_to_tensor(rewards, 256)
    uids = torch.tensor(
        [39, 34, 37, 35, 40, 38, 36, 33, 22, 58]
    )  # Update scores for these hotkeys
    scoring_results = ScoringResults(
        combined_scores=rewards_tensor,
        combined_uids=uids,
    )

    moving_average_scores = await update_moving_averages(
        moving_average_scores,
        scoring_results,
    )

    assert moving_average_scores.sum().item() != 0


@pytest.mark.asyncio
@patch_all_dependencies
async def test_large_rewards(*args):
    moving_average_scores = torch.zeros(256)
    rewards = {"hotkey_39": 0.7715857625007629 * 20}
    rewards_tensor = dict_to_tensor(rewards, 256)
    uids = torch.tensor([39])  # Only update the score for hotkey_39
    scoring_results = ScoringResults(
        combined_scores=rewards_tensor, combined_uids=uids
    )

    previous_moving_average = moving_average_scores[39]
    moving_average_scores = await update_moving_averages(
        moving_average_scores,
        scoring_results,
    )
    current_moving_average = moving_average_scores[39]

    assert current_moving_average > previous_moving_average
    assert (
        abs(
            current_moving_average - MOVING_AVERAGE_ALPHA * rewards["hotkey_39"]
        )
        < 1e-2
    )


@pytest.mark.asyncio
@patch_all_dependencies
async def test_rewards_with_nans(*args):
    moving_average_scores = torch.zeros(256)
    rewards = {"hotkey_0": float("nan")}
    rewards_tensor = dict_to_tensor(rewards, 256)
    uids = torch.tensor([0])  # Only update the score for hotkey_0
    scoring_results = ScoringResults(
        combined_scores=rewards_tensor, combined_uids=uids
    )

    moving_average_scores = await update_moving_averages(
        moving_average_scores,
        scoring_results,
    )

    assert torch.isnan(moving_average_scores).sum().item() == 0


@pytest.mark.asyncio
@patch_all_dependencies
async def test_decay_application(*args):
    # Set up initial scores and mock config
    previous_ma_scores = torch.ones(256)
    rewards = {f"hotkey_{i}": 0.0 for i in range(256)}  # No new rewards
    rewards_tensor = dict_to_tensor(rewards, 256)
    uids = torch.tensor([])  # No responding UIDs
    scoring_results = ScoringResults(
        combined_scores=rewards_tensor, combined_uids=uids
    )

    # Mock the config to set a specific decay rate
    mock_decay_rate = 0.05
    mock_config = MagicMock()
    mock_config.alchemy.ma_decay = mock_decay_rate

    # Mock should_apply_decay to return True for even UIDs and False for odd UIDs
    def mock_should_apply_decay(uid):
        return uid % 2 == 0

    with patch(
        "neurons.validator.forward.get_config", return_value=mock_config
    ), patch(
        "neurons.validator.averages.should_apply_decay",
        side_effect=mock_should_apply_decay,
    ):
        updated_ma_scores = await update_moving_averages(
            previous_ma_scores, scoring_results
        )

    # Check that decay has been applied correctly
    for uid in range(256):
        if uid % 2 == 0:
            # Even UIDs should have decayed
            assert (
                updated_ma_scores[uid] < 1.0
            ), f"UID {uid} should have decayed, but didn't"
            assert (
                updated_ma_scores[uid] > 0.9
            ), f"UID {uid} decayed too much: {updated_ma_scores[uid].item()}"
        else:
            # Odd UIDs should not have decayed
            assert torch.isclose(
                updated_ma_scores[uid], torch.tensor(1.0), atol=1e-5
            ), f"UID {uid} should not have decayed, but got {updated_ma_scores[uid].item()}"

    # Check that some scores have decayed and some haven't
    assert torch.any(updated_ma_scores < 1.0), "No scores were decayed"
    assert torch.any(updated_ma_scores == 1.0), "All scores were decayed"

    # Check that all decayed scores are within a reasonable range
    decayed_scores = updated_ma_scores[updated_ma_scores < 1.0]
    assert torch.all(
        decayed_scores > 0.9
    ), f"Some scores decayed too much: {decayed_scores[decayed_scores <= 0.9]}"
    assert torch.all(
        decayed_scores < 1.0
    ), f"Some decayed scores didn't actually decay: {decayed_scores[decayed_scores >= 1.0]}"

    # Print the actual decay for debugging
    actual_decay = 1 - decayed_scores.mean().item()
    logger.debug(f"Actual average decay: {actual_decay}")
