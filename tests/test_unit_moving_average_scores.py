import pytest
from unittest.mock import patch, MagicMock
import torch
from loguru import logger
from functools import wraps
from typing import Dict

from neurons.validator.config import get_device
from neurons.validator.forward import update_moving_averages


# Mock functions and classes
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


# Create instances of our mocks
mock_meta = mock_metagraph()
mock_client = mock_backend_client()


# Custom decorator to apply all patches
def patch_all_dependencies(func):
    @wraps(func)
    @patch("neurons.validator.forward.get_metagraph", return_value=mock_meta)
    @patch("neurons.validator.forward.get_backend_client", return_value=mock_client)
    @patch("neurons.validator.forward.get_device", return_value=torch.device("cpu"))
    async def wrapper(*args, **kwargs):
        return await func()

    return wrapper


def dict_to_tensor(rewards_dict: Dict[str, float], n: int) -> torch.FloatTensor:
    rewards_tensor = torch.zeros(n)
    for key, value in rewards_dict.items():
        index = int(key.split("_")[1])
        if index < n:
            rewards_tensor[index] = value
    return rewards_tensor


@pytest.mark.asyncio
@patch_all_dependencies
async def test_non_zero_moving_averages():
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

    moving_average_scores = await update_moving_averages(
        moving_average_scores,
        rewards_tensor,
    )

    assert moving_average_scores.sum().item() != 0


@pytest.mark.asyncio
@patch_all_dependencies
async def test_large_rewards():
    moving_average_scores = torch.zeros(256)
    rewards = {"hotkey_39": 0.7715857625007629 * 20}
    rewards_tensor = dict_to_tensor(rewards, 256)

    previous_moving_average = moving_average_scores[39]
    moving_average_scores = await update_moving_averages(
        moving_average_scores,
        rewards_tensor,
    )
    current_moving_average = moving_average_scores[39]

    assert current_moving_average > previous_moving_average


@pytest.mark.asyncio
@patch_all_dependencies
async def test_rewards_with_nans():
    moving_average_scores = torch.zeros(256)
    rewards = {"hotkey_0": float("nan")}
    rewards_tensor = dict_to_tensor(rewards, 256)

    moving_average_scores = await update_moving_averages(
        moving_average_scores,
        rewards_tensor,
    )

    assert torch.isnan(moving_average_scores).sum().item() == 0


@pytest.mark.asyncio
@patch_all_dependencies
async def test_zero_rewards():
    moving_average_scores = torch.zeros(256)
    rewards = {f"hotkey_{i}": 0.0 for i in range(256)}
    rewards_tensor = dict_to_tensor(rewards, 256)

    previous_moving_average_scores_sum = moving_average_scores.sum()
    moving_average_scores = await update_moving_averages(
        moving_average_scores,
        rewards_tensor,
    )
    current_moving_average_scores_sum = moving_average_scores.sum()

    assert previous_moving_average_scores_sum >= current_moving_average_scores_sum


@pytest.mark.asyncio
@patch_all_dependencies
async def test_ones_rewards():
    moving_average_scores = torch.zeros(256)
    rewards = {f"hotkey_{i}": 1.0 for i in range(256)}
    rewards_tensor = dict_to_tensor(rewards, 256)

    previous_moving_average_scores_sum = moving_average_scores.sum()
    moving_average_scores = await update_moving_averages(
        moving_average_scores,
        rewards_tensor,
    )
    current_moving_average_scores_sum = moving_average_scores.sum()

    assert previous_moving_average_scores_sum < current_moving_average_scores_sum
