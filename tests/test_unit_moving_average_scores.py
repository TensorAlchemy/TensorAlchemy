import pytest
from unittest.mock import patch, MagicMock
import torch
from loguru import logger

from neurons.validator.config import get_device
from neurons.validator.forward import update_moving_averages


def fake_backend_client():
    class FakeBackendClient:
        async def post_moving_averages(self, *args, **kwargs):
            logger.info("[fake] posting moving averages...")

    return FakeBackendClient()


@pytest.fixture(autouse=True)
def mock_backend_client():
    with patch(
        "neurons.validator.forward.get_backend_client", side_effect=fake_backend_client
    ):
        yield


@pytest.fixture
def mock_metagraph():
    mock = MagicMock()
    mock.hotkeys = [f"hotkey_{i}" for i in range(256)]
    return mock


@pytest.mark.asyncio
async def test_non_zero_moving_averages(mock_metagraph):
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

    with patch("neurons.validator.forward.get_metagraph", return_value=mock_metagraph):
        moving_average_scores = await update_moving_averages(
            moving_average_scores, rewards
        )

    assert moving_average_scores.sum().item() != 0


@pytest.mark.asyncio
async def test_large_rewards(mock_metagraph):
    moving_average_scores = torch.zeros(256)
    rewards = {"hotkey_39": 0.7715857625007629 * 20}

    with patch("neurons.validator.forward.get_metagraph", return_value=mock_metagraph):
        previous_moving_average = moving_average_scores[39]
        moving_average_scores = await update_moving_averages(
            moving_average_scores, rewards
        )
        current_moving_average = moving_average_scores[39]

    assert current_moving_average > previous_moving_average


@pytest.mark.asyncio
async def test_rewards_with_nans(mock_metagraph):
    moving_average_scores = torch.zeros(256)
    rewards = {"hotkey_0": float("nan")}

    with patch("neurons.validator.forward.get_metagraph", return_value=mock_metagraph):
        moving_average_scores = await update_moving_averages(
            moving_average_scores, rewards
        )

    assert torch.isnan(moving_average_scores).sum().item() == 0


@pytest.mark.asyncio
async def test_zero_rewards(mock_metagraph):
    moving_average_scores = torch.zeros(256)
    rewards = {f"hotkey_{i}": 0.0 for i in range(256)}

    with patch("neurons.validator.forward.get_metagraph", return_value=mock_metagraph):
        previous_moving_average_scores_sum = moving_average_scores.sum()
        moving_average_scores = await update_moving_averages(
            moving_average_scores, rewards
        )
        current_moving_average_scores_sum = moving_average_scores.sum()

    assert previous_moving_average_scores_sum >= current_moving_average_scores_sum


@pytest.mark.asyncio
async def test_ones_rewards(mock_metagraph):
    moving_average_scores = torch.zeros(256)
    rewards = {f"hotkey_{i}": 1.0 for i in range(256)}

    with patch("neurons.validator.forward.get_metagraph", return_value=mock_metagraph):
        previous_moving_average_scores_sum = moving_average_scores.sum()
        moving_average_scores = await update_moving_averages(
            moving_average_scores, rewards
        )
        current_moving_average_scores_sum = moving_average_scores.sum()

    assert previous_moving_average_scores_sum < current_moving_average_scores_sum
