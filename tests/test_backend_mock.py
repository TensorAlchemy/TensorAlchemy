import pytest

import torch
from unittest.mock import AsyncMock, MagicMock, patch

from neurons.validator.backend.client_mock import MockTensorAlchemyBackendClient


@pytest.fixture
def mock_tensor_alchemy_backend_client():
    with patch(
        "neurons.validator.backend.client.TensorAlchemyBackendClient",
        MockTensorAlchemyBackendClient,
    ):
        yield MockTensorAlchemyBackendClient()


@pytest.mark.asyncio
async def test_poll_task(mock_tensor_alchemy_backend_client):
    result = await mock_tensor_alchemy_backend_client.poll_task()
    assert result is not None


@pytest.mark.asyncio
async def test_get_task(mock_tensor_alchemy_backend_client):
    result = await mock_tensor_alchemy_backend_client.get_task()
    assert result is not None


@pytest.mark.asyncio
async def test_get_votes(mock_tensor_alchemy_backend_client):
    result = await mock_tensor_alchemy_backend_client.get_votes()
    assert result is not None


@pytest.mark.asyncio
async def test_post_moving_averages(mock_tensor_alchemy_backend_client):
    hotkeys = ["key1", "key2"]
    moving_average_scores = torch.tensor([0.5, 0.7])
    await mock_tensor_alchemy_backend_client.post_moving_averages(
        hotkeys, moving_average_scores
    )


@pytest.mark.asyncio
async def test_post_batch(mock_tensor_alchemy_backend_client):
    batch = MagicMock()
    result = await mock_tensor_alchemy_backend_client.post_batch(batch)
    assert isinstance(result, AsyncMock)


@pytest.mark.asyncio
async def test_post_weights(mock_tensor_alchemy_backend_client):
    hotkeys = ["key1", "key2"]
    raw_weights = torch.tensor([0.3, 0.7])
    await mock_tensor_alchemy_backend_client.post_weights(hotkeys, raw_weights)


@pytest.mark.asyncio
async def test_update_task_state(mock_tensor_alchemy_backend_client):
    task_id = "task123"
    state = MagicMock()
    await mock_tensor_alchemy_backend_client.update_task_state(task_id, state)


@pytest.mark.asyncio
async def test_sign_message(mock_tensor_alchemy_backend_client):
    result = mock_tensor_alchemy_backend_client._sign_message("test message")
    assert result == "mocked_signature"
