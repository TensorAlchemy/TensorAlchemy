import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch
from neurons.validator.backend.exceptions import PostWeightsError
from neurons.validator.weights import set_weights


@pytest.fixture
def mock_dependencies(monkeypatch):
    mocks = {
        "config": MagicMock(netuid=1),
        "metagraph": MagicMock(uids=torch.tensor([1, 2, 3, 4, 5])),
        "subtensor": MagicMock(
            set_weights=MagicMock(return_value=(True, "Success"))
        ),
        "wallet": MagicMock(),
        "backend_client": AsyncMock(),
        "process_weights": MagicMock(
            return_value=(
                torch.tensor([1, 2, 3]),
                torch.tensor([0.2, 0.3, 0.5]),
            )
        ),
    }

    mock_modules = {
        "neurons.validator.weights": {
            "get_config": lambda: mocks["config"],
            "get_metagraph": lambda: mocks["metagraph"],
            "get_subtensor": lambda: mocks["subtensor"],
            "get_wallet": lambda: mocks["wallet"],
            "get_backend_client": lambda: mocks["backend_client"],
        },
        "bittensor.utils.weight_utils": {
            "process_weights_for_netuid": mocks["process_weights"],
        },
    }

    for module, functions in mock_modules.items():
        for func_name, func in functions.items():
            monkeypatch.setattr(f"{module}.{func_name}", func)

    class MockExecutor:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, fn, *args, **kwargs):
            return asyncio.create_task(fn(*args, **kwargs))

    monkeypatch.setattr("concurrent.futures.ThreadPoolExecutor", MockExecutor)

    return mocks


@pytest.mark.asyncio
async def test_set_weights_success(mock_dependencies):
    hotkeys = ["hotkey1", "hotkey2", "hotkey3", "hotkey4", "hotkey5"]
    moving_average_scores = torch.tensor([0.5, 1.0, 0.7, 0.3, 0.9])

    await set_weights(hotkeys, moving_average_scores)

    # Check if weights were posted to backend
    mock_dependencies["backend_client"].post_weights.assert_called_once()

    # Check if set_weights was called on subtensor
    mock_dependencies["subtensor"].set_weights.assert_called_once()

    # Check the arguments of set_weights
    call_args = mock_dependencies["subtensor"].set_weights.call_args[1]
    assert torch.equal(call_args["uids"], torch.tensor([1, 2, 3]))
    assert torch.equal(call_args["weights"], torch.tensor([0.2, 0.3, 0.5]))


@pytest.mark.asyncio
async def test_set_weights_backend_error(mock_dependencies):
    hotkeys = ["hotkey1", "hotkey2", "hotkey3", "hotkey4", "hotkey5"]
    moving_average_scores = torch.tensor([0.5, 1.0, 0.7, 0.3, 0.9])
    mock_dependencies["backend_client"].post_weights.side_effect = (
        PostWeightsError("Backend error")
    )

    await set_weights(hotkeys, moving_average_scores)

    # Check if set_weights was still called on subtensor despite backend error
    mock_dependencies["subtensor"].set_weights.assert_called_once()


@pytest.mark.asyncio
async def test_set_weights_processing_error(mock_dependencies):
    hotkeys = ["hotkey1", "hotkey2", "hotkey3", "hotkey4", "hotkey5"]
    moving_average_scores = torch.tensor([0.5, 1.0, 0.7, 0.3, 0.9])
    mock_dependencies["process_weights"].side_effect = Exception(
        "Processing error"
    )

    await set_weights(hotkeys, moving_average_scores)

    # Wait for a short time to allow the async task to complete
    await asyncio.sleep(0.1)

    # Check if set_weights was not called on subtensor due to processing error
    mock_dependencies["subtensor"].set_weights.assert_not_called()


@pytest.mark.asyncio
async def test_set_weights_normalization(mock_dependencies):
    hotkeys = ["hotkey1", "hotkey2", "hotkey3"]
    moving_average_scores = torch.tensor([1.0, 2.0, 3.0])

    await set_weights(hotkeys, moving_average_scores)

    # Wait for a short time to allow the async task to complete
    await asyncio.sleep(0.1)

    # Check if weights were normalized correctly
    normalized_weights = torch.nn.functional.normalize(
        moving_average_scores, p=1, dim=0
    )
    mock_dependencies["backend_client"].post_weights.assert_called_once()

    # Extract the args from the call
    call_args = mock_dependencies["backend_client"].post_weights.call_args[0]

    # Check that the hotkeys match
    assert call_args[0] == hotkeys

    # Check that the weights are close to the expected normalized weights
    assert torch.allclose(call_args[1], normalized_weights, atol=1e-6)
