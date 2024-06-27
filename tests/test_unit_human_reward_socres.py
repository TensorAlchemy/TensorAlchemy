import pytest
from unittest.mock import patch, MagicMock, AsyncMock

import torch
import bittensor as bt
from functools import wraps

# Import the actual get_metagraph function
import neurons.validator.config as validator_config


def mock_metagraph():
    test_uids = torch.tensor([0, 1, 2, 3, 4])  # Example UIDs
    test_hotkeys = [f"hotkey_{uid.item()}" for uid in test_uids]

    to_return = MagicMock()
    to_return.n = len(test_hotkeys)
    to_return.hotkeys = test_hotkeys

    print(f"Creating mock metagraph with n = {to_return.n}")
    return to_return


def mock_backend_client():
    mock_client = AsyncMock()
    mock_votes = {
        "round_1": {
            "hotkey_0": 1,
            "hotkey_1": 2,
        },
        "round_2": {
            "hotkey_2": 3,
            "hotkey_3": 4,
        },
    }
    mock_client.get_votes.return_value = mock_votes
    return mock_client


# Create instances of our mocks
mock_meta = mock_metagraph()
mock_client = mock_backend_client()


def patch_all_dependencies(func):
    @wraps(func)
    @patch(
        "neurons.validator.config.get_metagraph",
        return_value=mock_meta,
    )
    @patch(
        "neurons.validator.config.get_backend_client",
        return_value=mock_client,
    )
    @patch(
        "neurons.validator.rewards.models.base.get_metagraph",
        return_value=mock_meta,
    )
    @patch(
        "neurons.validator.rewards.models.human.get_metagraph",
        return_value=mock_meta,
    )
    @patch(
        "neurons.validator.rewards.models.human.get_backend_client",
        return_value=mock_client,
    )
    async def wrapper(*args, **kwargs):
        return await func()

    return wrapper


@pytest.mark.asyncio
@patch_all_dependencies
async def test_apply_human_voting_weight():
    # Import here to ensure patches are applied first
    from neurons.validator.config import get_metagraph
    from neurons.validator.rewards.pipeline import apply_function
    from neurons.validator.rewards.models import (
        PackedRewardModel,
        EmptyScoreRewardModel,
        HumanValidationRewardModel,
    )

    # Verify that get_metagraph() is properly mocked
    assert get_metagraph().n == 5, f"Expected 5, got {get_metagraph().n}"

    # Create mock responses as Synapse objects
    responses = []
    for _uid, hotkey in enumerate(get_metagraph().hotkeys):
        response = generate_synapse()
        response.axon = bt.TerminalInfo(
            status_code=200,
            status_message="Success",
            process_time=0.1,
            ip="0.0.0.0",
            port=8080,
            version=100,
            nonce=0,
            uuid="test-uuid",
            hotkey=hotkey,
        )
        responses.append(response)

    # First, apply EmptyScoreRewardModel
    empty_rewards, _ = await apply_function(
        PackedRewardModel(
            weight=1.0,
            model=EmptyScoreRewardModel(),
        ),
        generate_synapse(),
        responses,
    )

    # Assert all rewards are 0
    assert torch.all(empty_rewards == 0)

    # Now, apply HumanValidationRewardModel
    human_rewards, human_rewards_normalised = await apply_function(
        PackedRewardModel(
            weight=1.0,
            model=HumanValidationRewardModel(),
        ),
        generate_synapse(),
        responses,
    )

    print(human_rewards)
    print(human_rewards_normalised)

    # Assert rewards have changed for UIDs with votes
    assert human_rewards[0].item() > 0
    assert human_rewards[1].item() > human_rewards[0].item()
    assert human_rewards[2].item() > human_rewards[1].item()
    assert human_rewards[3].item() > human_rewards[2].item()

    # Assert rewards have changed for normalized UIDs with votes
    assert human_rewards_normalised[0].item() > 0
    assert human_rewards_normalised[1].item() > human_rewards_normalised[0].item()
    assert human_rewards_normalised[2].item() > human_rewards_normalised[1].item()
    assert human_rewards_normalised[3].item() > human_rewards_normalised[2].item()

    assert len(human_rewards) == 5

    # This UID didn't receive any votes
    assert human_rewards[4].item() == 0
    assert human_rewards_normalised[4].item() == 0


def generate_synapse() -> bt.Synapse:
    from neurons.protocol import ImageGeneration, ModelType

    return ImageGeneration(
        generation_type="TEXT_TO_IMAGE",
        seed=-1,
        model_type=ModelType.ALCHEMY.value,
        images=[
            bt.Tensor.serialize(
                torch.full(
                    [3, 1024, 1024],
                    254,
                    dtype=torch.float,
                )
            )
        ],
    )
