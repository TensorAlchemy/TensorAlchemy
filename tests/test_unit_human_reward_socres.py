import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import torch
import bittensor as bt

from neurons.protocol import ImageGeneration, ModelType
from neurons.validator.rewards.types import RewardModelType
from neurons.validator.rewards.pipeline import apply_reward_function
from neurons.validator.rewards.models import get_function, get_reward_models
from neurons.validator.config import get_device


def generate_synapse() -> bt.Synapse:
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


@pytest.mark.asyncio
async def test_apply_human_voting_weight():
    # Setup
    test_uids = torch.tensor([1, 2, 3, 4, 5])  # Example UIDs
    test_hotkeys = [f"hotkey_{uid.item()}" for uid in test_uids]

    # Mock metagraph
    mock_metagraph = MagicMock()
    mock_metagraph.hotkeys = test_hotkeys
    mock_metagraph.n = len(test_hotkeys)

    # Mock backend client
    mock_backend_client = AsyncMock()
    mock_votes = {
        "round_1": {
            "hotkey_1": 10,
            "hotkey_2": 5,
        },
        "round_2": {
            "hotkey_3": 8,
            "hotkey_4": 3,
        },
    }
    mock_backend_client.get_votes.return_value = mock_votes

    # Create mock responses as Synapse objects
    responses = []
    for uid, hotkey in zip(test_uids, test_hotkeys):
        response = generate_synapse()
        response.dendrite = bt.TerminalInfo(
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

    # Apply patches for the entire test
    with patch(
        "neurons.validator.rewards.models.human.get_backend_client",
        return_value=mock_backend_client,
    ), patch(
        "neurons.validator.rewards.models.human.get_metagraph",
        return_value=mock_metagraph,
    ):
        # First, apply EmptyScoreRewardModel
        empty_rewards, _ = await apply_reward_function(
            get_function(get_reward_models(), RewardModelType.EMPTY),
            generate_synapse(),
            responses,
            torch.zeros(mock_metagraph.n).to(get_device()),
            test_uids,
        )

        # Assert all rewards are 0
        assert torch.all(empty_rewards == 0)

        # Now, apply HumanValidationRewardModel
        human_rewards, _ = await apply_reward_function(
            get_function(get_reward_models(), RewardModelType.HUMAN),
            generate_synapse(),
            responses,
            empty_rewards,
            test_uids,
        )

        # Assert rewards have changed for UIDs with votes
        assert human_rewards[1] > 0
        assert human_rewards[2] > 0
        assert human_rewards[3] > 0
        assert human_rewards[4] > 0
        assert human_rewards[5] == 0  # This UID didn't receive any votes

        # Assert relative magnitudes
        assert human_rewards[1] > human_rewards[2]  # UID 1 got more votes than UID 2
        assert human_rewards[3] > human_rewards[4]  # UID 3 got more votes than UID 4

        # Verify that the backend client's get_votes method was called
        mock_backend_client.get_votes.assert_awaited_once()
