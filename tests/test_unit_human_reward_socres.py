# In tests/test_unit_human_reward_scores.py

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import torch
import bittensor as bt

from neurons.protocol import ImageGeneration, ModelType
from neurons.validator.rewards.types import RewardModelType
from neurons.validator.rewards.pipeline import (
    get_function,
    apply_reward_function,
    REWARD_MODELS,
)


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
    test_uids = [1, 2, 3, 4, 5]  # Example UIDs
    test_hotkeys = [f"hotkey_{uid}" for uid in test_uids]

    # Mock metagraph
    mock_metagraph = MagicMock()
    mock_metagraph.hotkeys = test_hotkeys

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
    ):
        # First, apply EmptyScoreRewardModel
        empty_rewards, _ = await apply_reward_function(
            get_function(REWARD_MODELS, RewardModelType.EMPTY),
            generate_synapse(),
            responses,
            {hotkey: 0.0 for hotkey in test_hotkeys},
        )

        # Assert all rewards are 0
        assert all(reward == 0.0 for reward in empty_rewards.values())

        # Now, apply HumanValidationRewardModel
        human_rewards, _ = await apply_reward_function(
            get_function(REWARD_MODELS, RewardModelType.HUMAN),
            generate_synapse(),
            responses,
            empty_rewards,
        )

        # Assert rewards have changed for hotkeys with votes
        assert human_rewards["hotkey_1"] > 0
        assert human_rewards["hotkey_2"] > 0
        assert human_rewards["hotkey_3"] > 0
        assert human_rewards["hotkey_4"] > 0
        assert human_rewards["hotkey_5"] == 0  # hotkey didn't get votes

        # Assert relative magnitudes
        assert (
            human_rewards["hotkey_1"] > human_rewards["hotkey_2"]
        )  # hotkey_1 got more votes than hotkey_2
        assert (
            human_rewards["hotkey_3"] > human_rewards["hotkey_4"]
        )  # hotkey_3 got more votes than hotkey_4

        # Verify that the backend client's get_votes method was called
        mock_backend_client.get_votes.assert_awaited_once()
