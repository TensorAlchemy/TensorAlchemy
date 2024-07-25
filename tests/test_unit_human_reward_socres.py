import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from functools import wraps

import torch
import bittensor as bt
from loguru import logger

# Import the actual get_metagraph function
import neurons.validator.config as validator_config
from neurons.utils.image import image_tensor_to_base64


def mock_metagraph():
    test_uids = torch.tensor([0, 1, 2, 3, 4])  # Example UIDs
    test_hotkeys = [f"hotkey_{uid.item()}" for uid in test_uids]

    to_return = MagicMock()
    to_return.n = len(test_hotkeys)
    to_return.hotkeys = test_hotkeys

    logger.info(f"Creating mock metagraph with n = {to_return.n}")
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
        "neurons.validator.scoring.models.base.get_metagraph",
        return_value=mock_meta,
    )
    @patch(
        "neurons.validator.scoring.pipeline.get_metagraph",
        return_value=mock_meta,
    )
    @patch(
        "neurons.validator.scoring.models.human.get_metagraph",
        return_value=mock_meta,
    )
    @patch(
        "neurons.validator.scoring.models.human.get_backend_client",
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
    from neurons.validator.scoring.pipeline import (
        apply_function,
        apply_reward_functions,
    )
    from neurons.validator.scoring.models.types import PackedRewardModel
    from neurons.validator.scoring.models import (
        RewardModelType,
        EmptyScoreRewardModel,
        HumanValidationRewardModel,
    )
    from neurons.validator.scoring.types import (
        ScoringResults,
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
    empty_rewards = await apply_function(
        torch.zeros(get_metagraph().n).to(
            validator_config.get_device(),
        ),
        PackedRewardModel(
            weight=1.0,
            model=EmptyScoreRewardModel(),
        ),
        generate_synapse(),
        responses,
    )

    # Assert all rewards are 0
    assert torch.all(empty_rewards.scores == 0)

    # Now, apply HumanValidationRewardModel
    rewards: ScoringResults = await apply_reward_functions(
        [
            PackedRewardModel(
                weight=1.0,
                model=HumanValidationRewardModel(),
            )
        ],
        generate_synapse(),
        responses,
    )

    human_rewards: torch.Tensor = rewards.get_score(
        RewardModelType.HUMAN,
    ).scores

    human_rewards_normalized: torch.Tensor = rewards.get_score(
        RewardModelType.HUMAN,
    ).normalized

    # Assert rewards have changed for UIDs with votes
    assert human_rewards[0] > 1
    assert human_rewards[1] > human_rewards[0]
    assert human_rewards[2] > human_rewards[1]
    assert human_rewards[3] > human_rewards[2]

    # Assert rewards have changed for normalized UIDs with votes
    assert human_rewards_normalized[0] > 0
    assert human_rewards_normalized[1] > human_rewards_normalized[0]
    assert human_rewards_normalized[2] > human_rewards_normalized[1]
    assert human_rewards_normalized[3] > human_rewards_normalized[2]

    assert len(human_rewards) == 5

    # This UID didn't receive any votes
    # Scores should contain seed value by default
    assert human_rewards[4] == torch.tensor(1)

    # Normalied results should not contain a seed value
    assert human_rewards_normalized[4] == torch.tensor(0)


def generate_synapse() -> bt.Synapse:
    from neurons.protocol import ImageGeneration, ModelType

    return ImageGeneration(
        generation_type="TEXT_TO_IMAGE",
        seed=-1,
        model_type=ModelType.ALCHEMY.value,
        images=[
            image_tensor_to_base64(
                torch.full(
                    [3, 1024, 1024],
                    254,
                    dtype=torch.float,
                )
            )
        ],
    )
