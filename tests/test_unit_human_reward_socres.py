import copy

import torch
import bittensor as bt

from neurons.protocol import ImageGeneration, ModelType
from neurons.validator.config import get_device
from neurons.validator.rewards.types import RewardModelType
from neurons.validator.rewards.pipeline import (
    get_function,
    apply_reward_function,
    REWARD_MODELS,
)


async def test_apply_human_voting_weight():
    test_index = 0
    synapse = ImageGeneration(
        generation_type="TEXT_TO_IMAGE",
        seed=-1,
        model_type=ModelType.ALCHEMY.value,
        images=[
            bt.Tensor.serialize(torch.full([3, 1024, 1024], 254, dtype=torch.float))
        ],
    )

    rewards = torch.tensor(
        [
            0.6522690057754517,
            0.7715857625007629,
            0.7447815537452698,
            0.7694319486618042,
            0.03637188673019409,
            0.7205913066864014,
            0.0890098512172699,
            0.7766138315200806,
            0.0,
            0.0,
        ]
    ).to(get_device())

    previous_reward = copy.copy(rewards[test_index].item())
    new_rewards = await apply_reward_function(
        get_function(REWARD_MODELS, RewardModelType.HUMAN),
        synapse,
        [],  # some synapse responses
        torch.ones(len(rewards)).to(get_device()),
    )
    current_reward = new_rewards[test_index].item()
    assert current_reward > previous_reward
