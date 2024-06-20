import copy

import torch

from neurons.validator import config
from neurons.validator.rewards.reward import (
    apply_human_voting_weight,
)


def test_apply_human_voting_weight():
    device = config.get_default_device()
    human_voting_weight = 0.02 / 32
    test_index = 0
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
    ).to(device)
    human_voting_scores = torch.tensor(
        [
            91 / 100,
            1 / 100,
            1 / 100,
            1 / 100,
            1 / 100,
            1 / 100,
            1 / 100,
            1 / 100,
            1 / 100,
            1 / 100,
        ]
    ).to(device)
    previous_reward = copy.copy(rewards[test_index].item())
    new_rewards = apply_human_voting_weight(
        rewards, human_voting_scores, human_voting_weight
    )
    current_reward = new_rewards[test_index].item()
    assert current_reward > previous_reward
