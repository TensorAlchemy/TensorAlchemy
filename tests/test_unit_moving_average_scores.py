import torch
import pytest
from dotenv import load_dotenv

import bittensor as bt

from neurons.validator.config import get_device
from neurons.validator.forward import update_moving_averages


async def test_non_zero_moving_averages():
    moving_average_scores = torch.zeros(256)
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
    uids = torch.tensor([39, 34, 37, 35, 40, 38, 36, 33, 22, 58]).to(get_device())

    scattered_rewards = moving_average_scores.scatter(0, uids, rewards).to(get_device())

    moving_average_scores = await update_moving_averages(
        moving_average_scores,
        scattered_rewards,
    )

    assert moving_average_scores.sum().item() != 0


async def test_large_rewards():
    test_uid_index = 39
    moving_average_scores = torch.zeros(256)
    uids = torch.tensor([test_uid_index]).to(get_device())
    rewards = torch.tensor([0.7715857625007629 * 20]).to(get_device())

    scattered_rewards = moving_average_scores.scatter(
        0,
        uids,
        rewards,
    ).to(get_device())

    previous_moving_average = moving_average_scores[test_uid_index]
    moving_average_scores = await update_moving_averages(
        moving_average_scores, scattered_rewards
    )
    current_moving_average = moving_average_scores[test_uid_index]

    assert current_moving_average > previous_moving_average


async def test_rewards_with_nans():
    moving_average_scores = torch.zeros(256)
    rewards = torch.zeros(len(moving_average_scores)).to(get_device())
    rewards[0] = float("nan")

    moving_average_scores = await update_moving_averages(
        moving_average_scores,
        rewards,
    )
    assert torch.isnan(moving_average_scores).sum().item() == 0


async def test_zero_rewards():
    moving_average_scores = torch.zeros(256)
    rewards = torch.zeros(len(moving_average_scores)).to(get_device())

    previous_moving_average_scores_sum = moving_average_scores.sum()
    moving_average_scores = await update_moving_averages(moving_average_scores, rewards)
    current_moving_average_scores_sum = moving_average_scores.sum()

    assert previous_moving_average_scores_sum >= current_moving_average_scores_sum


async def test_ones_rewards():
    moving_average_scores = torch.zeros(256)
    rewards = torch.ones(len(moving_average_scores)).to(get_device())

    previous_moving_average_scores_sum = moving_average_scores.sum()
    moving_average_scores = await update_moving_averages(moving_average_scores, rewards)
    current_moving_average_scores_sum = moving_average_scores.sum()

    assert previous_moving_average_scores_sum < current_moving_average_scores_sum
