from typing import List

import torch


async def get_human_voting_scores(
    self,
    hotkeys: List[str],
) -> torch.Tensor:
    (
        _,
        human_voting_scores_normalised,
    ) = await self.human_voting_reward_model.get_rewards(
        hotkeys,
    )

    return human_voting_scores_normalised


def apply_human_voting_weight(
    rewards: torch.Tensor,
    human_voting_scores: torch.Tensor,
    human_voting_weight: float,
) -> torch.Tensor:
    scattered_rewards_adjusted = (
        #
        rewards
        + (human_voting_weight * human_voting_scores)
    )

    return scattered_rewards_adjusted


async def get_human_rewards(
    hotkeys: List[str],
    rewards: torch.Tensor,
) -> torch.Tensor:
    human_voting_scores = await get_human_voting_scores(
        hotkeys,
    )
    scattered_rewards_adjusted = apply_human_voting_weight(
        rewards, human_voting_scores, human_voting_weight
    )
    return scattered_rewards_adjusted
