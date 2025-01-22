from typing import List, Optional

import torch
from pydantic import BaseModel, ConfigDict, Field

from scoring.models.types import RewardModelType


def combine_uids(
    uids_a: torch.Tensor,
    uids_b: torch.Tensor,
) -> torch.Tensor:
    if uids_a.numel() == 0:
        return uids_b

    if uids_b.numel() == 0:
        return uids_a

    # Concatenate and remove duplicates
    return torch.unique(
        torch.cat(
            (
                uids_a.flatten(),
                uids_b.flatten(),
            )
        )
    )


class ScoringResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    uids: torch.Tensor
    type: RewardModelType
    scores: torch.Tensor
    normalized: torch.Tensor
    raw: Optional[torch.Tensor] = None


class ScoringResults(BaseModel):
    scores: List[ScoringResult] = Field(default=[])

    combined_scores: torch.Tensor
    combined_uids: torch.Tensor = Field(default=torch.Tensor([]))

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_score(self, to_find: RewardModelType) -> Optional[ScoringResult]:
        for item in self.scores:
            if item.type == to_find:
                return item

        return None

    def add_score(self, other: ScoringResult) -> None:
        self.scores.append(other)
        self.combined_uids = combine_uids(self.combined_uids, other.uids)

    def add_scores(self, others: List[ScoringResult]) -> None:
        self.scores += others

    def update(self, other: "ScoringResults") -> None:
        self.scores += other.scores
