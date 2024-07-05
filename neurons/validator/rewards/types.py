from enum import Enum
from typing import List, Optional

import torch
from pydantic import ConfigDict, BaseModel, Field


class RewardModelType(str, Enum):
    # Masking models
    # TODO: Maybe move these out
    NSFW = "NSFW"
    BLACKLIST = "BLACKLIST"

    # Reward models
    EMPTY = "EMPTY"
    HUMAN = "HUMAN"
    IMAGE = "IMAGE"
    SIMILARITY = "SIMILARITY"


class ScoringResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: RewardModelType
    scores: torch.Tensor
    normalized: torch.Tensor


class ScoringResults(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    combined_scores: torch.Tensor
    scores: List[ScoringResult] = Field(default=[])

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_score(self, to_find: RewardModelType) -> Optional[ScoringResult]:
        for item in self.scores:
            if item.type == to_find:
                return item

        return None

    def add_score(self, other: ScoringResult) -> None:
        self.scores.append(other)

    def add_scores(self, others: List[ScoringResult]) -> None:
        self.scores += others

    def update(self, other: "ScoringResults") -> None:
        self.scores += other.scores
