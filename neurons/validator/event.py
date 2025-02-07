from enum import Enum
from typing import List

import torch
from pydantic import BaseModel

from scoring.types import ScoringResults


class EventSchema(BaseModel):
    images: List[str]  # Assuming images are stored as base64 strings
    task_type: str
    block: float
    uids: List[int]
    hotkeys: List[str]
    prompt: str
    step_length: float
    task_type: str
    compute_count: int
    results: ScoringResults
    stake: float
    rank: float
    vtrust: float
    dividends: float
    emissions: float

    def to_dict(self) -> dict:
        event_dict = self.model_dump(exclude_none=True)
        results_dict = {}
        for score in self.results.scores:
            uids: torch.tensor = score.uids

            results_dict[score.type.value] = {
                "uids": uids.tolist(),
                "scores": score.scores[uids].tolist(),
                "normalized": score.normalized[uids].tolist(),
                "raw": (
                    score.raw[uids].tolist() if score.raw is not None else None
                ),
            }

        combined_uids: torch.tensor = self.results.combined_uids
        event_dict["results"] = results_dict
        event_dict["combined_uids"] = combined_uids.tolist()
        event_dict["combined_scores"] = self.results.combined_scores[
            combined_uids
        ].tolist()

        return event_dict


def convert_enum_keys_to_strings(data):
    if isinstance(data, dict):
        return {
            k.value if isinstance(k, Enum) else k: convert_enum_keys_to_strings(
                v
            )
            for k, v in data.items()
        }
    elif isinstance(data, list):
        return [convert_enum_keys_to_strings(item) for item in data]
    elif isinstance(data, Enum):
        return data.value
    return data
