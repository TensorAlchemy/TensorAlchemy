from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional

from neurons.validator.scoring.models.types import RewardModelType


def convert_enum_keys_to_strings(data) -> dict:
    if isinstance(data, dict):
        new_dict = {}
        for k, v in data.items():
            new_key = k.value if isinstance(k, Enum) else k
            new_value = convert_enum_keys_to_strings(v)
            new_dict[new_key] = new_value

        return new_dict

    if isinstance(data, list):
        new_list = []
        for item in data:
            new_item = convert_enum_keys_to_strings(item)
            new_list.append(new_item)

        return new_list

    if isinstance(data, Enum):
        return data.value

    return data


@dataclass
class EventSchema:
    images: List
    task_type: str
    block: float
    uids: List[int]
    hotkeys: List[str]
    prompt: str
    step_length: float
    model_type: str

    # Reward data
    rewards: Dict[str, List[float]]

    # Bittensor data
    stake: List[float]
    rank: List[float]
    vtrust: List[float]
    dividends: List[float]
    emissions: List[float]

    set_weights: Optional[List[List[float]]]

    @staticmethod
    def from_dict(event_dict: dict) -> "EventSchema":
        """Converts a dictionary to an EventSchema object."""

        rewards = convert_enum_keys_to_strings(
            {
                RewardModelType.BLACKLIST: event_dict.get(
                    RewardModelType.BLACKLIST
                ),
                RewardModelType.HUMAN: event_dict.get(RewardModelType.HUMAN),
                RewardModelType.IMAGE: event_dict.get(RewardModelType.IMAGE),
                RewardModelType.NSFW: event_dict.get(RewardModelType.NSFW),
            }
        )

        return EventSchema(
            task_type=event_dict["task_type"],
            model_type=event_dict["model_type"],
            block=event_dict["block"],
            uids=event_dict["uids"],
            hotkeys=event_dict["hotkeys"],
            prompt=event_dict["prompt"],
            step_length=event_dict["step_length"],
            images=event_dict["images"],
            rewards=rewards,
            set_weights=None,
            stake=event_dict["stake"],
            rank=event_dict["rank"],
            vtrust=event_dict["vtrust"],
            dividends=event_dict["dividends"],
            emissions=event_dict["emissions"],
            # moving_averages=event_dict["moving_averages"]
        )
