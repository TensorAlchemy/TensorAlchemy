from dataclasses import dataclass
from typing import Dict, List, Optional

from neurons.validator.rewards.types import RewardModelType


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

        rewards = {
            RewardModelType.BLACKLIST: event_dict.get(RewardModelType.BLACKLIST),
            RewardModelType.SIMILARITY: event_dict.get(RewardModelType.SIMILARITY),
            RewardModelType.HUMAN: event_dict.get(RewardModelType.HUMAN),
            RewardModelType.IMAGE: event_dict.get(RewardModelType.IMAGE),
            RewardModelType.NSFW: event_dict.get(RewardModelType.NSFW),
        }

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
