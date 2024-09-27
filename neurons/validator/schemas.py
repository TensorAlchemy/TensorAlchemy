from typing import List, Dict

from pydantic import BaseModel

from scoring.models import RewardModelType


class Batch(BaseModel):
    batch_id: str
    # Results
    prompt: str
    computes: List[str]

    # Filtering
    nsfw_scores: List[float]
    blacklist_scores: List[int] = []
    should_drop_entries: List[int] = []

    # Miner
    miner_hotkeys: List[str]
    miner_coldkeys: List[str]

    # Validator
    validator_hotkey: str


class ScoresUploadRequest(BaseModel):
    task_id: str
    scores: Dict[RewardModelType, Dict[str, float]]
