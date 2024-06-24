from typing import List

from pydantic import BaseModel


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
