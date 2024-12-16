import time
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class RequestStats(BaseModel):
    """Track request statistics for a single caller"""

    history: List[float] = Field(default_factory=list)
    delta: List[float] = Field(default_factory=list)
    count: int = Field(default=0)
    rate_limited_count: int = Field(default=0)
    last_request_time: float = Field(
        default_factory=lambda: time.perf_counter()
    )


class MinerMetrics(BaseModel):
    """Track miner performance metrics"""

    step: int = Field(default=0)
    block: Optional[int] = None
    stake: Optional[float] = None
    trust: Optional[float] = None
    consensus: Optional[float] = None
    incentive: Optional[float] = None
    emissions: Optional[float] = None
    miner_index: Optional[int] = None


class MinerState(BaseModel):
    """Core miner state"""

    metrics: MinerMetrics = Field(default_factory=MinerMetrics)
    request_stats: Dict[str, RequestStats] = Field(default_factory=dict)
