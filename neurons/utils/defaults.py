from datetime import datetime
from dataclasses import dataclass


@dataclass
class Stats:
    start_time: datetime
    start_dt: datetime
    total_requests: int
    timeouts: int
    nsfw_count: int
    generation_time: int


# Get default stats
def get_defaults(self) -> Stats:
    now = datetime.now()
    stats = Stats(
        start_time=now,
        start_dt=datetime.strftime(now, "%Y/%m/%d %H:%M"),
        total_requests=0,
        nsfw_count=0,
        timeouts=0,
        generation_time=0,
    )
    return stats
