from enum import Enum


# This should match backend side
class TaskState(str, Enum):
    FAILED = "FAILED"
    INITIAL = "INITIAL"
    PENDING = "PENDING"
    REJECTED = "REJECTED"
    COMPLETED = "COMPLETED"
