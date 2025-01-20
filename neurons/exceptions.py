from typing import Any, Dict, Optional

from loguru import logger

from neurons.constants import MINIMUM_COMPUTES_FOR_SUBMIT


class ApiError(Exception):
    status: int = 400
    message: str = ""
    code: str = ""

    def get_status(self) -> int:
        return self.status

    def to_json(self) -> Optional[Dict[str, Any]]:
        if self.status in [204]:
            return None

        if len(self.message) < 1:
            logger.error(f"Please overwrite message for <{self}>")

        return {"error": self.message, "code": self.code}


class MinimumValidImagesError(ApiError):
    status: int = 400
    message: str = (
        f"Submitted compute count must be greater than {MINIMUM_COMPUTES_FOR_SUBMIT}"
    )
    code: str = "MINIMUM_VALID_IMAGES_ERROR"


class StakeBelowThreshold(Exception):
    """Exception raised when stake is below the required threshold."""
