import json
import time
from base64 import b64encode
from typing import Dict, List

import bittensor as bt
import httpx
import torch
from httpx import Response
from loguru import logger
from tenacity import (
    retry,
    stop_after_delay,
    wait_fixed,
    retry_if_result,
)

from neurons.config.utils import is_testnet
from neurons.constants import DEVELOP_URL, TESTNET_URL, MAINNET_URL
from neurons.exceptions import StakeBelowThreshold
from neurons.protocol import denormalize_image_model, ImageGenerationTaskModel
from neurons.validator.backend.exceptions import (
    GetVotesError,
    GetTaskError,
    PostMovingAveragesError,
    PostWeightsError,
    UpdateTaskError,
    UploadScoresError,
)
from neurons.config import get_config, AlchemyHost
from neurons.validator.backend.models import TaskState
from neurons.validator.schemas import Batch, ScoresUploadRequest


class TensorAlchemyBackendClient:
    def __init__(self, hotkey: bt.Keypair = None):
        self.config = get_config()

        if hotkey:
            self.hotkey = hotkey
        else:
            self.hotkey = bt.wallet(config=self.config).hotkey

        self.api_url = MAINNET_URL

        if is_testnet():
            self.api_url = DEVELOP_URL

            if self.config.alchemy.host == AlchemyHost.TESTNET:
                self.api_url = TESTNET_URL

        logger.info(f"Using backend server {self.api_url}")

    def _client(self):
        """Create client"""
        return httpx.AsyncClient(
            event_hooks={
                "request": [
                    # Add signature to request
                    self._add_auth_headers,
                    self._include_validator_version,
                ]
            }
        )

    # Get tasks from the client server
    async def poll_task(self, timeout: int = 30, backoff: int = 1):
        """Performs polling for new task.
        If no new task found within `timeout`
        returns None."""

        @retry(
            stop=stop_after_delay(timeout),
            wait=wait_fixed(backoff),
            # Retry if task is not found (returns None)
            retry=retry_if_result(lambda r: r is None),
            # Returns None after timeout (no task is found)
            retry_error_callback=lambda _: None,
        )
        async def _poll_task_with_retry():
            try:
                return_value = await self.get_task(timeout=3)
            except GetTaskError as e:
                logger.error(f"poll task error: {e}")
                return None

            return return_value

        logger.info(
            f"polling backend for incoming image generation task ({timeout}s) ..."
        )

        return await _poll_task_with_retry()

    async def get_task(
        self, timeout: int = 3
    ) -> ImageGenerationTaskModel | None:
        """Fetch new task from backend.

        Returns task or None if there is no pending task
        """
        try:
            async with self._client() as client:
                response = await client.get(
                    f"{self.api_url}/tasks", timeout=timeout
                )

        except httpx.ReadTimeout as ex:
            raise GetTaskError(f"/tasks read timeout ({timeout}s)") from ex
        except Exception as ex:
            raise GetTaskError("/tasks unknown error") from ex

        try:
            task: Dict = response.json()
        except Exception:
            pass

        if response.status_code == 200:
            logger.info(f"[get_task] task={task}")
            try:
                return denormalize_image_model(**task)
            except Exception as e:
                logger.error(f"[get_task] failed to parse task response: {e}")
                return None

        if response.status_code == 403:
            if task.get("code") == "STAKE_BELOW_THRESHOLD":
                return None

        if response.status_code == 404:
            return None

        if response.status_code == 401:
            if task.get("code") == "VALIDATOR_NOT_FOUND_YET":
                return None
            if task.get("code") == "PENDING_SYNC_METAGRAPH":
                return None
            if task.get("code") == "VALIDATOR_HAS_NOT_ENOUGH_STAKE":
                return None

        raise GetTaskError(
            f"/tasks failed with status_code {response.status_code}:"
            f" {self._error_response_text(response)}"
        )

    async def get_votes(self, timeout: int = 3) -> Dict:
        """Get human votes from backend"""
        try:
            async with self._client() as client:
                response = await client.get(
                    f"{self.api_url}/votes", timeout=timeout
                )
        except httpx.ReadTimeout:
            raise GetVotesError(f"/votes read timeout({timeout}s)")

        if response.status_code != 200:
            raise GetVotesError(
                f"/votes failed with status_code {response.status_code}: "
                f"{self._error_response_text(response)}"
            )
        return response.json()

    async def post_moving_averages(
        self,
        hotkeys: List[str],
        moving_average_scores: torch.Tensor,
        timeout: int = 10,
    ) -> None:
        """Post moving averages"""
        try:
            async with self._client() as client:
                response = await client.post(
                    f"{self.api_url}/validator/averages",
                    json={
                        "averages": {
                            hotkey: moving_average.item()
                            for hotkey, moving_average in zip(
                                hotkeys, moving_average_scores
                            )
                        }
                    },
                    headers={"Content-Type": "application/json"},
                    timeout=timeout,
                )
        except httpx.ReadTimeout:
            raise PostMovingAveragesError(
                f"failed to post moving averages - read timeout ({timeout}s)"
            )

        if response.status_code != 200:
            raise PostMovingAveragesError(
                f"failed to post moving averages with status_code "
                f"{response.status_code}: {self._error_response_text(response)}"
            )

    async def fail_task(self, task_id: str, timeout: int = 10) -> Response:
        """Task failed for some reason"""
        try:
            async with self._client() as client:
                response = await client.post(
                    f"{self.api_url}/tasks/{task_id}/fail",
                    timeout=timeout,
                )

        except Exception as e:
            logger.error(f"Failed to fail task {str(e)}")

        return response

    async def post_batch(self, batch: Batch, timeout: int = 10) -> Response:
        """Post batch of images"""
        try:
            async with self._client() as client:
                response = await client.post(
                    f"{self.api_url}/batches",
                    json=batch.model_dump(),
                    timeout=timeout,
                )

                if response.status_code == 200:
                    response_data = response.json()
                    if response_data.get("code") == "STAKE_BELOW_THRESHOLD":
                        raise StakeBelowThreshold(
                            "Stake is below the required threshold."
                        )

                return response

        except StakeBelowThreshold as e:
            logger.error(f"StakeBelowThreshold: {str(e)}")
            raise

        except Exception as e:
            logger.error(f"Failed to upload batch: {str(e)}")

    async def post_weights(
        self, hotkeys: List[str], raw_weights: torch.Tensor, timeout: int = 10
    ) -> None:
        """Post weights"""
        try:
            async with self._client() as client:
                response = await client.post(
                    f"{self.api_url}/validator/weights",
                    json={
                        "weights": {
                            hotkey: moving_average.item()
                            for hotkey, moving_average in zip(
                                hotkeys, raw_weights
                            )
                        }
                    },
                    timeout=timeout,
                )
        except httpx.ReadTimeout:
            raise PostWeightsError(
                f"failed to post weights - read timeout ({timeout}s)"
            )

        if response.status_code != 200:
            raise PostWeightsError(
                f"failed to post moving averages with status_code "
                f"{response.status_code}: {self._error_response_text(response)}"
            )

    async def update_task_state(
        self, task_id: str, state: TaskState, timeout: int = 3
    ) -> None:
        """Updates image generation task state"""
        try:
            suffix = {
                # ,
                TaskState.FAILED: "fail",
                TaskState.REJECTED: "reject",
            }[state]
        except KeyError:
            logger.warning(f"not updating task state for state {state}")
            return None

        endpoint = f"{self.api_url}/tasks/{task_id}/{suffix}"

        async with self._client() as client:
            response = await client.post(endpoint, timeout=timeout)
        if response.status_code != 200:
            raise UpdateTaskError(
                f"updating task state failed with status_code "
                f"{response.status_code}: {self._error_response_text(response)}"
            )

        return None

    async def upload_scores(
        self,
        scores_upload_request: ScoresUploadRequest,
        timeout: int = 10,
    ) -> None:
        """Upload scores to the backend"""
        try:
            data = scores_upload_request.model_dump()
            async with self._client() as client:
                response = await client.post(
                    f"{self.api_url}/batches/{scores_upload_request.task_id}/scores",
                    json=data,
                    timeout=timeout,
                )
        except httpx.ReadTimeout:
            raise UploadScoresError(
                f"failed to upload scores - read timeout ({timeout}s)"
            )

        if response.status_code != 200:
            raise UploadScoresError(
                f"failed to upload scores with status_code "
                f"{response.status_code}: {self._error_response_text(response)}"
            )

    async def _add_auth_headers(self, request: httpx.Request):
        """Sign request (adding X-Signature and X-Timestamp headers)
        using validator's hotkey
        """
        try:
            request.headers.update(
                {"Authorization": self._get_authorization_header()}
            )
        except Exception as e:
            logger.error(
                f"Exception raised while adding auth headers: {e}; sending request without authentication"
            )

    def _get_authorization_header(self) -> str:
        """Returns authorization header with use of hotkey signing"""
        timestamp = int(time.time())
        token_expiration_time = 60
        message = json.dumps(
            {
                "exp": timestamp + token_expiration_time,
                "iat": timestamp,
                "hotkey": self.hotkey.ss58_address,
            }
        )

        signature = self.hotkey.sign(message.encode())
        signature_base64: str = b64encode(signature).decode()

        message_base64: str = b64encode(message.encode()).decode()

        return f"Hotkey {message_base64}.{signature_base64}"

    async def _include_validator_version(self, request: httpx.Request):
        """Put validator's version in request headers"""
        try:
            from neurons.validator.utils.version import get_validator_version

            request.headers.update(
                {"X-Validator-Version": get_validator_version()}
            )
        except Exception:
            logger.error(
                f"Exception raised while including validator's version"
            )

    def _error_response_text(self, response: httpx.Response):
        if response.status_code == 502:
            return "Bad Gateway"

        # Limit response text to 1024 symbols to prevent spamming in validator
        # logs
        return response.text[:1024]
