import base64
import time
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
    AsyncRetrying,
    RetryError,
)

from neurons.constants import DEV_URL, PROD_URL
from neurons.protocol import denormalize_image_model, ImageGenerationTaskModel
from neurons.validator.backend.exceptions import (
    GetVotesError,
    GetTaskError,
    PostMovingAveragesError,
    PostWeightsError,
    UpdateTaskError,
)
from neurons.validator.backend.models import TaskState


class TensorAlchemyBackendClient:
    def __init__(self, config: bt.config):
        self.config = config

        self.wallet = bt.wallet(config=self.config)
        self.hotkey = self.wallet.hotkey

        self.api_url = DEV_URL if config.subtensor.network == "test" else PROD_URL
        if config.alchemy.force_prod:
            self.api_url = PROD_URL

        logger.info(f"Using backend server {self.api_url}")

        # Setup hooks for all requests to backend
        self.client = httpx.AsyncClient(
            event_hooks={
                "request": [
                    # Add signature to request
                    self._sign_request
                ]
            }
        )

    # Get tasks from the client server
    async def poll_task(self, timeout: int = 60, backoff: int = 1):
        """Performs polling for new task. If no new task found within `timeout`, returns None."""
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_delay(timeout),
                wait=wait_fixed(backoff),
                # Retry if task is not found (returns None)
                retry=retry_if_result(lambda r: r is None),
            ):
                with attempt:
                    return await self.get_task(timeout=1)
        except RetryError:
            logger.info(f"there is no pending task")
            return None

    async def get_task(self, timeout: int = 3) -> ImageGenerationTaskModel | None:
        """Fetch new task from backend.

        Returns task or None if there is no pending task
        """
        try:
            response = await self.client.get(f"{self.api_url}/tasks", timeout=timeout)
        except httpx.ReadTimeout as ex:
            raise GetTaskError(f"/tasks read timeout ({timeout}s)") from ex

        if response.status_code == 200:
            task = response.json()
            logger.info(f"[get_task] task={task}")
            return denormalize_image_model(**task)
        if response.status_code == 404:
            try:
                if response.json().get("code") == "NO_TASKS_FOUND":
                    return None
            except Exception as e:
                pass

        raise GetTaskError(
            f"/tasks failed with status_code {response.status_code}:"
            f" {self._error_response_text(response)}"
        )

    async def get_votes(self, timeout: int = 3) -> Dict:
        """Get human votes from backend"""
        try:
            response = await self.client.get(f"{self.api_url}/votes", timeout=timeout)
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
            response = await self.client.post(
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

    async def post_batch(self, batch: dict, timeout: int = 10) -> Response:
        """Post batch of images"""
        response = await self.client.post(
            f"{self.api_url}/batch",
            json=batch,
            timeout=timeout,
        )
        return response

    async def post_weights(
        self, hotkeys: List[str], raw_weights: torch.Tensor, timeout: int = 10
    ) -> None:
        """Post weights"""
        try:
            response = await self.client.post(
                f"{self.api_url}/validator/weights",
                json={
                    "weights": {
                        hotkey: moving_average.item()
                        for hotkey, moving_average in zip(hotkeys, raw_weights)
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

        response = await self.client.post(endpoint, timeout=timeout)
        if response.status_code != 200:
            raise UpdateTaskError(
                f"updating task state failed with status_code "
                f"{response.status_code}: {self._error_response_text(response)}"
            )

        return None

    async def _sign_request(self, request: httpx.Request):
        """Sign request (adding X-Signature and X-Timestamp headers)
        using validator's hotkey
        """
        try:
            timestamp = str(int(time.time()))
            message = f"{request.method} {request.url}?timestamp={timestamp}"

            signature = self._sign_message(message)

            request.headers.update({"X-Signature": signature, "X-Timestamp": timestamp})
        except Exception as e:
            logger.error(
                f"Exception raised while signing request: {e}; sending plain old request"
            )

        # Print the modified request for debugging
        # logger.info(f"modified request={request}")
        # logger.info(f"modified request headers={request.headers}")

    def _sign_message(self, message: str):
        """Sign message using validator's hotkey"""
        signature = self.hotkey.sign(message.encode())
        return base64.b64encode(signature).decode()

    def _error_response_text(self, response: httpx.Response):
        if response.status_code == 502:
            return "Bad Gateway"

        # Limit response text to 1024 symbols to prevent spamming in validator logs
        return response.text[:1024]
