import json
from typing import Dict, List

import httpx
import bittensor as bt
import torch
from httpx import Response
from loguru import logger

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

        self.api_url = DEV_URL if config.subtensor.network == "test" else PROD_URL
        if config.alchemy.force_prod:
            self.api_url = PROD_URL

        logger.info(f"Using backend server {self.api_url}")

    async def get_task(self, timeout=3) -> ImageGenerationTaskModel | None:
        """Fetch new task from backend.

        Returns task or None if there is no pending task
        """
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.api_url}/tasks", timeout=timeout)
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
                f"/tasks failed with status_code {response.status_code}: {response.text}"
            )

        return None

    async def task_reject(self, task_id: str) -> None:
        return await self._update_task_state(task_id, TaskState.REJECTED)

    async def task_fail(self, task_id: str) -> None:
        return await self._update_task_state(task_id, TaskState.FAILED)

    async def get_votes(self, timeout=3) -> Dict:
        """Get human votes from backend"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.api_url}/votes", timeout=timeout)
            if response.status_code != 200:
                raise GetVotesError(
                    f"/votes failed with status_code {response.status_code}: {response.text}"
                )
            return response.json()

    async def post_moving_averages(
        self,
        hotkeys: List[str],
        moving_average_scores: torch.Tensor,
        timeout=10,
    ) -> None:
        """Post moving averages"""
        async with httpx.AsyncClient() as client:
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
            if response.status_code != 200:
                raise PostMovingAveragesError(
                    f"failed to post moving averages with status_code "
                    f"{response.status_code}: {response.text}"
                )

    async def post_batch(self, batch: dict, timeout=10) -> Response:
        """Post batch of images"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.api_url}/batch",
                json=batch,
                timeout=timeout,
            )
            return response

    async def post_weights(
        self, hotkeys: List[str], raw_weights: torch.Tensor, timeout=10
    ) -> None:
        """Post weights"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.api_url}/validator/weights",
                json={
                    "weights": {
                        hotkey: moving_average.item()
                        for hotkey, moving_average in zip(hotkeys, raw_weights)
                    }
                },
                timeout=timeout,
            )
            if response.status_code != 200:
                raise PostWeightsError(
                    f"failed to post moving averages with status_code "
                    f"{response.status_code}: {response.text}"
                )

    async def _update_task_state(
        self, task_id: str, state: TaskState, timeout=3
    ) -> None:
        endpoint = f"{self.api_url}/tasks/{task_id}"

        if state == TaskState.FAILED:
            endpoint = f"{endpoint}/fail"
        elif endpoint == TaskState.REJECTED:
            endpoint = f"{endpoint}/reject"

        async with httpx.AsyncClient() as client:
            response = await client.get(endpoint, timeout=timeout)
            if response.status_code != 200:
                raise UpdateTaskError(
                    f"updating task state failed with status_code "
                    f"{response.status_code}: {response.text}"
                )

        return None
