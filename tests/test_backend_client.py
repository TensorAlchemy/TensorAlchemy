import unittest
from unittest.mock import patch, AsyncMock

import torch
from loguru import logger

from neurons.validator.backend.client import TensorAlchemyBackendClient
from neurons.validator.backend.exceptions import (
    GetVotesError,
    GetTaskError,
    PostMovingAveragesError,
    PostWeightsError,
    UpdateTaskError,
)
from neurons.validator.backend.models import TaskState
from neurons.protocol import ImageGenerationTaskModel
import bittensor as bt

from neurons.validator.schemas import Batch


class TestTensorAlchemyBackendClient(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        mnemonic = bt.Keypair.generate_mnemonic(12)
        self.client = TensorAlchemyBackendClient(
            hotkey=bt.Keypair.create_from_mnemonic(mnemonic)
        )

    @patch("httpx.AsyncClient.get")
    async def test_get_task_success(self, mock_get):
        task_data = {
            "id": "1111",
            "image_count": 1,
            "prompt": "test",
            "height": 1024,
            "width": 1024,
            "guidance_scale": 3,
            "seed": 1,
            "steps": 50,
            "task_type": "TEXT_TO_IMAGE",
        }
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json = lambda: task_data
        mock_get.return_value = mock_response

        result = await self.client.get_task()
        self.assertIsInstance(result, ImageGenerationTaskModel)
        self.assertEqual(result.task_id, task_data["id"])

    @patch("httpx.AsyncClient.get")
    async def test_get_task_no_task(self, mock_get):
        mock_response = AsyncMock()
        mock_response.status_code = 404
        mock_response.json = lambda: {"code": "NO_TASKS_FOUND"}
        mock_get.return_value = mock_response

        result = await self.client.get_task()
        self.assertIsNone(result)

    @patch("httpx.AsyncClient.get")
    async def test_get_task_error(self, mock_get):
        mock_response = AsyncMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_get.return_value = mock_response

        with self.assertRaises(GetTaskError):
            await self.client.get_task()

    @patch("httpx.AsyncClient.get")
    async def test_get_votes_success(self, mock_get):
        votes_data = [{"miner_hotkey": "1234"}]
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json = lambda: votes_data
        mock_get.return_value = mock_response

        result = await self.client.get_votes()
        self.assertEqual(result, votes_data)

    @patch("httpx.AsyncClient.get")
    async def test_get_votes_error(self, mock_get):
        mock_response = AsyncMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_get.return_value = mock_response

        with self.assertRaises(GetVotesError):
            await self.client.get_votes()

    @patch("httpx.AsyncClient.post")
    async def test_post_moving_averages_success(self, mock_post):
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        hotkeys = ["hotkey1", "hotkey2"]
        moving_average_scores = torch.tensor([0.5, 0.6])

        await self.client.post_moving_averages(hotkeys, moving_average_scores)
        mock_post.assert_called_once()

    @patch("httpx.AsyncClient.post")
    async def test_post_moving_averages_error(self, mock_post):
        mock_response = AsyncMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response

        hotkeys = ["hotkey1", "hotkey2"]
        moving_average_scores = torch.tensor([0.5, 0.6])

        with self.assertRaises(PostMovingAveragesError):
            await self.client.post_moving_averages(hotkeys, moving_average_scores)

    @patch("httpx.AsyncClient.post")
    async def test_post_weights(self, mock_post):
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        hotkeys = ["hotkey1", "hotkey2"]
        weights = torch.tensor([0.5, 0.6])

        await self.client.post_weights(hotkeys, weights)
        mock_post.assert_called_once()

    @patch("httpx.AsyncClient.post")
    async def test_post_weights_error(self, mock_post):
        mock_response = AsyncMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response

        hotkeys = ["hotkey1", "hotkey2"]
        weights = torch.tensor([0.5, 0.6])

        with self.assertRaises(PostWeightsError):
            await self.client.post_weights(hotkeys, weights)

    @patch("httpx.AsyncClient.post")
    async def test_update_task_state_success(self, mock_post):
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        task_id = "1234"
        state = TaskState.FAILED

        await self.client.update_task_state(task_id, state)
        mock_post.assert_called_once()

    @patch("httpx.AsyncClient.post")
    async def test_update_task_state_error(self, mock_post):
        mock_response = AsyncMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response

        task_id = "1234"
        state = TaskState.FAILED

        with self.assertRaises(UpdateTaskError):
            await self.client.update_task_state(task_id, state)

    @patch("httpx.AsyncClient.post")
    async def test_post_batch_success(self, mock_post):
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        batch = Batch(
            prompt="test",
            computes=[],
            batch_id="1234",
            nsfw_scores=[],
            blacklist_scores=[],
            should_drop_entries=[],
            validator_hotkey="fake_hotkey",
            miner_hotkeys=[],
            miner_coldkeys=[],
        )

        await self.client.post_batch(batch)
        mock_post.assert_called_once()
