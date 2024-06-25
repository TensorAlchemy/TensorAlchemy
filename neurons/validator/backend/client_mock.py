from unittest.mock import AsyncMock, MagicMock


class MockTensorAlchemyBackendClient:
    def __init__(self):
        self.config = MagicMock()
        self.wallet = MagicMock()
        self.hotkey = MagicMock()
        self.api_url = "http://mock-api-url.com"

    async def poll_task(
        self,
        _timeout: int = 60,
        _backoff: int = 1,
    ):
        return AsyncMock()()

    async def get_task(
        self,
        _timeout: int = 3,
    ):
        return AsyncMock()()

    async def get_votes(
        self,
        _timeout: int = 3,
    ):
        return AsyncMock()()

    async def post_moving_averages(
        self,
        _hotkeys,
        _moving_average_scores,
        _timeout: int = 10,
    ):
        return AsyncMock()()

    async def post_batch(
        self,
        _batch,
        _timeout: int = 10,
    ):
        return AsyncMock()()

    async def post_weights(
        self,
        _hotkeys,
        _raw_weights,
        _timeout: int = 10,
    ):
        return AsyncMock()()

    async def update_task_state(
        self,
        _task_id: str,
        _state,
        _timeout: int = 3,
    ):
        return AsyncMock()()

    def _sign_message(self, _message: str):
        return "mocked_signature"
