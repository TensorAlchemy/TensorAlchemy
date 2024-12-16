import pytest
from unittest.mock import MagicMock, patch

from neurons.miners.base.miner import BaseMiner
from neurons.protocol import IsAlive, ImageGeneration, ModelType


from neurons.miners.base.models import MinerState

class MockBaseMiner(BaseMiner):
    def __init__(self, **kwargs):
            # Initialize with MinerState which contains request_stats
            self.state = MinerState()
            self.axon = MagicMock()
            super().__init__(**kwargs)
        
    def initialize_implementation(self):
        """Mock implementation initialization"""
        pass

    def create_attachments(self):
        """Mock attachment creation"""
        pass

    async def _attempt_generate_images(self, synapse, model_config):
        return ["mock_image_1", "mock_image_2"]


from unittest.mock import AsyncMock
class TestBaseMiner:
    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.axon = MagicMock()
        config.axon.port = 1234
        config.axon.get.return_value = None
        config.netuid = 1
        return config

    @pytest.fixture
    def mock_metagraph(self):
        metagraph = MagicMock()
        metagraph.hotkeys = ["test_hotkey_1", "test_hotkey_2"]
        metagraph.S = [100.0, 200.0]
        metagraph.uids = [1, 2]
        return metagraph

    @pytest.fixture
    def mock_subtensor(self):
        subtensor = MagicMock()
        return subtensor

    @pytest.fixture
    def mock_wallet(self):
        wallet = MagicMock()
        return wallet

    @pytest.fixture
    def mock_components(
        self, mock_config, mock_metagraph, mock_subtensor, mock_wallet
    ):
        with patch(
            "neurons.config.get_config",
            return_value=mock_config,
        ), patch(
            "neurons.config.get_wallet",
            return_value=mock_wallet,
        ), patch(
            "neurons.config.get_subtensor",
            return_value=mock_subtensor,
        ), patch(
            "neurons.config.get_metagraph",
            return_value=mock_metagraph,
        ), patch(
            "neurons.miners.base.miner.get_config",
            return_value=mock_config,
        ), patch(
            "neurons.miners.base.miner.get_wallet",
            return_value=mock_wallet,
        ), patch(
            "neurons.miners.base.miner.get_subtensor",
            return_value=mock_subtensor,
        ), patch(
            "neurons.miners.base.miner.get_metagraph",
            return_value=mock_metagraph,
        ):
            yield {
                "config": mock_config,
                "wallet": mock_wallet,
                "subtensor": mock_subtensor,
                "metagraph": mock_metagraph,
            }

    @pytest.fixture
    def base_miner(self, mock_components):
        with patch.object(BaseMiner, "loop", return_value=None), \
             patch("bittensor.axon") as mock_axon, \
             patch.object(BaseMiner, "initialize_components", return_value=None), \
             patch.object(BaseMiner, "start", return_value=None), \
             patch("neurons.miners.base.miner.get_metagraph", return_value=mock_components["metagraph"]):
            
            mock_axon.return_value.attach.return_value.start.return_value = mock_axon.return_value
            miner = MockBaseMiner()
            miner.axon = mock_axon.return_value
            yield miner

def test_initialize_components(self, base_miner):
        assert base_miner.background_timer is not None
        
    @patch("neurons.miners.base.miner.get_coldkey_for_hotkey")
    @patch("neurons.miners.base.miner.get_stake_for_hotkey")  
    async def test_base_blacklist(self, mock_get_stake, mock_get_coldkey, base_miner):
        mock_get_coldkey.return_value = "test_coldkey"
        mock_get_stake.return_value = 1000

        synapse = MagicMock(spec=ImageGeneration)
        synapse.dendrite = MagicMock() 
        synapse.dendrite.hotkey = "test_hotkey"

        # Test case where coldkey is whitelisted
        base_miner.state.request_stats = {}
        base_miner.state.coldkey_whitelist = {"test_coldkey"}
        is_blacklisted, reason = await base_miner._base_blacklist(synapse)
        assert not is_blacklisted

    @patch("neurons.miners.base.miner.get_coldkey_for_hotkey")
    @patch("neurons.miners.base.miner.get_caller_stake")
    def test_base_blacklist(
        self, mock_get_caller_stake, mock_get_coldkey, base_miner
    ):
        mock_get_coldkey.return_value = "test_coldkey"
        mock_get_caller_stake.return_value = 1000

        synapse = MagicMock(spec=ImageGeneration)
        synapse.dendrite = MagicMock()
        synapse.dendrite.hotkey = "test_hotkey"

        base_miner.coldkey_whitelist = {"test_coldkey"}
        is_blacklisted, reason = base_miner._base_blacklist(synapse)
        assert not is_blacklisted
        assert reason == "Whitelisted coldkey recognized."

        base_miner.coldkey_whitelist = set()
        base_miner.hotkey_whitelist = {"test_hotkey"}
        is_blacklisted, reason = base_miner._base_blacklist(synapse)
        assert not is_blacklisted
        assert reason == "Whitelisted hotkey recognized."

        base_miner.hotkey_whitelist = set()
        mock_get_caller_stake.return_value = None
        is_blacklisted, reason = base_miner._base_blacklist(synapse)
        assert is_blacklisted
        assert "Blacklisted a non-registered hotkey's" in reason



    @patch("neurons.miners.base.miner.get_subtensor")
    @patch("neurons.miners.base.miner.get_config")
    def test_register_axon(
        self, mock_get_config, mock_get_subtensor, base_miner
    ):
        base_miner.axon = MagicMock()
        mock_subtensor = MagicMock()
        mock_get_subtensor.return_value = mock_subtensor
        mock_get_config.return_value.netuid = 1

        base_miner.register_axon()
        mock_subtensor.serve_axon.assert_called_once_with(
            axon=base_miner.axon, netuid=1
        )

    @pytest.mark.asyncio
    async def test_is_alive(self, base_miner):
        synapse = MagicMock(spec=IsAlive)
        result = await base_miner.is_alive(synapse)
        assert result.completion == "True"

    def test_get_miner_index(self, base_miner, mock_components):
        mock_components["wallet"].hotkey.ss58_address = "test_hotkey_1"
        assert base_miner.get_miner_index() == 0

        mock_components["wallet"].hotkey.ss58_address = "non_existent_hotkey"
        assert base_miner.get_miner_index() is None

    def test_check_still_registered(self, base_miner):
        with patch.object(BaseMiner, "get_miner_index", return_value=5):
            assert base_miner.check_still_registered() is True
        with patch.object(BaseMiner, "get_miner_index", return_value=None):
            assert base_miner.check_still_registered() is False

    @pytest.mark.asyncio
    async def test_generate_image(self, base_miner):
        synapse = MagicMock(spec=ImageGeneration)
        synapse.model_type = ModelType.CUSTOM
        synapse.generation_type = "TEXT_TO_IMAGE"
        synapse.timeout = 60

        result = await base_miner.generate_image(synapse)

        assert result == synapse
        assert result.images == ["mock_image_1", "mock_image_2"]

    def test_log_generation_time(self, base_miner):
        start_time = 1000
        base_miner.stats.total_requests = 1
        base_miner.stats.generation_time = 0

        with patch("time.perf_counter", return_value=1010):
            base_miner._log_generation_time(start_time)

        assert base_miner.stats.generation_time == 10
