import base64
import copy
import uuid
from io import BytesIO

import bittensor as bt
import pytest
import torch
import torchvision.transforms as T
from substrateinterface import Keypair

from neurons.constants import DEV_URL, PROD_URL
from neurons.validator.backend.client import TensorAlchemyBackendClient
from neurons.validator.config import add_args, check_config, config
from neurons.validator.rewards.reward import HumanValidationRewardModel


class Neuron:
    @classmethod
    def check_config(cls, config: "bt.Config"):
        check_config(cls, config)

    @classmethod
    def config(cls):
        return config(cls)

    @classmethod
    def add_args(cls, parser):
        add_args(cls, parser)

    def __init__(self):
        self.config = Neuron.config()
        self.config.wallet.name = "validator"
        self.config.wallet.hotkey = "default"
        self.check_config(self.config)

    def load_state(self, path, moving_average_scores, device, metagraph):
        try:
            r"""Load hotkeys and moving average scores from filesystem."""
            state_dict = torch.load(f"{path}/model.torch")
            neuron_weights = torch.tensor(state_dict["neuron_weights"])

            has_nans = torch.isnan(neuron_weights).any()
            has_infs = torch.isinf(neuron_weights).any()

            # Check to ensure that the size of the neruon weights matches the metagraph size.
            if neuron_weights.shape < (metagraph.n,):
                moving_average_scores[: len(neuron_weights)] = neuron_weights.to(device)

            # Check for nans in saved state dict
            elif not any([has_nans, has_infs]):
                moving_average_scores = neuron_weights.to(device)

            # Zero out any negative scores
            for i, average in enumerate(moving_average_scores):
                if average < 0:
                    moving_average_scores[i] = 0

        except Exception as e:
            moving_average_scores = moving_average_scores

        return moving_average_scores


neuron: Neuron = None

pytest.skip(allow_module_level=True)


@pytest.fixture(autouse=True, scope="session")
def setup() -> None:
    global neuron

    neuron = Neuron()


def get_netuid(network):
    if network == "test":
        return 25
    else:
        return 26


class MockWallet:
    hotkey: Keypair

    def __init__(self):
        self.hotkey = Keypair()


class MockValidator:
    wallet: MockWallet

    def __init__(self):
        self.wallet = MockWallet()


def get_url(network):
    api_url = DEV_URL if network == "test" else PROD_URL
    return api_url


def get_args(
    network, neuron
) -> Tuple[
    "bt.metagraph.Metagraph", TensorAlchemyBackendClient, torch.Tensor, List[str]
]:
    neuron.config.netuid = get_netuid(network)
    neuron.config.subtensor.network = network
    subtensor = bt.subtensor(config=neuron.config)
    metagraph = bt.metagraph(
        netuid=neuron.config.netuid, network=neuron.config.subtensor.network, sync=False
    )
    metagraph.sync(subtensor=subtensor)
    moving_averages = torch.zeros(metagraph.n).to(neuron.config.device)
    moving_averages = neuron.load_state(
        neuron.config.alchemy.full_path,
        moving_averages,
        neuron.config.device,
        metagraph,
    )
    hotkeys = copy.deepcopy(metagraph.hotkeys)

    backend_client = TensorAlchemyBackendClient(neuron.config)

    return metagraph, backend_client, moving_averages, hotkeys


def create_dummy_batches(metagraph):
    uids = [0, 1, 2, 3, 4, 5]

    images = []
    for _ in uids:
        im_file = BytesIO()
        T.transforms.ToPILImage()(
            torch.full([3, 1024, 1024], 254, dtype=torch.float)
        ).save(im_file, format="PNG")
        im_bytes = im_file.getvalue()
        im_b64 = base64.b64encode(im_bytes)
        images.append(im_b64.decode())

    batches = [
        {
            "batch_id": str(uuid.uuid4()),
            "validator_hotkey": "5Cv9sBYUsif5rgkUbZfaQAVzBbnb9rZAUTNsyx8Eitzk9MA9",
            "prompt": "test",
            "nsfw_scores": [1 for _ in uids],
            "blacklist_scores": [1 for _ in uids],
            "miner_hotkeys": [metagraph.hotkeys[uid] for uid in uids],
            "miner_coldkeys": [metagraph.coldkeys[uid] for uid in uids],
            "computes": images,
            "should_drop_entries": [0 for uid in uids],
        }
    ]
    return batches


@pytest.mark.parametrize("network", ["test", "finney"])
async def test_post_moving_averages(network):
    # TODO: not sure how we should update e2e tests for signed requests;
    #  one option is to create a test validator and use it
    _, backend_client, moving_averages, hotkeys = get_args(network, neuron)
    response = await backend_client.post_moving_averages(hotkeys, moving_averages)
    assert response == True


@pytest.mark.parametrize("network", ["test", "finney"])
async def test_post_weights(network):
    _, backend_client, moving_averages, hotkeys = get_args(network, neuron)
    raw_weights = torch.nn.functional.normalize(moving_averages, p=1, dim=0)
    response = await backend_client.post_weights(hotkeys, raw_weights)
    assert response.status_code == 200


@pytest.mark.parametrize("network", ["test", "finney"])
async def test_submit_batch(network):
    metagraph, backend_client, _, _ = get_args(network, neuron)
    dummy_batch = create_dummy_batches(metagraph)
    response = await backend_client.post_batch(backend_client, dummy_batch[0])
    assert response.status_code == 200


@pytest.mark.parametrize("network", ["test", "finney"])
def test_get_votes(network):
    metagraph, backend_client, _, _ = get_args(network, neuron)
    hv_reward_model = HumanValidationRewardModel(metagraph, backend_client)
    human_voting_scores = backend_client.get_votes()
    assert human_voting_scores.status_code == 200
    assert human_voting_scores.json() != {}
