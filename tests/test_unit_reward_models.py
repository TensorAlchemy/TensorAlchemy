from io import BytesIO

import pytest
import requests
import torch
import torchvision.transforms as transforms
from PIL import Image

import bittensor as bt

from neurons.protocol import ImageGeneration, ModelType
from neurons.validator.reward import BlacklistFilter, NSFWRewardModel

blacklist_reward_model: BlacklistFilter = None
nsfw_reward_model: NSFWRewardModel = None


@pytest.fixture(autouse=True, scope="session")
def setup() -> None:
    global blacklist_reward_model, nsfw_reward_model

    blacklist_reward_model = BlacklistFilter()
    nsfw_reward_model = NSFWRewardModel()


async def test_black_image():
    responses = [
        ImageGeneration(
            generation_type="TEXT_TO_IMAGE",
            seed=-1,
            model_type=ModelType.ALCHEMY,
            images=[
                bt.Tensor.serialize(torch.full([3, 1024, 1024], 254, dtype=torch.float))
            ],
        ),
        ImageGeneration(
            generation_type="TEXT_TO_IMAGE",
            seed=-1,
            model_type=ModelType.ALCHEMY,
            images=[
                bt.Tensor.serialize(torch.full([3, 1024, 1024], 0, dtype=torch.float))
            ],
        ),
    ]
    rewards = await blacklist_reward_model.get_rewards(
        responses, rewards=torch.ones(len(responses))
    )
    assert rewards[0].item() == 1
    assert rewards[1].item() == 0


async def test_incorrect_image_size():
    responses = [
        ImageGeneration(
            generation_type="TEXT_TO_IMAGE",
            seed=-1,
            model_type=ModelType.ALCHEMY,
            images=[
                bt.Tensor.serialize(torch.full([3, 1024, 1024], 254, dtype=torch.float))
            ],
        ),
        ImageGeneration(
            generation_type="TEXT_TO_IMAGE",
            seed=-1,
            model_type=ModelType.ALCHEMY,
            images=[
                bt.Tensor.serialize(torch.full([3, 100, 1024], 254, dtype=torch.float))
            ],
        ),
    ]
    rewards = await blacklist_reward_model.get_rewards(
        responses, rewards=torch.ones(len(responses))
    )
    assert rewards[0].item() == 1
    assert rewards[1].item() == 0


async def test_nsfw_image():
    nsfw_image_url = "https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/a05eaa75-ac8c-4460-b6b0-b7eb47e06987/width=1024/00027-4120052916.jpeg"
    transform = transforms.Compose([transforms.PILToTensor()])
    response_nsfw = ImageGeneration(
        generation_type="TEXT_TO_IMAGE",
        seed=-1,
        model_type=ModelType.ALCHEMY,
        prompt="An nsfw woman.",
        images=[
            bt.Tensor.serialize(
                transform(Image.open(BytesIO(requests.get(nsfw_image_url).content)))
            )
        ],
    )
    response_no_nsfw = ImageGeneration(
        generation_type="TEXT_TO_IMAGE",
        seed=-1,
        model_type=ModelType.ALCHEMY,
        prompt="A majestic lion jumping from a big stone at night",
        images=[bt.Tensor.serialize(transform(Image.open(r"tests/non_nsfw.jpeg")))],
    )
    responses = [response_nsfw, response_no_nsfw]
    rewards = await nsfw_reward_model.get_rewards(
        responses, rewards=torch.ones(len(responses))
    )
    assert rewards[0].item() == 0
    assert rewards[1].item() == 1
