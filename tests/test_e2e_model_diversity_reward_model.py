import pytest
import torch
from neurons.protocol import ImageGeneration, ModelType
from neurons.validator.rewards.reward import ModelDiversityRewardModel
from neurons.validator.utils import get_promptdb_backup

reward_model = ModelDiversityRewardModel()
prompt_history_db = get_promptdb_backup(netuid=25, limit=10)

pytest.skip(allow_module_level=True)


@pytest.mark.parametrize("prompt", prompt_history_db)
def test_synapse_default(prompt):
    synapse_benchmark = ImageGeneration(
        generation_type="text_to_image",
        prompt=prompt[0],
        seed=-1,
        model_type=ModelType.alchemy.value.lower(),
    )

    synapse_benchmark_duplicate = ImageGeneration(
        generation_type="text_to_image",
        prompt=prompt[0],
        seed=-1,
        model_type=ModelType.alchemy.value.lower(),
    )

    responses = [reward_model.generate_image(synapse_benchmark_duplicate)]
    rewards = reward_model.get_rewards(
        responses, rewards=torch.zeros(len(responses)), synapse=synapse_benchmark
    )

    assert rewards[0].item() == 1


@pytest.mark.parametrize("prompt", prompt_history_db)
def test_synapse_wrong_seed(prompt):
    synapse_benchmark = ImageGeneration(
        generation_type="text_to_image",
        prompt=prompt[0],
        seed=-1,
        model_type=ModelType.alchemy.value.lower(),
    )

    synapse_wrong_seed = ImageGeneration(
        generation_type="text_to_image",
        prompt=prompt[0],
        seed=3,
        model_type=ModelType.alchemy.value.lower(),
    )

    responses = [reward_model.generate_image(synapse_wrong_seed)]
    rewards = reward_model.get_rewards(
        responses, rewards=torch.zeros(len(responses)), synapse=synapse_benchmark
    )

    assert rewards[0].item() == 0


@pytest.mark.parametrize("prompt", prompt_history_db)
def test_synapse_low_steps(prompt):
    synapse_benchmark = ImageGeneration(
        generation_type="text_to_image",
        prompt=prompt[0],
        seed=-1,
        model_type=ModelType.alchemy.value.lower(),
    )

    synapse_low_steps = ImageGeneration(
        generation_type="text_to_image",
        prompt=prompt[0],
        steps=10,
        model_type=ModelType.alchemy.value.lower(),
    )

    responses = [reward_model.generate_image(synapse_low_steps)]
    rewards = reward_model.get_rewards(
        responses, rewards=torch.zeros(len(responses)), synapse=synapse_benchmark
    )

    assert rewards[0].item() == 0
