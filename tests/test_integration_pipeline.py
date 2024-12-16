from unittest.mock import AsyncMock, patch

import pytest
import torch
from loguru import logger
from neurons.protocol import ImageGeneration, ModelType
from neurons.utils.image import image_tensor_to_base64, image_to_tensor
from neurons.validator.backend.exceptions import PostMovingAveragesError
from neurons.validator.forward import update_moving_averages
from scoring.models import RewardModelType
from scoring.pipeline import get_scoring_results
from scoring.types import ScoringResults
from tests.fixtures import TEST_IMAGES, mock_get_config, mock_get_metagraph

import bittensor as bt


class MockScoringModel:
    def load(self, *args, **kwargs):
        return self

    def inference_rank(self, prompt: str, images):
        image_scores = {
            "BLACK": -2.26,
            "WHITE": -1.0,
            "COMPLEX_A": 0.1,
            "COMPLEX_B": 0.2,
            "COMPLEX_C": 0.3,
            "COMPLEX_D": 0.4,
            "COMPLEX_E": 0.5,
            "COMPLEX_F": 0.6,
            "COMPLEX_G": 0.7,
            "REAL_IMAGE_LOW_INFERENCE": 0.8,
            "REAL_IMAGE": 1.27,
        }
        image_tensor = image_to_tensor(images[0])

        for key, score in image_scores.items():
            is_close: bool = torch.allclose(
                image_tensor,
                TEST_IMAGES[key],
                atol=1e-2,
            )

            if image_tensor.shape == TEST_IMAGES[key].shape and is_close:
                return 1, [score]

        raise ValueError(f"Score not set for image {image_tensor}")


def mock_rm(*args, **kwargs):
    return MockScoringModel()


class MockBackendClient:
    def __init__(self, should_fail=False):
        self.should_fail = should_fail
        self.post_moving_averages_called = 0

    async def post_moving_averages(self, hotkeys, moving_average_scores):
        self.post_moving_averages_called += 1
        if self.should_fail:
            raise PostMovingAveragesError("Mocked error")

    async def get_votes(self):
        return {
            "round_1": {"hotkey_0": 1, "hotkey_1": 2},
            "round_2": {"hotkey_2": 3, "hotkey_3": 4},
        }


def mock_get_backend_client():
    return MockBackendClient()


def mock_openai_response():
    return AsyncMock(
        return_value={
            "elements": [
                {"description": "Any Image"},
            ]
        }
    )


def mock_local_get_metagraph(n: int = 11):
    return mock_get_metagraph(n=n)


# Patch configuration
mock_configs = {
    "neurons.config": {
        "get_config": mock_get_config,
        "get_metagraph": mock_local_get_metagraph,
        "get_backend_client": mock_get_backend_client,
    },
    "neurons.validator.forward": {
        "get_config": mock_get_config,
        "get_metagraph": mock_local_get_metagraph,
        "get_backend_client": mock_get_backend_client,
    },
    "neurons.validator.averages": {
        "get_config": mock_get_config,
        "get_metagraph": mock_local_get_metagraph,
        "get_backend_client": mock_get_backend_client,
    },
    "scoring.models.base": {"get_metagraph": mock_local_get_metagraph},
    "scoring.pipeline": {"get_metagraph": mock_local_get_metagraph},
    "scoring.models.rewards.human": {
        "get_backend_client": mock_get_backend_client,
    },
    "scoring.models.rewards.image_reward": {"RM": mock_rm()},
    "scoring.models.masks.duplicate": {
        "get_metagraph": mock_local_get_metagraph,
    },
    "scoring.models.rewards.enhanced_clip.utils": {
        "openai_breakdown": mock_openai_response()
    },
}


def apply_patches(func):
    for module, mocks in mock_configs.items():
        func = patch.multiple(module, **mocks)(func)
    return func


def generate_synapse(hotkey: str, image_content: torch.Tensor) -> bt.Synapse:
    synapse = ImageGeneration(
        seed=-1,
        width=64,
        height=64,
        prompt="lion sitting in jungle",
        generation_type="TEXT_TO_IMAGE",
        model_type=ModelType.ALCHEMY.value,
        images=[image_tensor_to_base64(image_content)],
    )
    synapse.axon = bt.TerminalInfo(hotkey=hotkey)
    return synapse


async def run_pipeline_test():
    responses = []
    image_types = [
        # NOTE: Random order
        #       we will expect the sorted order
        #       of ascending scores later to be alphabetical
        "BLACK",
        "WHITE",
        "COMPLEX_F",
        "REAL_IMAGE_LOW_INFERENCE",
        "REAL_IMAGE",
        "COMPLEX_A",
        "COMPLEX_C",
        "COMPLEX_B",
        "COMPLEX_D",
        "COMPLEX_G",
        "COMPLEX_E",
    ]

    for i, hotkey in enumerate(mock_local_get_metagraph().hotkeys):
        image_content = TEST_IMAGES[image_types[i]]
        responses.append(generate_synapse(hotkey, image_content))

    assert len(responses) == len(image_types), "Incorrect synapse builder"

    results: ScoringResults = await get_scoring_results(
        ModelType.CUSTOM,
        responses[0],
        responses,
    )

    logger.info("Detailed scores for each reward type:")
    for score in results.scores:
        logger.info(f"{score.type}: {score.scores}")

    logger.info(f"Combined scores before filtering: {results.combined_scores}")

    # Verify ImageReward scores
    image_reward_scores = results.get_score(RewardModelType.IMAGE).scores

    assert len(image_reward_scores) == len(
        image_types
    ), "Incorrect length of responses"

    # Define expected order of scores based on image quality
    expected_order = [
        "BLACK",
        "WHITE",
        "COMPLEX_A",
        "COMPLEX_B",
        "COMPLEX_C",
        "COMPLEX_D",
        "COMPLEX_E",
        "COMPLEX_F",
        "COMPLEX_G",
        "REAL_IMAGE_LOW_INFERENCE",
        "REAL_IMAGE",
    ]

    # Get indices that would sort the scores in ascending order
    _, sorted_indices = torch.sort(image_reward_scores, descending=False)

    # Check if the order of hotkeys
    # matches the expected order of image qualities
    for i, (expected_image, actual_index) in enumerate(
        zip(expected_order, sorted_indices)
    ):
        assert expected_image == image_types[actual_index], (
            f"Mismatch in score order. Position {i}: "
            f"Expected {expected_image}, got {image_types[actual_index]}"
        )

        expected_index = image_types.index(expected_image)
        assert actual_index == expected_index, (
            f"Mismatch in index. For {expected_image}: "
            f"Expected index {expected_index}, got {actual_index}"
        )

    logger.info("Image quality order check passed successfully.")

    # Check that all scores are above 1.0 (due to initial seed)
    assert torch.all(
        image_reward_scores >= 1.0
    ), f"All scores should be above 1.0, got: {image_reward_scores}"

    # Check relative relationships
    assert (
        image_reward_scores[4] > image_reward_scores[3]
    ), "REAL_IMAGE score should be higher than REAL_IMAGE_LOW_INFERENCE"
    assert (
        image_reward_scores[0] < image_reward_scores[2]
    ), "BLACK image score should be lower than COMPLEX_A"

    # Check that scores for good images are significantly higher than bad ones
    good_images = image_reward_scores[[2, 3, 4, 5, 7, 8]]
    bad_images = image_reward_scores[[0, 1]]
    assert torch.all(
        good_images > bad_images.max()
    ), "All good image scores should be higher than the highest bad image score"

    # Check that scores are different for different images
    assert len(torch.unique(image_reward_scores)) == len(
        image_reward_scores
    ), "All scores should be unique for different images"

    expected_score_types = {
        RewardModelType.HUMAN,
        RewardModelType.IMAGE,
        RewardModelType.NSFW,
        RewardModelType.BLACKLIST,
        RewardModelType.DUPLICATE,
    }
    actual_score_types = {score.type for score in results.scores}

    for expected_score_type in expected_score_types:
        assert expected_score_type in actual_score_types, (
            f"Missing score type. Expected: {expected_score_type}, "
            f"Got: {actual_score_types}"
        )

    human_scores = results.get_score(RewardModelType.HUMAN).scores
    logger.info(human_scores)
    assert (
        human_scores[0] > 1 and human_scores[1] > human_scores[0]
    ), "Human voting rewards not applied correctly"

    assert results.combined_scores[1] == 0, "Black image not masked"

    for i in range(2, 11):
        assert results.combined_scores[i].item() > 0, (
            f"Valid image {i} incorrectly masked. "
            f"Score: {results.combined_scores[i]}"
        )

    final_rewards = results.combined_scores.clone()

    assert final_rewards[0] == 0, (
        f"Rewards not zeroed for miner exceeding isalive threshold. "
        f"Value: {final_rewards[0]}"
    )
    assert final_rewards[2] != 0, (
        f"Rewards incorrectly zeroed for miner below isalive threshold. "
        f"Value: {final_rewards[2]}"
    )

    for i in range(1, 5):
        if i != 2:
            assert final_rewards[i] == results.combined_scores[i], (
                f"Reward for miner {i} changed unexpectedly. "
                f"Before: {results.combined_scores[i]}, "
                f"After: {final_rewards[i]}"
            )

    non_zero_rewards = final_rewards[final_rewards != 0]
    sorted_rewards, indices = torch.sort(non_zero_rewards, descending=True)
    logger.info(f"Non-zero rewards: {non_zero_rewards}")
    logger.info(f"Sorted non-zero rewards: {sorted_rewards}")
    logger.info(f"Indices of sorted rewards: {indices}")

    assert sorted_rewards[0] > sorted_rewards[-1] * 1.5, (
        f"Highest reward ({sorted_rewards[0]}) is not significantly higher "
        f"than lowest reward ({sorted_rewards[-1]})"
    )

    return results, final_rewards


@pytest.mark.asyncio
@pytest.mark.parametrize("num_runs", [5])
@apply_patches
async def test_full_pipeline_integration_multiple_runs(num_runs):
    all_results = []
    all_final_rewards = []

    for run in range(num_runs):
        logger.info(f"Starting run {run + 1} of {num_runs}")
        results, final_rewards = await run_pipeline_test()
        all_results.append(results)
        all_final_rewards.append(final_rewards)

    for i in range(1, num_runs):
        assert torch.allclose(
            all_final_rewards[0],
            all_final_rewards[i],
            atol=1e-6,
        ), f"Inconsistent results between run 1 and run {i + 1}"

    for run in range(num_runs):
        assert len(all_results[run].scores) == len(
            all_results[0].scores
        ), f"Inconsistent number of score types in run {run + 1}"
        for idx, (score_0, score_run) in enumerate(
            zip(all_results[0].scores, all_results[run].scores)
        ):
            assert score_0.type == score_run.type, (
                f"Mismatched score types at index {idx} in run {run + 1}. "
                f"Expected {score_0.type}, got {score_run.type}"
            )
            assert torch.allclose(
                score_0.scores,
                score_run.scores,
                atol=1e-6,
            ), (
                f"Inconsistent scores for {score_0.type} in run {run + 1}. "
                f"Run 0 scores: {score_0.scores}, "
                f"Run {run + 1} scores: {score_run.scores}"
            )

    logger.info("All runs completed successfully with consistent results!")


@pytest.mark.asyncio
@pytest.mark.parametrize("num_runs", [3])
@apply_patches
async def test_full_pipeline_integration_with_moving_averages(num_runs):
    all_results = []
    all_final_rewards = []
    moving_average_scores = torch.zeros(11)
    ma_history = [moving_average_scores.clone()]
    mock_backend_client = MockBackendClient()
    with patch(
        "neurons.validator.averages.get_backend_client",
        return_value=mock_backend_client,
    ):
        for run in range(num_runs):
            logger.info(f"Starting run {run + 1} of {num_runs}")
            scoring_results, final_rewards = await run_pipeline_test()
            logger.debug(f"Run {run + 1} scoring results: {scoring_results}")
            logger.debug(f"Run {run + 1} final rewards: {final_rewards}")

            all_results.append(scoring_results)
            all_final_rewards.append(final_rewards)

            moving_average_scores = await update_moving_averages(
                moving_average_scores,
                scoring_results,
            )
            ma_history.append(moving_average_scores.clone())
            logger.info(
                f"Run {run + 1} - Updated moving average scores: "
                f"{moving_average_scores}"
            )

    assert (
        mock_backend_client.post_moving_averages_called == num_runs
    ), "Backend client should be called once per run"

    for i in range(1, num_runs):
        assert torch.allclose(
            all_final_rewards[0],
            all_final_rewards[i],
            atol=1e-6,
        ), f"Inconsistent filtered rewards between run 1 and run {i + 1}"

    for run in range(num_runs):
        assert len(all_results[run].scores) == len(
            all_results[0].scores
        ), f"Inconsistent number of score types in run {run + 1}"

        for idx, (score_0, score_run) in enumerate(
            zip(all_results[0].scores, all_results[run].scores)
        ):
            assert score_0.type == score_run.type, (
                f"Mismatched score types at index {idx} in run {run + 1}. "
                f"Expected {score_0.type}, got {score_run.type}"
            )
            assert (
                score_0.scores is not None
            ), f"Score type {score_0.type} returned None scores in run 0"
            assert (
                score_run.scores is not None
            ), f"Score type {score_0.type} returned None scores in run {run + 1}"

            try:
                assert torch.allclose(
                    score_0.scores,
                    score_run.scores,
                    atol=1e-6,
                ), (
                    f"Inconsistent scores for {score_0.type} in run {run + 1}. "
                    f"Run 0 scores: {score_0.scores}, "
                    f"Run {run + 1} scores: {score_run.scores}"
                )
            except AssertionError as e:
                logger.error(f"Assertion failed: {str(e)}")
                logger.info(f"Score type: {score_0.type}")
                logger.info(f"Run 0 scores: {score_0.scores}")
                logger.info(f"Run {run + 1} scores: {score_run.scores}")
                raise

    assert torch.all(
        (moving_average_scores >= 0) & (moving_average_scores <= 1)
    ), f"Final moving average scores not in range [0, 1]: {moving_average_scores}"

    assert not torch.allclose(
        ma_history[0],
        ma_history[-1],
        atol=1e-6,
    ), "Moving averages should change over multiple runs"

    logger.info("All runs completed successfully with consistent results!")
    logger.info(f"Initial moving average scores: {ma_history[0]}")
    logger.info(f"Final moving average scores: {ma_history[-1]}")
