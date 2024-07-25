from typing import List, Any, Tuple
from unittest.mock import patch, MagicMock, AsyncMock

import bittensor as bt
import pytest
import torch
from PIL.Image import Image as ImageType
from loguru import logger

from neurons.protocol import ImageGeneration, ModelType
from neurons.utils.image import image_tensor_to_base64, image_to_tensor
from neurons.validator.backend.exceptions import PostMovingAveragesError
from neurons.validator.forward import update_moving_averages
from neurons.validator.scoring.models import RewardModelType
from neurons.validator.scoring.pipeline import (
    get_scoring_results,
    filter_rewards,
)
from neurons.validator.scoring.types import ScoringResults
from tests.fixtures import TEST_IMAGES


# Mock functions
def mock_metagraph():
    test_uids = torch.tensor([0, 1, 2, 3, 4])
    test_hotkeys = [f"hotkey_{uid.item()}" for uid in test_uids]
    to_return = MagicMock()
    to_return.n = len(test_hotkeys)
    to_return.hotkeys = test_hotkeys
    return to_return


def mock_backend_client():
    mock_client = AsyncMock()
    mock_votes = {
        "round_1": {
            "hotkey_0": 1,
            "hotkey_1": 2,
        },
        "round_2": {
            "hotkey_2": 3,
            "hotkey_3": 4,
        },
    }
    mock_client.get_votes.return_value = mock_votes
    return mock_client


mock_meta = mock_metagraph()
mock_client = mock_backend_client()


class MockScoringModel:
    """Mock RM.load from ImageReward"""

    def inference_rank(
        self, prompt: str, images: List[ImageType]
    ) -> Tuple[Any, float]:
        logger.error(f"prompt: {prompt}, images: {images}")
        image = images[0]
        score = 0.0
        # Those scores are real
        image_scores = {
            "BLACK": -2.26,
            "SOLID_COLOR_FILLED": -2.27,
            "REAL_IMAGE": 1.27,
            "REAL_IMAGE_LOW_INFERENCE": 0.9,
        }
        image_tensor = image_to_tensor(image)

        for key in image_scores.keys():
            # Find matching image
            if image_tensor.shape == TEST_IMAGES[key].shape and torch.allclose(
                image_tensor, TEST_IMAGES[key], atol=1e-2
            ):
                return (1, image_scores[key])

        raise ValueError(f"Score not set for image {image_tensor}")


def mock_imagereward_load(*args, **kwargs):
    return MockScoringModel()


def generate_synapse(hotkey: str, image_content: torch.Tensor) -> bt.Synapse:
    synapse = ImageGeneration(
        prompt="lion sitting in jungle",
        generation_type="TEXT_TO_IMAGE",
        seed=-1,
        model_type=ModelType.ALCHEMY.value,
        images=[image_tensor_to_base64(image_content)],
    )
    synapse.axon = bt.TerminalInfo(hotkey=hotkey)
    return synapse


async def run_pipeline_test():
    # Generate test responses
    responses = []
    for i, hotkey in enumerate(mock_meta.hotkeys):
        if i == 0:
            image_content = TEST_IMAGES["SOLID_COLOR_FILLED"]
        elif i == 1:
            image_content = TEST_IMAGES["BLACK"]
        elif i == 2:
            image_content = TEST_IMAGES["SOLID_COLOR_FILLED"]
        elif i == 3:
            image_content = TEST_IMAGES["REAL_IMAGE"]
        elif i == 4:
            image_content = TEST_IMAGES["REAL_IMAGE_LOW_INFERENCE"]
        responses.append(generate_synapse(hotkey, image_content))

    # Run the full pipeline
    results: ScoringResults = await get_scoring_results(
        ModelType.CUSTOM, responses[0], responses
    )

    logger.info("Detailed scores for each reward type:")
    for score in results.scores:
        logger.info("{}: {}".format(score.type, score.scores))

    logger.info(
        "Combined scores before filtering: {}".format(results.combined_scores)
    )

    # Check if we have all expected score types
    expected_score_types = {
        RewardModelType.HUMAN,
        RewardModelType.IMAGE,
        RewardModelType.NSFW,
        RewardModelType.BLACKLIST,
    }
    actual_score_types = {score.type for score in results.scores}
    for expected_score_type in expected_score_types:
        assert (
            expected_score_type in actual_score_types
        ), "Missing score type. Expected to contain: {}, Got: {}".format(
            expected_score_type, actual_score_types
        )

    # Check human voting rewards
    human_scores = results.get_score(RewardModelType.HUMAN).scores
    assert (
        human_scores[0] > 1 and human_scores[1] > human_scores[0]
    ), "Human voting rewards not applied correctly"

    # Check masking
    assert results.combined_scores[1] == 0, "Black image not masked"

    # Check if other scores are non-zero for valid images
    for i in [0, 2, 3, 4]:
        assert (
            results.combined_scores[i] != 0
        ), "Valid image {} incorrectly masked. Score: {}".format(
            i, results.combined_scores[i]
        )

    # Test reward filtering
    isalive_dict = {0: 10, 2: 5}  # Miner 0 exceeds threshold, 2 doesn't
    isalive_threshold = 8
    filtered_rewards = filter_rewards(
        isalive_dict, isalive_threshold, results.combined_scores
    )

    logger.info("isalive_dict: {}".format(isalive_dict))
    logger.info("isalive_threshold: {}".format(isalive_threshold))
    logger.info("Final filtered rewards: {}".format(filtered_rewards))

    assert (
        filtered_rewards[0] == 0
    ), "Rewards not zeroed for miner exceeding isalive threshold. Value: {}".format(
        filtered_rewards[0]
    )
    assert (
        filtered_rewards[2] != 0
    ), "Rewards incorrectly zeroed for miner below isalive threshold. Value: {}".format(
        filtered_rewards[2]
    )

    # Check if miners not in isalive_dict retain their original scores
    for i in range(1, 5):  # Checking miners 1, 3, and 4
        if i != 2:  # Skip 2 as it's in isalive_dict
            assert (
                filtered_rewards[i] == results.combined_scores[i]
            ), "Reward for miner {} changed unexpectedly. Before: {}, After: {}".format(
                i, results.combined_scores[i], filtered_rewards[i]
            )

    # Check if final scores are within the correct range (0 to 1)
    assert torch.all(
        (filtered_rewards >= 0) & (filtered_rewards <= 1)
    ), "Final rewards not in range [0, 1]: {}".format(filtered_rewards)

    # Analyze the order of rewards
    non_zero_rewards = filtered_rewards[filtered_rewards != 0]
    sorted_rewards, indices = torch.sort(non_zero_rewards, descending=True)
    logger.info("Non-zero rewards: {}".format(non_zero_rewards))
    logger.info("Sorted non-zero rewards: {}".format(sorted_rewards))
    logger.info("Indices of sorted rewards: {}".format(indices))

    # Instead of asserting the exact order, we can check if the highest reward is significantly higher than the lowest
    assert (
        sorted_rewards[0] > sorted_rewards[-1] * 1.5
    ), "Highest reward ({}) is not significantly higher than lowest reward ({})".format(
        sorted_rewards[0], sorted_rewards[-1]
    )

    return results, filtered_rewards


class MockBackendClient:
    def __init__(self, should_fail=False):
        self.should_fail = should_fail
        self.post_moving_averages_called = 0

    async def post_moving_averages(self, hotkeys, moving_average_scores):
        self.post_moving_averages_called += 1
        if self.should_fail:
            raise PostMovingAveragesError("Mocked error")


@pytest.mark.asyncio
@patch(
    "neurons.validator.config.get_metagraph",
    return_value=mock_meta,
)
@patch(
    "neurons.validator.config.get_backend_client",
    return_value=mock_client,
)
@patch(
    "neurons.validator.forward.get_metagraph",
    return_value=mock_meta,
)
@patch(
    "neurons.validator.scoring.models.base.get_metagraph",
    return_value=mock_meta,
)
@patch(
    "neurons.validator.scoring.pipeline.get_metagraph",
    return_value=mock_meta,
)
@patch(
    "neurons.validator.scoring.models.rewards.human.get_metagraph",
    return_value=mock_meta,
)
@patch(
    "neurons.validator.scoring.models.rewards.human.get_backend_client",
    return_value=mock_client,
)
@patch(
    "neurons.validator.scoring.models.rewards.image_reward.RM.load",
    mock_imagereward_load,
)
async def test_full_pipeline_integration_multiple_runs(*mocks):
    num_runs = 5
    all_results = []
    all_filtered_rewards = []

    for run in range(num_runs):
        logger.info("Starting run {} of {}".format(run + 1, num_runs))
        results, filtered_rewards = await run_pipeline_test()
        all_results.append(results)
        all_filtered_rewards.append(filtered_rewards)

    # Check consistency across runs
    for i in range(1, num_runs):
        assert torch.allclose(
            all_filtered_rewards[0], all_filtered_rewards[i], atol=1e-6
        ), "Inconsistent results between run 1 and run {}".format(i + 1)

    # Additional consistency checks
    for run in range(num_runs):
        assert len(all_results[run].scores) == len(
            all_results[0].scores
        ), "Inconsistent number of score types in run {}".format(run + 1)
        for idx, score_0 in enumerate(all_results[0].scores):
            score_type_0 = score_0.type
            score_run = all_results[run].scores[idx]
            score_type_run = score_run.type

            assert score_type_0 == score_type_run, (
                f"Mismatched score types at index {idx} in run {run + 1}."
                + f" Expected {score_type_0}, got {score_type_run}"
            )

            assert torch.allclose(
                score_0.scores, score_run.scores, atol=1e-6
            ), (
                f"Inconsistent scores for {score_type_0} in run {run + 1}. "
                f"Run 0 scores: {score_0.scores}, Run {run + 1} scores: {score_run.scores}"
            )

    logger.info("All runs completed successfully with consistent results!")


@pytest.mark.asyncio
@patch(
    "neurons.validator.config.get_metagraph",
    return_value=mock_meta,
)
@patch(
    "neurons.validator.config.get_backend_client",
    return_value=mock_client,
)
@patch(
    "neurons.validator.forward.get_metagraph",
    return_value=mock_meta,
)
@patch(
    "neurons.validator.scoring.models.base.get_metagraph",
    return_value=mock_meta,
)
@patch(
    "neurons.validator.scoring.pipeline.get_metagraph",
    return_value=mock_meta,
)
@patch(
    "neurons.validator.scoring.models.rewards.human.get_metagraph",
    return_value=mock_meta,
)
@patch(
    "neurons.validator.scoring.models.rewards.human.get_backend_client",
    return_value=mock_client,
)
@patch(
    "neurons.validator.scoring.models.rewards.image_reward.RM.load",
    mock_imagereward_load,
)
async def test_full_pipeline_integration_with_moving_averages(*mocks):
    num_runs = 3
    all_results = []
    all_filtered_rewards = []
    moving_average_scores = torch.zeros(5)  # Assuming 5 miners
    ma_history = [moving_average_scores.clone()]

    mock_backend_client = MockBackendClient()
    with patch(
        "neurons.validator.forward.get_backend_client",
        return_value=mock_backend_client,
    ):
        for run in range(num_runs):
            logger.info("Starting run {} of {}".format(run + 1, num_runs))
            results, filtered_rewards = await run_pipeline_test()
            all_results.append(results)
            all_filtered_rewards.append(filtered_rewards)

            # Update moving averages
            moving_average_scores = await update_moving_averages(
                moving_average_scores, filtered_rewards
            )
            ma_history.append(moving_average_scores.clone())

            logger.info(
                f"Run {run + 1} - Updated moving average scores: {moving_average_scores}"
            )

    assert (
        mock_backend_client.post_moving_averages_called == num_runs
    ), "Backend client should be called once per run"

    # Check consistency across runs
    for i in range(1, num_runs):
        assert torch.allclose(
            all_filtered_rewards[0], all_filtered_rewards[i], atol=1e-6
        ), "Inconsistent filtered rewards between run 1 and run {}".format(
            i + 1
        )

    # Additional consistency checks
    for run in range(num_runs):
        assert len(all_results[run].scores) == len(
            all_results[0].scores
        ), "Inconsistent number of score types in run {}".format(run + 1)

        for idx, score_0 in enumerate(all_results[0].scores):
            score_type_0 = score_0.type
            score_run = all_results[run].scores[idx]
            score_type_run = score_run.type

            assert (
                score_type_0 == score_type_run
            ), f"Mismatched score types at index {idx} in run {run + 1}. Expected {score_type_0}, got {score_type_run}"

            assert (
                score_0.scores is not None
            ), f"Score type {score_type_0} returned None scores in run 0"
            assert (
                score_run.scores is not None
            ), f"Score type {score_type_0} returned None scores in run {run + 1}"

            try:
                assert torch.allclose(
                    score_0.scores, score_run.scores, atol=1e-6
                ), (
                    f"Inconsistent scores for {score_type_0} in run {run + 1}. "
                    f"Run 0 scores: {score_0.scores}, Run {run + 1} scores: {score_run.scores}"
                )
            except AssertionError as e:
                logger.error(f"Assertion failed: {str(e)}")
                logger.info(f"Score type: {score_type_0}")
                logger.info(f"Run 0 scores: {score_0.scores}")
                logger.info(f"Run {run + 1} scores: {score_run.scores}")
                raise

    # Check if final moving average scores are within expected range
    assert torch.all(
        (moving_average_scores >= 0) & (moving_average_scores <= 1)
    ), "Final moving average scores not in range [0, 1]: {}".format(
        moving_average_scores
    )

    # Verify that moving averages have changed over time
    assert not torch.allclose(
        ma_history[0], ma_history[-1], atol=1e-6
    ), "Moving averages should change over multiple runs"

    logger.info("All runs completed successfully with consistent results!")
    logger.info(f"Initial moving average scores: {ma_history[0]}")
    logger.info(f"Final moving average scores: {ma_history[-1]}")
