import sys
import torch
import pytest
from dotenv import load_dotenv

from neurons.validator.rewards.reward import get_human_rewards
from neurons.validator.validator import StableValidator

validator: StableValidator = None

pytest.skip(allow_module_level=True)


@pytest.fixture(autouse=True, scope="session")
def setup() -> None:
    load_dotenv()

    sys.argv += [
        "--netuid",
        "25",
        "--subtensor.network",
        "test",
        "--wallet.name",
        "validator",
        "--wallet.hotkey default",
    ]
    validator = StableValidator()
    validator.load_state()

    if validator.moving_average_scores.sum() == 0:
        validator.moving_average_scores = torch.tensor(
            [
                0.0000e00,
                4.5834e-01,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                3.5371e-10,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                1.4045e-11,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                7.6022e-08,
                0.0000e00,
                0.0000e00,
                1.2612e-44,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                5.2420e-01,
                4.5746e-01,
                4.6468e-01,
                4.6027e-01,
                5.1175e-01,
                5.2633e-01,
                5.0278e-01,
                4.9042e-01,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                5.5324e-08,
                0.0000e00,
                6.8319e-08,
                0.0000e00,
                0.0000e00,
                5.4010e-12,
                0.0000e00,
                1.2612e-44,
                0.0000e00,
                7.8911e-08,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                7.3861e-08,
                5.3888e-08,
                0.0000e00,
                0.0000e00,
                7.0852e-08,
                7.7377e-08,
                7.6482e-08,
                7.1486e-08,
                6.4163e-08,
                3.0326e-12,
                8.3539e-08,
                7.2657e-08,
                0.0000e00,
                5.4044e-08,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                1.3815e-08,
                7.1958e-08,
                0.0000e00,
                3.9497e-08,
                2.1916e-07,
                0.0000e00,
                7.7840e-08,
                7.4375e-08,
                5.8878e-08,
                0.0000e00,
                0.0000e00,
                1.9986e-01,
                0.0000e00,
                0.0000e00,
                4.4630e-08,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
            ]
        ).to(validator.device)

    yield

    validator.background_timer.cancel()


def test_human_rewards_score_increase():
    test_uid_index = (
        (
            validator.moving_average_scores
            == torch.min(
                validator.moving_average_scores[validator.moving_average_scores > 0]
            )
        )
        .nonzero()[0]
        .item()
    )
    previous_score = validator.moving_average_scores[test_uid_index]
    scores = get_human_rewards(
        validator,
        validator.moving_average_scores,
        mock=True,
        mock_winner=validator.hotkeys[test_uid_index],
    )
    current_score = scores[[test_uid_index]]
    assert current_score > previous_score


def test_human_weights_increase():
    test_uid_index = (
        (
            validator.moving_average_scores
            == torch.min(
                validator.moving_average_scores[validator.moving_average_scores > 0]
            )
        )
        .nonzero()[0]
        .item()
    )
    weights = torch.nn.functional.normalize(validator.moving_average_scores, p=1, dim=0)
    previous_weight = weights[test_uid_index]
    scores = get_human_rewards(
        validator,
        validator.moving_average_scores,
        mock=True,
        mock_winner=validator.hotkeys[test_uid_index],
    )
    current_weight = torch.nn.functional.normalize(scores, p=1, dim=0)[test_uid_index]
    assert current_weight > previous_weight


def test_human_rewards_score_decrease():
    reference_uid_index = 0
    mock_loser_index = (
        (
            validator.moving_average_scores
            == torch.max(
                validator.moving_average_scores[validator.moving_average_scores > 0]
            )
        )
        .nonzero()[0]
        .item()
    )
    mock_winner_index = (
        (
            validator.moving_average_scores
            == torch.min(
                validator.moving_average_scores[validator.moving_average_scores > 0]
            )
        )
        .nonzero()[0]
        .item()
    )
    previous_score_delta = (
        validator.moving_average_scores[mock_loser_index]
        - validator.moving_average_scores[reference_uid_index]
    )
    for i in range(0, 100):
        scores = validator.moving_average_scores
        scores = get_human_rewards(
            validator,
            scores,
            mock=True,
            mock_winner=validator.hotkeys[mock_winner_index],
            mock_loser=validator.hotkeys[mock_loser_index],
        )
    current_score_delta = scores[mock_loser_index] - scores[reference_uid_index]
    assert current_score_delta < previous_score_delta


def test_human_weights_decrease():
    mock_loser_index = (
        (
            validator.moving_average_scores
            == torch.max(
                validator.moving_average_scores[validator.moving_average_scores > 0]
            )
        )
        .nonzero()[0]
        .item()
    )
    mock_winner_index = (
        (
            validator.moving_average_scores
            == torch.min(
                validator.moving_average_scores[validator.moving_average_scores > 0]
            )
        )
        .nonzero()[0]
        .item()
    )
    previous_weight = torch.nn.functional.normalize(
        validator.moving_average_scores, p=1, dim=0
    )[mock_loser_index]
    scores = get_human_rewards(
        validator,
        validator.moving_average_scores,
        mock=True,
        mock_winner=validator.hotkeys[mock_winner_index],
        mock_loser=validator.hotkeys[mock_loser_index],
    )
    current_weight = torch.nn.functional.normalize(scores, p=1, dim=0)[mock_loser_index]

    assert current_weight < previous_weight
