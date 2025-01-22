from unittest.mock import AsyncMock, patch

import pytest

from scoring.models.rewards.enhanced_clip.utils import (
    MissingApiKeyError,
    break_down_prompt,
)


@pytest.fixture
def sample_prompt():
    return (
        "A serene lake surrounded by tall pine trees under a starry night sky"
    )


@pytest.mark.asyncio
async def test_break_down_prompt_all_services_fail(sample_prompt):
    with patch(
        "scoring.models.rewards.enhanced_clip.utils.corcel_breakdown",
        side_effect=MissingApiKeyError,
    ), patch(
        "scoring.models.rewards.enhanced_clip.utils.openai_breakdown",
        side_effect=MissingApiKeyError,
    ):
        with pytest.raises(MissingApiKeyError):
            await break_down_prompt(sample_prompt)


@pytest.mark.asyncio
async def test_break_down_prompt_success(sample_prompt):
    mock_result = {
        "elements": [{"description": "lake"}, {"description": "trees"}]
    }
    with patch(
        "scoring.models.rewards.enhanced_clip.utils.corcel_breakdown",
        AsyncMock(return_value=mock_result),
    ):
        result = await break_down_prompt(sample_prompt)
        assert result == mock_result


if __name__ == "__main__":
    pytest.main()
