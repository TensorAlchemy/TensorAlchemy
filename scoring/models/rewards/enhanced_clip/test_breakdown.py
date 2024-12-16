from unittest.mock import AsyncMock, patch

import pytest
from scoring.models.rewards.enhanced_clip.utils import (
    MissingApiKeyError,
    break_down_prompt,
    get_prompt_breakdown_function,
    get_query_messages,
    process_api_response,
)


@pytest.fixture
def sample_prompt():
    return (
        "A serene lake surrounded by tall pine trees under a starry night sky"
    )


def test_get_prompt_breakdown_function():
    function = get_prompt_breakdown_function()
    assert function["type"] == "function"
    assert function["function"]["name"] == "break_down_prompt"
    assert "parameters" in function["function"]


def test_get_query_messages(sample_prompt):
    messages = get_query_messages(sample_prompt)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert sample_prompt in messages[1]["content"]


@pytest.mark.asyncio
async def test_process_api_response():
    mock_response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "arguments": '{"elements": [{"description": "lake"}, {"description": "trees"}]}'
                            }
                        }
                    ]
                }
            }
        ]
    }
    result = await process_api_response(mock_response)
    assert "elements" in result
    assert len(result["elements"]) == 2
    assert result["elements"][0]["description"] == "lake"


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
