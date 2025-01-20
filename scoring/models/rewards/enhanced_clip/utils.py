import json
import traceback
from typing import Awaitable, Callable, Dict, List, TypedDict, Union

from httpx import HTTPStatusError, ReadTimeout
import httpx
from loguru import logger

# Configuration functions
from neurons.config import (
    MissingApiKeyError,
    get_corcel_api_key,
    get_openai_client,
)
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)
from openai.types.shared_params import FunctionDefinition

# Constants
API_TIMEOUT_SECONDS = 10.0


# Type definitions
class ElementDict(TypedDict):
    description: str
    importance: float


class PromptBreakdown(TypedDict):
    elements: List[ElementDict]


BreakdownFunction = Callable[[str], Awaitable[PromptBreakdown]]


def get_prompt_breakdown_function() -> ChatCompletionToolParam:
    return ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="break_down_prompt",
            description="Break down an image prompt into key elements for CLIP",
            parameters={
                "type": "object",
                "properties": {
                    "elements": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "description": {
                                    "type": "string",
                                    "description": "Key element of the image, "
                                    + "Single Word (for CLIP analysis)",
                                },
                            },
                            "required": ["description"],
                        },
                        "description": "Key elements from the prompt",
                    }
                },
                "required": ["elements"],
            },
        ),
    )


def get_query_messages(
    prompt: str,
) -> List[
    Union[ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam]
]:
    return [
        ChatCompletionSystemMessageParam(
            role="system",
            content="Break down image prompts into key elements."
            + " Each element should be concise and evaluatable."
            + " Assign importance based on significance to the image.",
        ),
        ChatCompletionUserMessageParam(
            role="user",
            content=f"Break down this image prompt: {prompt}",
        ),
    ]


async def process_api_response(response_data: Dict) -> PromptBreakdown:
    if "choices" in response_data and response_data["choices"]:
        choice = response_data["choices"][0]

        if "message" in choice and "tool_calls" in choice["message"]:
            tool_call = choice["message"]["tool_calls"][0]

            if isinstance(tool_call, dict) and "function" in tool_call:
                return json.loads(tool_call["function"]["arguments"])

    raise ValueError("Unexpected response structure from API")


async def openai_breakdown(prompt: str) -> PromptBreakdown:
    client: AsyncOpenAI = get_openai_client()
    messages = get_query_messages(prompt)
    tool = get_prompt_breakdown_function()

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        tool_choice={
            "type": "function",
            "function": {"name": tool["function"]["name"]},
        },
        tools=[tool],
        messages=messages,
        timeout=API_TIMEOUT_SECONDS,
    )

    return await process_api_response(response.model_dump())


async def corcel_breakdown(prompt: str) -> PromptBreakdown:
    api_key = get_corcel_api_key(required=True)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    messages = get_query_messages(prompt)
    tool = get_prompt_breakdown_function()

    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": m["role"], "content": m["content"]} for m in messages
        ],
        "temperature": 0,
        "tools": [tool],
        "tool_choice": {
            "type": "function",
            "function": {"name": tool["function"]["name"]},
        },
    }

    async with httpx.AsyncClient(timeout=API_TIMEOUT_SECONDS) as client:
        response = await client.post(
            "https://api.corcel.io/cortext/text",
            headers=headers,
            json=payload,
            timeout=API_TIMEOUT_SECONDS,
        )
        response.raise_for_status()  # This will raise an HTTPStatusError for 4xx/5xx responses
        print(response.content)
        result = response.json()
        return await process_api_response(result)


async def break_down_prompt(
    prompt: str,
) -> PromptBreakdown:
    services: Dict[str, BreakdownFunction] = {
        "corcel": corcel_breakdown,
        "openai": openai_breakdown,
    }

    last_error = None
    for service_name, service_method in services.items():
        try:
            return await service_method(prompt)

        except MissingApiKeyError:
            logger.debug(f"Skipping {service_name} due to missing API key")
            continue

        except HTTPStatusError as e:
            logger.warning(
                f"{service_name} API returned error {e.response.status_code}: {e.response.text}"
            )
            last_error = e
            continue

        except ReadTimeout as e:
            logger.warning(
                f"{service_name} API request timed out after {API_TIMEOUT_SECONDS} seconds"
            )
            last_error = e
            continue

        except Exception as e:

            logger.error(
                f"Unexpected error with {service_name}: "
                + traceback.format_exc()
            )
            last_error = e
            continue

    if isinstance(last_error, MissingApiKeyError):
        raise MissingApiKeyError("All services had missing API keys")
    else:
        raise Exception(f"All services failed. Last error: {str(last_error)}")
