import traceback
from typing import List, Optional, TypedDict

from loguru import logger

from neurons.config import MissingApiKeyError
from neurons.config.clients import MissingResponseError
from neurons.utils.validator.corcel import call_corcel
from neurons.utils.validator.openai import create_completion_request


class ElementDict(TypedDict):
    description: str
    importance: float


class PromptBreakdown(TypedDict):
    elements: List[ElementDict]


def get_breakdown_prompt(prompt: str) -> str:
    return (
        "Break down the following image prompt into key elements. "
        + "Each element should be a single word or short phrase that is concise and evaluatable.\n\n"
        + f"Prompt: {prompt}\n\n"
        + "Return elements as a comma-separated list."
    )


def parse_elements(response: str) -> PromptBreakdown:
    """Parse response text into PromptBreakdown format"""
    # First split by newlines and join to handle any inconsistent formatting
    cleaned_response = " ".join(response.split())

    # Split by commas and clean up each element
    elements = [
        ElementDict(description=element.strip(), importance=1.0)
        for element in cleaned_response.split(",")
        if element.strip()
    ]

    if not elements:
        raise ValueError("No elements found in response")

    return PromptBreakdown(elements=elements)


async def openai_breakdown(prompt: str) -> PromptBreakdown:
    response = await create_completion_request(
        model="gpt-4o-mini", prompt=get_breakdown_prompt(prompt)
    )

    if not response:
        raise ValueError("No response received from OpenAI")

    return parse_elements(response)


async def corcel_breakdown(prompt: str) -> PromptBreakdown:
    response = await call_corcel(get_breakdown_prompt(prompt))

    if not response:
        raise ValueError("No response received from Corcel")

    # Convert response to string if it's a Response object
    response_text = str(response)

    return parse_elements(response_text)


async def break_down_prompt(prompt: str) -> PromptBreakdown:
    """Try to break down prompt using available services"""
    services = [
        ("openai", openai_breakdown),
        ("corcel", corcel_breakdown),
    ]

    for name, service in services:
        try:
            return await service(prompt)
        except MissingApiKeyError:
            logger.debug(f"Skipping {name} due to missing API key")
        except MissingResponseError:
            logger.debug(f"Skipping {name} due to no response")
        except Exception:
            logger.error(f"Error with {name}: " + traceback.format_exc())
            continue

    raise MissingApiKeyError(
        "No service available "
        + "- all services failed due to missing API keys"
    )
