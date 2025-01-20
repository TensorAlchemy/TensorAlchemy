from typing import Dict, List

from loguru import logger
from neurons.config import MissingApiKeyError
from neurons.validator.services.openai.service import get_openai_service
from neurons.validator.utils.corcel import call_corcel


class PromptElement:
    def __init__(self, description: str):
        self.description = description


def get_breakdown_prompt(prompt: str) -> str:
    return (
        "Break down the following image prompt into key elements. "
        "Each element should be a single word that is concise and evaluatable.\n\n"
        f"Prompt: {prompt}\n\n"
        "Return elements one per line."
    )


def parse_elements(response: str) -> List[PromptElement]:
    """Parse response text into list of elements, skipping empty lines"""
    return [
        PromptElement(line.strip()) 
        for line in response.split("\n") 
        if line.strip()
    ]

async def openai_breakdown(prompt: str) -> List[PromptElement]:
    service = get_openai_service()
    response = await service.create_completion_request(
        model="gpt-4o-mini",
        prompt=get_breakdown_prompt(prompt)
    )
    
    if not response:
        raise ValueError("No response received from OpenAI")

    return parse_elements(response)

async def corcel_breakdown(prompt: str) -> List[PromptElement]:
    response = call_corcel(get_breakdown_prompt(prompt))
    
    if not response:
        raise ValueError("No response received from Corcel")

    # Convert response to string if it's a Response object
    response_text = str(response)
        
    return parse_elements(response_text)


async def break_down_prompt(prompt: str) -> List[PromptElement]:
    """Try to break down prompt using available services"""
    services = [
        ("corcel", corcel_breakdown),
        ("openai", openai_breakdown)
    ]

    for name, service in services:
        try:
            return await service(prompt)
        except MissingApiKeyError:
            logger.debug(f"Skipping {name} due to missing API key")
        except Exception as e:
            logger.error(f"Error with {name}: {str(e)}")
            
    raise Exception("All services failed to break down prompt")
