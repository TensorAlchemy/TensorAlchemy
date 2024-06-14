import asyncio
import os

import openai
from loguru import logger
from openai import OpenAI, AsyncOpenAI
from tenacity import retry, wait_fixed, stop_after_attempt, retry_if_exception_type


class OpenAIRequestFailed(Exception):
    pass


class OpenAIService:

    def __init__(self):
        openai_api_key = os.environ.get("OPENAI_API_KEY", None)
        if not openai_api_key:
            raise ValueError("Please set OPENAI_API_KEY")
        self.openai_client = AsyncOpenAI(api_key=openai_api_key)

    @retry(
        wait=wait_fixed(1),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(OpenAIRequestFailed),
        reraise=True,
    )
    async def create_completion_request(self, model: str, prompt: str) -> str:
        """
        Create a completion of prompt
        """
        try:
            response = await self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": prompt,
                    },
                ],
                temperature=1,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
        except Exception as e:
            logger.error(f"openai completion request failed: {e}")
            raise OpenAIRequestFailed(str(e)) from e

        logger.info(f"OpenAI response object: {response}")
        response = response.choices[0].message.content
        if response:
            logger.info(f"Prompt generated with OpenAI: {response}")
        return response

    @retry(
        wait=wait_fixed(1),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(OpenAIRequestFailed),
        reraise=True,
    )
    async def check_prompt_for_nsfw(self, prompt: str) -> bool:
        """Check if prompts contains any NSFW content

        Returns True if prompt contains any nsfw content
        """
        try:
            response = await self.openai_client.moderations.create(input=prompt)
        except Exception as e:
            logger.error(f"[check_prompt_for_nsfw] failed to do openai request: {e}")
            raise OpenAIRequestFailed(str(e)) from e

        # Check if the moderation flagged the prompt as NSFW
        moderation_results = response.results[0]
        nsfw_flagged = moderation_results.flagged
        # Uncomment in case of need to debug categories returned
        # categories = moderation_results.categories
        # logger.info(f"nsfw_flagged={nsfw_flagged}, categories={categories}")

        return nsfw_flagged


openai_service: OpenAIService = None


def get_openai_service() -> OpenAIService:
    global openai_service
    if not openai_service:
        openai_service = OpenAIService()
    return openai_service
