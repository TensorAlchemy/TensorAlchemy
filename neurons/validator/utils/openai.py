from typing import Optional
from loguru import logger
from neurons.config import get_openai_client
from neurons.config.clients import MissingResponseError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)


class OpenAIRequestFailed(Exception):
    pass


@retry(
    wait=wait_fixed(1),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(OpenAIRequestFailed),
    reraise=True,
)
async def create_completion_request(model: str, prompt: str) -> str:
    """
    Create a completion of prompt.

    Returns the completion text or raises MissingResponseError if no valid completion
    """
    try:
        response = await get_openai_client().chat.completions.create(
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
        logger.error(e)
        raise OpenAIRequestFailed(str(e)) from e

    logger.info(f"OpenAI response object: {response}")
    if len(response.choices) < 1:
        raise MissingResponseError("OpenAI")

    to_return: Optional[str] = response.choices[0].message.content

    if not to_return:
        raise MissingResponseError("OpenAI")

    logger.info(f"Prompt generated with OpenAI: {to_return}")
    return to_return


@retry(
    wait=wait_fixed(1),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(OpenAIRequestFailed),
    reraise=True,
)
async def check_prompt_for_nsfw(prompt: str) -> bool:
    """Check if prompts contains any NSFW content

    Returns True if prompt contains any nsfw content
    """
    try:
        response = await get_openai_client().moderations.create(input=prompt)
    except Exception as e:
        logger.error(
            f"[check_prompt_for_nsfw] failed to do openai request: {e}"
        )
        raise OpenAIRequestFailed(str(e)) from e

    # Check if the moderation flagged the prompt as NSFW
    if len(response.results) == 0:
        raise OpenAIRequestFailed("moderation results are empty...")
    moderation_results = response.results[0]
    nsfw_flagged = moderation_results.flagged

    # Uncomment in case of need to debug categories returned
    # categories = moderation_results.categories
    # logger.info(f"nsfw_flagged={nsfw_flagged}, categories={categories}")

    return nsfw_flagged
