import asyncio
import random
from typing import Optional

import requests
from loguru import logger
from neurons.config import get_corcel_api_key
from neurons.config.clients import MissingResponseError


def corcel_parse_response(text):
    if not isinstance(text, str):
        logger.warning(f"Input is not a string: {text}")
        return str(text)

    parts = [part.strip() for part in text.split('"') if part.strip()]

    if not parts:
        logger.info(f"No non-empty parts found in: {text}")
        return text

    result = parts[0]
    logger.info(f"Returning parsed text: {result}")
    return result


async def call_corcel(prompt: str) -> Optional[str]:
    HEADERS = {
        "Content-Type": "application/json",
        "Authorization": f"{get_corcel_api_key()}",
    }
    JSON = {
        "miners_to_query": 1,
        "top_k_miners_to_query": 160,
        "ensure_responses": True,
        "miner_uids": [],
        "messages": [
            {
                "role": "system",
                "content": prompt,
            }
        ],
        "model": "cortext-ultra",
        "stream": False,
        "top_p": 1.0,
        "temperature": 1,
        "max_tokens": 250,
        "seed": random.randint(0, 1_000_000),
    }

    logger.info(f"Using args: {JSON}")

    try:
        response = await asyncio.to_thread(
            requests.post,
            "https://api.corcel.io/cortext/text",
            json=JSON,
            headers=HEADERS,
            timeout=15,
        )

        to_return: Optional[str] = response.json()[0]["choices"][0]["message"][
            "content"
        ]

        if not to_return:
            raise MissingResponseError("Corcel")

        logger.info(f"Prompt generated with Corcel: {to_return}")

        return to_return

    except requests.exceptions.ReadTimeout:
        logger.info("Corcel request timed out after 15 seconds...")

    raise MissingResponseError("Corcel")
