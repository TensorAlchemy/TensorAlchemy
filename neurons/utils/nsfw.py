import re
import requests

from typing import List

from loguru import logger

from neurons.constants import (
    NSFW_WORDLIST_DEFAULT,
    NSFW_WORDLIST_URL,
)


def load_nsfw_words(url: str) -> List[str]:
    try:
        response = requests.get(url)
        # Raise an exception if the request was unsuccessful
        response.raise_for_status()

        # Split the content into lines and strip whitespace
        words = [line.strip() for line in response.text.splitlines()]

        # Remove empty lines
        words = [word for word in words if word]

        return words

    except requests.exceptions.RequestException as e:
        logger.error(f"Error occurred while loading NSFW words from {url}: {str(e)}")
        return NSFW_WORDLIST_DEFAULT


NSFW_WORDS: List[str] = load_nsfw_words(NSFW_WORDLIST_URL)


def clean_nsfw_from_prompt(prompt):
    for word in NSFW_WORDS:
        if re.search(r"\b{}\b".format(word), prompt):
            prompt = re.sub(r"\b{}\b".format(word), "", prompt).strip()
            logger.warning(f"Removed NSFW word {word.strip()} from prompt...")

    return prompt
