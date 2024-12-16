import asyncio
import json
import traceback
from typing import Dict

from google.cloud import storage
from loguru import logger
from neurons.config import get_config
from neurons.constants import IA_BUCKET_NAME, IA_TEST_BUCKET_NAME


def get_bucket_name() -> str:
    if get_config().netuid == "test":
        return IA_TEST_BUCKET_NAME

    return IA_BUCKET_NAME


def get_storage_client() -> storage.Client:
    """
    Create and return an anonymous storage client.

    Returns:
        storage.Client: An anonymous storage client.
    """
    return storage.Client.create_anonymous_client()


async def retrieve_public_file(
    source_name: str,
    bucket_name: str = get_bucket_name(),
    client: storage.Client = get_storage_client(),
) -> Dict:
    downloaded: Dict = None
    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_name)

        try:
            # Use asyncio.to_thread to run the
            # synchronous download_as_text method in a separate thread
            content = await asyncio.to_thread(blob.download_as_text)
            downloaded = json.loads(content)
            logger.info(
                f"Successfully downloaded {source_name} from {bucket_name}"
            )
        except Exception as e:
            logger.info(
                f"Failed to download {source_name} from {bucket_name}: {e}"
            )
    except Exception:
        logger.info(
            "An error occurred downloading from Google Cloud: "
            + traceback.format_exc()
        )

    return downloaded
