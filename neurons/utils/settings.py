import asyncio
import json

import httpx
from loguru import logger


async def download_validator_settings():
    url = "https://raw.githubusercontent.com/TensorAlchemy/validator-settings/main/settings.json"
    try:
        # Send a GET request to the URL using httpx asynchronously
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
        # Check if the request was successful
        response.raise_for_status()
        # Parse the JSON content
        data = response.json()
        # Print the downloaded JSON data
        logger.success(f"Downloaded JSON data: {json.dumps(data, indent=2)}")
        return data
    except httpx.RequestError as e:
        logger.error(f"An error occurred while downloading the JSON: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"An error occurred while parsing the JSON: {e}")
        return None
    except Exception as e:
        logger.error(f"An unknown error occurred: {e}")
        return None


# Example of how to run the async function
async def main():
    result = await download_validator_settings()
    if result:
        logger.success("Successfully downloaded and parsed the settings.")
    else:
        logger.error("Failed to download or parse the settings.")


# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())
