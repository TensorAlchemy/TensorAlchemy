import uuid
from typing import Dict, List

from loguru import logger
from neurons.config import clients
from neurons.protocol import ImageGeneration, ModelType
from neurons.utils.image import image_to_base64
from PIL.Image import Image as PILImage
from scoring.pipeline import get_scoring_results
from scoring.types import ScoringResults

import bittensor as bt


def generate_synapse(prompt: str, image_base64: str) -> bt.Synapse:
    hotkey: str = str(uuid.uuid4())
    try:
        synapse = ImageGeneration(
            seed=-1,
            width=1024,
            height=1024,
            prompt=prompt,
            images=[image_base64],
            generation_type="TEXT_TO_IMAGE",
            model_type=ModelType.CUSTOM.value,
        )
        synapse.axon = bt.TerminalInfo(hotkey=hotkey)
        logger.success("Successfully generated synapse")
        return synapse
    except Exception as e:
        logger.error(f"Error in generating synapse: {str(e)}")
        raise


async def score_images(synapses: List[bt.Synapse]) -> ScoringResults:
    logger.info(f"Scoring images for {len(synapses)} synapses")
    try:
        results = await get_scoring_results(
            # SCORING type will only run the scoring models
            # automatically skipping human validation etc
            #
            # see scoring.models.__init__.py
            ModelType.SCORING,
            synapses[0],
            synapses,
        )
        logger.success("Successfully scored images.")
        return results
    except Exception as e:
        logger.error(f"Error in scoring images: {str(e)}")
        raise


async def score_b64_imageset(
    prompt: str,
    images: List[str],
) -> ScoringResults:
    synapses = []

    for image in images:
        synapses.append(generate_synapse(prompt, image))

    class MetagraphMock:
        """
        We will pass in a mocked metagraph
        Reason: pipeline scores by hotkey for each item
        """

        n = len(synapses)
        hotkeys = [s.axon.hotkey for s in synapses]

    clients.metagraph = MetagraphMock()

    return await score_images(synapses)


async def score_imageset(
    prompt: str,
    images: List[PILImage],
) -> ScoringResults:
    return await score_b64_imageset(
        prompt,
        [image_to_base64(image) for image in images],
    )


async def score_named_imageset(
    prompt: str,
    images: Dict[str, PILImage],
) -> Dict[str, float]:
    # Call the score_imageset function with the list of images
    results: ScoringResults = await score_imageset(
        prompt,
        list(images.values()),
    )

    # Create a new dictionary to store named results
    to_return = {}

    # Pair the results with their original names
    for name, score in zip(images.keys(), results.combined_scores):
        to_return[name] = score

    # Create a new ScoringResults object with named scores
    return to_return
