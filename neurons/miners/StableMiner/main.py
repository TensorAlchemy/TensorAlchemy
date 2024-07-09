import os
import sys
import pathlib
import warnings
import traceback

import torch
from diffusers import (
    AutoPipelineForText2Image,
    AutoPipelineForImage2Image,
    DPMSolverMultistepScheduler,
)
from loguru import logger

from neurons.miners.StableMiner.schema import TaskType, TaskConfig

# Suppress the eth_utils network warnings
# "does not have a valid ChainId."
# NOTE: It's not our bug, it's upstream
# TODO: Remove after updating bittensor
warnings.simplefilter("ignore")

# Use the older torch style for now
os.environ["USE_TORCH"] = "1"

if __name__ == "__main__":
    try:
        # Add the base repository to the path so the miner can access it
        file_path: str = str(
            pathlib.Path(__file__).parent.parent.parent.parent.resolve(),
        )
        if file_path not in sys.path:
            sys.path.append(file_path)
        # Import StableMiner after fixing path
        from miner import StableMiner

        task_configs = [
            TaskConfig(
                task_type=TaskType.TEXT_TO_IMAGE,
                pipeline=AutoPipelineForText2Image,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
                scheduler=DPMSolverMultistepScheduler,
            ),
            TaskConfig(
                task_type=TaskType.IMAGE_TO_IMAGE,
                pipeline=AutoPipelineForImage2Image,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
                scheduler=DPMSolverMultistepScheduler,
            ),
        ]
        # Start the miner
        StableMiner(task_configs)
    except ImportError:
        logger.error(f"Error: {traceback.format_exc()}")
        logger.error("Please ensure all required packages are installed.")
        sys.exit(1)
    except Exception:
        logger.error(f"Error: {traceback.format_exc()}")
        sys.exit(1)
