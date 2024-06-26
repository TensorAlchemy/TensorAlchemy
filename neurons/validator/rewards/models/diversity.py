import argparse
import copy
import random
import time
from typing import Dict, List

import bittensor as bt
import sentry_sdk
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms as transforms
from datasets import Dataset
from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForText2Image,
    DPMSolverMultistepScheduler,
)
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    AutoFeatureExtractor,
    AutoImageProcessor,
    AutoModel,
    CLIPImageProcessor,
)

from neurons.validator.config import get_device
from neurons.miners.StableMiner.utils import (
    colored_log,
    nsfw_image_filter,
    sh,
    warm_up,
)
from neurons.protocol import ImageGeneration
from neurons.safety import StableDiffusionSafetyChecker
from neurons.utils import clean_nsfw_from_prompt, get_defaults
from neurons.validator.rewards.models.base import BaseRewardModel
from neurons.validator.rewards.types import RewardModelType


class DiversityRewardModel(BaseRewardModel):
    @property
    def name(self) -> str:
        return str(RewardModelType.DIVERSITY)

    def __init__(self):
        super().__init__()
        self.model_ckpt = "nateraw/vit-base-beans"
        self.extractor = AutoFeatureExtractor.from_pretrained(self.model_ckpt)
        self.processor = AutoImageProcessor.from_pretrained(self.model_ckpt)
        self.model = AutoModel.from_pretrained(self.model_ckpt)
        self.hidden_dim = self.model.config.hidden_size
        self.transformation_chain = T.Compose(
            [
                # We first resize the input image to 256x256 and then we take center crop.
                T.Resize(int((256 / 224) * self.extractor.size["height"])),
                T.CenterCrop(self.extractor.size["height"]),
                T.ToTensor(),
                T.Normalize(
                    mean=self.extractor.image_mean, std=self.extractor.image_std
                ),
            ]
        )

    def extract_embeddings(self, model: torch.nn.Module):
        """Utility to compute embeddings."""

        device = model.device

        def pp(batch):
            images = batch["image"]
            # `transformation_chain` is a compostion of preprocessing
            # transformations we apply to the input images to prepare them
            # for the model. For more details, check out the accompanying Colab Notebook.
            image_batch_transformed = torch.stack(
                [self.transformation_chain(image) for image in images]
            )
            new_batch = {"pixel_values": image_batch_transformed.to(device)}
            with torch.no_grad():
                embeddings = model(**new_batch).last_hidden_state[:, 0].cpu()
            return {"embeddings": embeddings}

        return pp

    async def get_rewards(
        self,
        _synapse: bt.Synapse,
        responses: torch.FloatTensor,
        rewards: torch.FloatTensor,
    ) -> torch.FloatTensor:
        extract_fn = self.extract_embeddings(self.model.to(get_device()))

        images = [
            T.transforms.ToPILImage()(bt.Tensor.deserialize(response.images[0]))
            for response, reward in zip(responses, rewards)
            if reward != 0.0
        ]
        ignored_indices = [
            index for index, reward in enumerate(rewards) if reward == 0.0
        ]
        if len(images) > 1:
            ds = Dataset.from_dict({"image": images})
            embeddings = ds.map(extract_fn, batched=True, batch_size=24)
            embeddings = embeddings["embeddings"]
            simmilarity_matrix = cosine_similarity(embeddings)

            dissimilarity_scores = torch.zeros(len(responses), dtype=torch.float32)
            for i in range(0, len(simmilarity_matrix)):
                for j in range(0, len(simmilarity_matrix)):
                    if i == j:
                        simmilarity_matrix[i][j] = 0
                dissimilarity_scores[i] = 1 - max(simmilarity_matrix[i])
        else:
            dissimilarity_scores = torch.tensor([1.0])

        if ignored_indices and (len(images) > 1):
            i = 0
            while i < len(rewards):
                if i in ignored_indices:
                    dissimilarity_scores = torch.cat(
                        [
                            dissimilarity_scores[:i],
                            torch.tensor([0]),
                            dissimilarity_scores[i:],
                        ]
                    )
                i += 1

        return dissimilarity_scores

    def normalize_rewards(self, rewards: torch.FloatTensor) -> torch.FloatTensor:
        return rewards / rewards.sum()


class ModelDiversityRewardModel(BaseRewardModel):
    @property
    def name(self) -> str:
        return RewardModelType.DIVERSITY

    def get_config(self) -> bt.config:
        argp = argparse.ArgumentParser(description="Miner Configs")

        # Add any args from the parent class
        argp.add_argument("--netuid", type=int, default=1)
        argp.add_argument("--wandb.project", type=str, default="")
        argp.add_argument("--wandb.entity", type=str, default="")
        argp.add_argument("--wandb.api_key", type=str, default="")
        argp.add_argument("--miner.optimize", action="store_true")
        argp.add_argument("--miner.device", type=str, default=get_device())

        seed = random.randint(0, 100_000_000_000)
        argp.add_argument("--miner.seed", type=int, default=seed)

        argp.add_argument(
            "--miner.custom_model",
            type=str,
            default="stabilityai/stable-diffusion-xl-base-1.0",
        )

        argp.add_argument(
            "--miner.alchemy_model",
            type=str,
            default="stabilityai/stable-diffusion-xl-base-1.0",
        )

        bt.subtensor.add_args(argp)
        bt.logging.add_args(argp)
        bt.wallet.add_args(argp)
        bt.axon.add_args(argp)

        config = bt.config(argp)

        return config

    def __init__(self):
        super().__init__()
        self.model_ckpt = "nateraw/vit-base-beans"
        self.extractor = AutoFeatureExtractor.from_pretrained(self.model_ckpt)
        self.processor = AutoImageProcessor.from_pretrained(self.model_ckpt)
        self.model = AutoModel.from_pretrained(self.model_ckpt)
        self.hidden_dim = self.model.config.hidden_size
        self.transformation_chain = T.Compose(
            [
                # We first resize the input image to 256x256 and then we take center crop.
                T.Resize(int((256 / 224) * self.extractor.size["height"])),
                T.CenterCrop(self.extractor.size["height"]),
                T.ToTensor(),
                T.Normalize(
                    mean=self.extractor.image_mean, std=self.extractor.image_std
                ),
            ]
        )
        # Set up transform functionc
        self.transform = transforms.Compose([transforms.PILToTensor()])
        self.threshold = 0.95
        self.config = self.get_config()
        self.stats = get_defaults(self)

        # Load Defaut Arguments
        self.t2i_args, self.i2i_args = self.get_args()

        # Load the model
        self.load_models()

        # Optimize model
        self.optimize_models()

    def get_args(self) -> Dict:
        return {
            "guidance_scale": 7.5,
            "num_inference_steps": 50,
        }, {"guidance_scale": 5, "strength": 0.6}

    def load_models(self):
        # Load the text-to-image model
        self.t2i_model = AutoPipelineForText2Image.from_pretrained(
            self.config.miner.alchemy_model,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        ).to(self.config.miner.device)
        self.t2i_model.set_progress_bar_config(disable=True)
        self.t2i_model.scheduler = DPMSolverMultistepScheduler.from_config(
            self.t2i_model.scheduler.config
        )

        # Load the image to image model using the same pipeline (efficient)
        self.i2i_model = AutoPipelineForImage2Image.from_pipe(self.t2i_model).to(
            self.config.miner.device,
        )
        self.i2i_model.set_progress_bar_config(disable=True)
        self.i2i_model.scheduler = DPMSolverMultistepScheduler.from_config(
            self.i2i_model.scheduler.config
        )

        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            "CompVis/stable-diffusion-safety-checker"
        ).to(self.config.miner.device)
        self.processor = CLIPImageProcessor()

        # Set up mapping for the different synapse types
        self.mapping = {
            "text_to_image": {
                "args": self.t2i_args,
                "model": self.t2i_model,
            },
            "image_to_image": {
                "args": self.i2i_args,
                "model": self.i2i_model,
            },
        }

    def optimize_models(self):
        if self.config.miner.optimize:
            self.t2i_model.unet = torch.compile(
                self.t2i_model.unet, mode="reduce-overhead", fullgraph=True
            )

            # Warm up model
            colored_log(
                ">>> Warming up model with compile... this takes roughly two minutes...",
                color="yellow",
            )
            warm_up(self.t2i_model, self.t2i_args)

    def extract_embeddings(self, model: torch.nn.Module):
        """Utility to compute embeddings."""
        device = model.device

        def pp(batch):
            images = batch["image"]
            # `transformation_chain` is a compostion of preprocessing
            # transformations we apply to the input images to prepare them
            # for the model. For more details, check out the accompanying Colab Notebook.
            image_batch_transformed = torch.stack(
                [self.transformation_chain(image) for image in images]
            )
            new_batch = {"pixel_values": image_batch_transformed.to(device)}
            with torch.no_grad():
                embeddings = model(**new_batch).last_hidden_state[:, 0].cpu()
            return {"embeddings": embeddings}

        return pp

    def generate_image(self, synapse: ImageGeneration) -> ImageGeneration:
        """
        Image generation logic shared between both text-to-image and image-to-image
        """

        # Misc
        timeout = synapse.timeout
        self.stats.total_requests += 1
        start_time = time.perf_counter()

        # Set up args
        local_args = copy.deepcopy(self.mapping[f"{synapse.generation_type}"]["args"])
        local_args["prompt"] = [clean_nsfw_from_prompt(synapse.prompt)]
        local_args["width"] = synapse.width
        local_args["height"] = synapse.height
        local_args["num_images_per_prompt"] = synapse.num_images_per_prompt
        try:
            local_args["guidance_scale"] = synapse.guidance_scale

            if synapse.negative_prompt:
                local_args["negative_prompt"] = [synapse.negative_prompt]
        except Exception:
            logger.error(
                "Values for guidance_scale or negative_prompt were not provided."
            )

        try:
            local_args["num_inference_steps"] = synapse.steps
        except Exception:
            logger.error("Values for steps were not provided.")

        # Get the model
        model = self.mapping[f"{synapse.generation_type}"]["model"]
        if synapse.generation_type == "IMAGE_TO_IMAGE":
            local_args["image"] = T.transforms.ToPILImage()(
                bt.Tensor.deserialize(synapse.prompt_image)
            )
        # Generate images & serialize
        for attempt in range(3):
            try:
                seed = synapse.seed
                local_args["generator"] = [
                    torch.Generator(device=self.config.miner.device).manual_seed(seed)
                ]
                images = model(**local_args).images

                synapse.images = [
                    bt.Tensor.serialize(self.transform(image)) for image in images
                ]
                colored_log(
                    f"{sh('Generating')} -> Succesful image generation after {attempt+1} attempt(s).",
                    color="cyan",
                )
                break
            except Exception as e:
                logger.error(
                    f"Error in attempt number {attempt+1} to generate an image: {e}... sleeping for 5 seconds..."
                )
                time.sleep(5)
                if attempt == 2:
                    images = []
                    synapse.images = []
                    logger.error(
                        f"Failed to generate any images after {attempt+1} attempts."
                    )
                    sentry_sdk.capture_exception(e)

        # Count timeouts
        if time.perf_counter() - start_time > timeout:
            self.stats.timeouts += 1

        # Log NSFW images
        if any(nsfw_image_filter(self, images)):
            logger.error(f"An image was flagged as NSFW: discarding image.")
            self.stats.nsfw_count += 1
            synapse.images = []

        # Log time to generate image
        generation_time = time.perf_counter() - start_time
        self.stats.generation_time += generation_time
        return synapse

    async def get_rewards(
        self,
        synapse: bt.Synapse,
        responses: List[bt.Synapse],
    ) -> Dict[int, float]:
        extract_fn = self.extract_embeddings(self.model.to(get_device()))

        images = [
            T.ToPILImage()(bt.Tensor.deserialize(response.images[0]))
            if response.images
            else None
            for response in responses
        ]

        validator_synapse = self.generate_image(synapse)
        validator_embeddings = extract_fn(
            {
                "image": [
                    T.ToPILImage()(bt.Tensor.deserialize(validator_synapse.images[0]))
                ]
            }
        )

        scores = {}
        for response, image in zip(responses, images):
            if image is None:
                scores[response.dendrite.hotkey] = 0
                continue

            image_embeddings = extract_fn({"image": [image]})
            cosine_similar_score = F.cosine_similarity(
                validator_embeddings["embeddings"],
                image_embeddings["embeddings"],
            )
            scores[response.dendrite.hotkey] = float(
                cosine_similar_score.item() > self.threshold
            )

        return scores

    def normalize_rewards(self, rewards: Dict[int, float]) -> Dict[int, float]:
        return rewards
