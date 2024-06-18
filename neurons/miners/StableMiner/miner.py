import torch
from base import BaseMiner
from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForText2Image,
    DPMSolverMultistepScheduler,
)
from neurons.protocol import ModelType
from neurons.safety import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor
from utils import colored_log, warm_up


class StableMiner(BaseMiner):
    def __init__(self):
        super().__init__()

        # Load the model
        self.load_models()

        # Optimize model
        self.optimize_models()

        # Serve the axon
        self.start_axon()

        # Start the miner loop
        self.loop()

    def load_models(self):
        # Load the text-to-image model
        self.t2i_model_custom = AutoPipelineForText2Image.from_pretrained(
            self.config.miner.custom_model,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        ).to(self.config.miner.device)
        self.t2i_model_custom.set_progress_bar_config(disable=True)
        self.t2i_model_custom.scheduler = DPMSolverMultistepScheduler.from_config(
            self.t2i_model_custom.scheduler.config
        )

        self.t2i_model_alchemy = AutoPipelineForText2Image.from_pretrained(
            self.config.miner.alchemy_model,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        ).to(self.config.miner.device)
        self.t2i_model_alchemy.set_progress_bar_config(disable=True)
        self.t2i_model_alchemy.scheduler = DPMSolverMultistepScheduler.from_config(
            self.t2i_model_alchemy.scheduler.config
        )

        # Load the image to image model using the same pipeline (efficient)
        self.i2i_model_custom = AutoPipelineForImage2Image.from_pipe(
            self.t2i_model_custom
        ).to(
            self.config.miner.device,
        )
        self.i2i_model_custom.set_progress_bar_config(disable=True)
        self.i2i_model_custom.scheduler = DPMSolverMultistepScheduler.from_config(
            self.i2i_model_custom.scheduler.config
        )

        self.i2i_model_alchemy = AutoPipelineForImage2Image.from_pipe(
            self.t2i_model_alchemy
        ).to(
            self.config.miner.device,
        )
        self.i2i_model_alchemy.set_progress_bar_config(disable=True)
        self.i2i_model_alchemy.scheduler = DPMSolverMultistepScheduler.from_config(
            self.i2i_model_alchemy.scheduler.config
        )

        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            "CompVis/stable-diffusion-safety-checker"
        ).to(self.config.miner.device)
        self.processor = CLIPImageProcessor()

        # Set up mapping for the different synapse types
        self.mapping = {
            f"text_to_image{ModelType.ALCHEMY.value.lower()}": {
                "args": self.t2i_args,
                "model": self.t2i_model_alchemy,
            },
            f"text_to_image{ModelType.CUSTOM.value.lower()}": {
                "args": self.t2i_args,
                "model": self.t2i_model_custom,
            },
            f"image_to_image{ModelType.ALCHEMY.value.lower()}": {
                "args": self.i2i_args,
                "model": self.i2i_model_alchemy,
            },
            f"image_to_image{ModelType.CUSTOM.value.lower()}": {
                "args": self.i2i_args,
                "model": self.i2i_model_custom,
            },
        }

    def optimize_models(self):
        if self.config.miner.optimize:
            self.t2i_model_alchemy.unet = torch.compile(
                self.t2i_model_alchemy.unet, mode="reduce-overhead", fullgraph=True
            )

            # Warm up model
            colored_log(
                ">>> Warming up model with compile... "
                + "this takes roughly two minutes...",
                color="yellow",
            )
            warm_up(self.t2i_model_alchemy, self.t2i_args)
