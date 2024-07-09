from typing import Optional

import torch
from torch import Type

from neurons.miners.StableMiner.schema import TaskConfig


class ModelLoader:
    def __init__(self, config):
        self.config = config

    def load(self, model_name: str, task_config: TaskConfig) -> torch.nn.Module:
        pipeline_class = task_config.pipeline
        model = pipeline_class.from_pretrained(
            model_name,
            torch_dtype=task_config.torch_dtype,
            use_safetensors=task_config.use_safetensors,
            variant=task_config.variant,
        )

        model.to(self.config.device)
        model.set_progress_bar_config(disable=True)

        if task_config.scheduler and hasattr(model, "scheduler"):
            model.scheduler = task_config.scheduler.from_config(model.scheduler.config)

        return model

    def load_safety_checker(
        self, safety_checker_class: Type
    ) -> Optional[torch.nn.Module]:
        if safety_checker_class:
            safety_checker = safety_checker_class.from_pretrained(
                "CompVis/stable-diffusion-safety-checker"
            ).to(self.config.device)
            return safety_checker
        return None

    def load_processor(self, processor_class: Type) -> Optional[torch.nn.Module]:
        if processor_class:
            processor = processor_class()
            return processor
        return None
