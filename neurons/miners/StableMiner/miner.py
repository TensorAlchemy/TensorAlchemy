from typing import Dict, Optional, List
import torch
from neurons.miners.StableMiner.base import BaseMiner
from loguru import logger

from neurons.miners.StableMiner.model_loader import ModelLoader
from neurons.miners.StableMiner.schema import ModelConfig, TaskType, TaskConfig
from neurons.protocol import ModelType
from neurons.miners.StableMiner.utils import warm_up


class StableMiner(BaseMiner):
    def __init__(self, task_configs: List[TaskConfig]) -> None:
        self.task_configs = {config.task_type: config for config in task_configs}
        self.model_configs: Dict[ModelType, Dict[TaskType, ModelConfig]] = {}
        self.safety_checker: Optional[torch.nn.Module] = None
        self.processor: Optional[torch.nn.Module] = None

        super().__init__()

        try:
            logger.info("Initializing StableMiner...")

            self.load_models()

            self.optimize_models()

            self.start_axon()

            self.loop()
        except Exception as e:
            logger.error(f"Error in StableMiner initialization: {e}")
            raise

    def load_models(self) -> None:
        try:
            for task_type, config in self.task_configs.items():
                logger.info(f"Loading models for task: {task_type}...")

                if config.safety_checker:
                    self.safety_checker = ModelLoader(
                        self.config.miner
                    ).load_safety_checker(config.safety_checker)
                    logger.info(f"Safety checker loaded for task: {task_type}")

                if config.processor:
                    self.processor = ModelLoader(self.config.miner).load_processor(
                        config.processor
                    )
                    logger.info(f"Processor loaded for task: {task_type}")

                logger.info(f"Setting up model configurations for task: {task_type}...")
                self.setup_model_configs()

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

    def get_model_config(
        self,
        model_type: ModelType,
        task_type: TaskType,
    ) -> ModelConfig:
        try:
            if model_type not in self.model_configs:
                raise ValueError(f"{model_type} was not found in model_configs!")

            if task_type not in self.model_configs[model_type]:
                raise ValueError(
                    f"{task_type} was not found in model_configs {model_type}!"
                )

            return self.model_configs[model_type][task_type]
        except ValueError as e:
            logger.error(e)
            raise

    def load_model(self, model_name: str, task_type: TaskType) -> torch.nn.Module:
        try:
            logger.info(f"Loading model {model_name} for task {task_type}...")
            config = self.task_configs[task_type]
            model_loader = ModelLoader(self.config.miner)
            model = model_loader.load(model_name, config)

            logger.info(f"Model {model_name} loaded successfully.")
            return model
        except Exception as e:
            logger.error(f"Error loading {task_type.value} model: {e}")
            raise

    def setup_model_configs(self) -> None:
        logger.info("Setting up model configurations...")
        self.model_configs = {
            ModelType.CUSTOM: {
                TaskType.TEXT_TO_IMAGE: ModelConfig(
                    args=self.t2i_args,
                    model=self.load_model(
                        self.config.miner.custom_model, TaskType.TEXT_TO_IMAGE
                    ),
                ),
                # TaskType.IMAGE_TO_IMAGE: ModelConfig(
                #     args=self.i2i_args,
                #     model=self.load_model(self.config.miner.custom_model, TaskType.IMAGE_TO_IMAGE),
                # ),
            },
            # ModelType.ALCHEMY: {
            #     TaskType.TEXT_TO_IMAGE: ModelConfig(
            #         args=self.t2i_args,
            #         model=self.load_model(self.config.miner.alchemy_model, TaskType.TEXT_TO_IMAGE),
            #     ),
            #     TaskType.IMAGE_TO_IMAGE: ModelConfig(
            #         args=self.i2i_args,
            #         model=self.load_model(self.config.miner.alchemy_model, TaskType.IMAGE_TO_IMAGE),
            #     ),
            # },
        }
        logger.info("Model configurations set up successfully.")

    def optimize_models(self) -> None:
        logger.info("Optimizing models...")
        if self.config.miner.optimize:
            try:
                for model_type, tasks in self.model_configs.items():
                    for task_type, config in tasks.items():
                        if config.model:
                            config.model.unet = torch.compile(
                                config.model.unet,
                                mode="reduce-overhead",
                                fullgraph=True,
                            )

                            # Warm up model
                            logger.info(
                                f">>> Warming up {model_type} {task_type} model with compile... "
                                "this takes roughly two minutes...",
                                color="yellow",
                            )
                            warm_up(config.model, config.args)
                logger.info("Models optimized successfully.")
            except Exception as e:
                logger.error(f"Error optimizing models: {e}")
                raise
        else:
            logger.info("Model optimization is disabled.")
