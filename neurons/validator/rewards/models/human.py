from typing import Dict, List
import bittensor as bt
from loguru import logger

from neurons.validator.rewards.models.base import BaseRewardModel
from neurons.validator.rewards.types import RewardModelType
from neurons.validator.config import get_backend_client, get_metagraph


class HumanValidationRewardModel(BaseRewardModel):
    @property
    def name(self) -> str:
        return str(RewardModelType.HUMAN)

    async def get_rewards(
        self,
        _synapse: bt.Synapse,
        responses: List[bt.Synapse],
    ) -> Dict[int, float]:
        logger.info("Extracting human votes...")

        human_voting_scores_dict = {}
        metagraph = get_metagraph()

        try:
            self.human_voting_scores = await get_backend_client().get_votes()
        except Exception as e:
            logger.error(f"Error while getting votes: {e}")
            return {response.dendrite.hotkey: 0.0 for response in responses}

        if self.human_voting_scores:
            for inner_dict in self.human_voting_scores.values():
                for hotkey, value in inner_dict.items():
                    if hotkey in human_voting_scores_dict:
                        human_voting_scores_dict[hotkey] += value
                    else:
                        human_voting_scores_dict[hotkey] = value

        rewards = {}
        for response in responses:
            hotkey = response.dendrite.hotkey
            try:
                uid = metagraph.hotkeys.index(hotkey)
                rewards[uid] = human_voting_scores_dict.get(hotkey, 0.0)
            except ValueError:
                logger.warning(
                    f"Hotkey {hotkey} not found in metagraph. Assigning 0 reward."
                )
                rewards[hotkey] = 0.0

        return rewards

    def normalize_rewards(self, rewards: Dict[int, float]) -> Dict[int, float]:
        if not rewards:
            return rewards

        total = sum(rewards.values())
        if total == 0:
            return rewards

        return {uid: score / total for uid, score in rewards.items()}
