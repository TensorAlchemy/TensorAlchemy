from datetime import datetime

from loguru import logger
from neurons.miners.Inpainter.utils import (
    get_caller_stake,
    get_coldkey_for_hotkey,
)
from neurons.utils.log import sh


def do_logs(self, synapse, local_args):
    """
    Output logs for each request that comes through.
    """
    time_elapsed = datetime.now() - self.stats.start_time
    hotkey = synapse.axon.hotkey

    logger.info(
        str(sh("Info"))
        + f" -> Date {datetime.strftime(self.stats.start_time, '%Y/%m/%d %H:%M')}"
        + f" | Elapsed {time_elapsed}"
        + f" | RPM {self.stats.total_requests/(time_elapsed.total_seconds()/60):.2f}"
        + f" | Model {self.config.miner.model}"
        + f" | Default seed {self.config.miner.seed}.",
    )
    logger.info(
        str(sh("Request"))
        + f" -> Type: {synapse.generation_type}"
        + f" | Request seed: {synapse.seed}"
        + f" | Total requests {self.stats.total_requests:,}"
        + f" | Timeouts {self.stats.timeouts:,}.",
    )

    args_list = [
        f"{k.capitalize()}: {f'{v:.2f}' if isinstance(v, float) else v}"
        for k, v in local_args.items()
    ]
    logger.info(f"{sh('Args')} -> {' | '.join(args_list)}.", color="magenta")

    miner_info = self.get_miner_info()
    logger.info(
        str(sh("Stats"))
        + f" -> Block: {miner_info['block']}"
        + f" | Stake: {miner_info['stake']:.4f}"
        + f" | Incentive: {miner_info['incentive']:.4f}"
        + f" | Trust: {miner_info['trust']:.4f}"
        + f" | Consensus: {miner_info['consensus']:.4f}.",
    )

    # Output stake
    requester_stake = get_caller_stake(synapse)
    if requester_stake is None:
        requester_stake = -1

    # Retrieve the coldkey of the caller
    caller_coldkey = get_coldkey_for_hotkey(hotkey)

    temp_string = f"Stake {int(requester_stake):,}"

    has_hotkey: bool = hotkey in self.hotkey_whitelist
    has_coldkey: bool = caller_coldkey in self.coldkey_whitelist

    if has_hotkey or has_coldkey:
        temp_string = "Whitelisted key"

    logger.info(
        #
        str(sh("Caller"))
        + f" -> {temp_string}"
        + f" | Hotkey {hotkey}.",
    )
