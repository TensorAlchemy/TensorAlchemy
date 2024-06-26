import copy
import time
from datetime import datetime

from loguru import logger
from neurons.utils.log import colored_log, sh


# Wrapper for the raw images
class Images:
    def __init__(self, images):
        self.images = images


def get_caller_stake(self, synapse):
    """
    Look up the stake of the requesting validator.
    """
    if synapse.dendrite.hotkey in self.metagraph.hotkeys:
        index = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        return self.metagraph.S[index].item()
    return None


def get_coldkey_for_hotkey(self, hotkey):
    """
    Look up the coldkey of the caller.
    """
    if hotkey in self.metagraph.hotkeys:
        index = self.metagraph.hotkeys.index(hotkey)
        return self.metagraph.coldkeys[index]
    return None


def do_logs(self, synapse, local_args):
    """
    Output logs for each request that comes through.
    """
    time_elapsed = datetime.now() - self.stats.start_time
    hotkey = synapse.dendrite.hotkey

    colored_log(
        str(sh("Info"))
        + f" -> Date {datetime.strftime(self.stats.start_time, '%Y/%m/%d %H:%M')}"
        + f" | Elapsed {time_elapsed}"
        + f" | RPM {self.stats.total_requests/(time_elapsed.total_seconds()/60):.2f}"
        + f" | Model {self.config.miner.model}"
        + f" | Default seed {self.config.miner.seed}.",
        color="green",
    )
    colored_log(
        str(sh("Request"))
        + f" -> Type: {synapse.generation_type}"
        + f" | Request seed: {synapse.seed}"
        + f" | Total requests {self.stats.total_requests:,}"
        + f" | Timeouts {self.stats.timeouts:,}.",
        color="yellow",
    )

    args_list = [
        f"{k.capitalize()}: {f'{v:.2f}' if isinstance(v, float) else v}"
        for k, v in local_args.items()
    ]
    colored_log(f"{sh('Args')} -> {' | '.join(args_list)}.", color="magenta")

    miner_info = self.get_miner_info()
    colored_log(
        str(sh("Stats"))
        + f" -> Block: {miner_info['block']}"
        + f" | Stake: {miner_info['stake']:.4f}"
        + f" | Incentive: {miner_info['incentive']:.4f}"
        + f" | Trust: {miner_info['trust']:.4f}"
        + f" | Consensus: {miner_info['consensus']:.4f}.",
        color="cyan",
    )

    # Output stake
    requester_stake = get_caller_stake(self, synapse)
    if requester_stake is None:
        requester_stake = -1

    # Retrieve the coldkey of the caller
    caller_coldkey = get_coldkey_for_hotkey(self, hotkey)

    temp_string = f"Stake {int(requester_stake):,}"

    has_hotkey: bool = hotkey in self.hotkey_whitelist
    has_coldkey: bool = caller_coldkey in self.coldkey_whitelist

    if has_hotkey or has_coldkey:
        temp_string = "Whitelisted key"

    colored_log(
        #
        str(sh("Caller")) + f" -> {temp_string}" + f" | Hotkey {hotkey}.",
        color="yellow",
    )


def warm_up(model, local_args):
    """
    Warm the model up if using optimization.
    """
    start = time.perf_counter()
    c_args = copy.deepcopy(local_args)
    c_args["prompt"] = "An alchemist brewing a vibrant glowing potion."
    model(**c_args).images
    logger.info(f"Warm up is complete after {time.perf_counter() - start}")


def nsfw_image_filter(self, images):
    clip_input = self.processor(
        [self.transform(image) for image in images], return_tensors="pt"
    ).to(self.config.miner.device)

    images, nsfw = self.safety_checker.forward(
        images=images,
        clip_input=clip_input.pixel_values.to(
            self.config.miner.device,
        ),
    )

    return nsfw
