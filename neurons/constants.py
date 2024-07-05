import os

IS_TEST: bool = False


IA_BUCKET_NAME = "image-alchemy"
IA_TEST_BUCKET_NAME = "image-alchemy-test"
IA_MINER_BLACKLIST = "blacklist_for_miners.json"
IA_MINER_WHITELIST = "whitelist_for_miners.json"


WANDB_MINER_PATH = os.path.expanduser("~/.TensorAlchemy/wandb/miner")
WANDB_VALIDATOR_PATH = os.path.expanduser("~/.TensorAlchemy/wandb/validator")

# Validator only
N_NEURONS = 12
N_NEURONS_TO_QUERY = 18
VPERMIT_TAO = 1024
FOLLOWUP_TIMEOUT = 10
MOVING_AVERAGE_ALPHA = 0.05
MOVING_AVERAGE_BETA = MOVING_AVERAGE_ALPHA / ((256 / 12) * 1.5)
EVENTS_RETENTION_SIZE = "2 GB"
VALIDATOR_DEFAULT_REQUEST_FREQUENCY = 60
VALIDATOR_DEFAULT_QUERY_TIMEOUT = 15
ENABLE_IMAGE2IMAGE = False

IA_VALIDATOR_BLACKLIST = "blacklist_for_validators.json"
IA_VALIDATOR_WHITELIST = "whitelist_for_validators.json"
IA_VALIDATOR_WEIGHT_FILES = "weights.json"
IA_VALIDATOR_SETTINGS_FILE = "validator_settings.json"
IA_MINER_WARNINGLIST = "warninglist_for_miners.json"

PROD_URL = "https://api.tensoralchemy.ai/api"
DEV_URL = "https://api-dev.tensoralchemy.ai/api"


VALIDATOR_SENTRY_DSN = (
    "https://740dd9e25d6e278889a1b9046c1f5e20@sentry.tensoralchemy.ai/4507287153737728"
)

NSFW_WORDLIST_URL = "https://raw.githubusercontent.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/master/en"
NSFW_WORDLIST_DEFAULT = [
    "anus",
    "ass",
    "asshole",
    "cock",
    "cum",
    "cumming",
    "dick",
    "hentai",
    "loli",
    "lolita",
    "naked",
    "nude",
    "orgasm",
    "penis",
    "porn",
    "pussy",
    "sex",
    "sexy",
    "tits",
    "undress",
    "undressed",
    "vagina",
]

MINIMUM_COMPUTES_FOR_SUBMIT = 3
