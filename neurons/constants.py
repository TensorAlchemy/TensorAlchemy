import os
import sys


def is_test() -> bool:
    # Check if pytest is in the command line arguments
    return any("pytest" in arg for arg in sys.argv)


IS_CI_ENV: bool = os.environ.get("CI") == "true"

IA_BUCKET_NAME = "image-alchemy"
IA_TEST_BUCKET_NAME = "image-alchemy-test"
IA_MINER_BLACKLIST = "blacklist_for_miners.json"
IA_MINER_WHITELIST = "whitelist_for_miners.json"


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
IA_MINER_WARNINGLIST = "warninglist_for_miners.json"

VALIDATOR_SETTINGS_FILE = "https://raw.githubusercontent.com/TensorAlchemy/validator-settings/main/settings.json"

MAINNET_URL = "https://api.tensoralchemy.ai/api"
TESTNET_URL = "https://api-testnet.tensoralchemy.ai/api"
DEVELOP_URL = "https://api-develop.tensoralchemy.ai/api"

SAAS_DASHBOARD_DEVELOP_URL = "https://saas-develop.tensoralchemy.ai/signin"
SAAS_DASHBOARD_TESTNET_URL = "https://saas-testnet.tensoralchemy.ai/signin"
SAAS_DASHBOARD_MAINNET_URL = "https://saas.tensoralchemy.ai/signin"
SAAS_DASHBOARD_URL_EXPIRATION_TIME = 3600

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
