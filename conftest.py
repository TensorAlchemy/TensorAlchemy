import os
import sys
import warnings

# Suppress the eth_utils network warnings
# "does not have a valid ChainId."
# NOTE: It's not our bug, it's upstream
# TODO: Remove after updating bittensor
warnings.simplefilter("ignore")

# Get the absolute path of the project's root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))

# Add the project's root directory to the Python path
sys.path.insert(0, project_root)

# Use the older torch style for now

os.environ["CI"] = "true"
