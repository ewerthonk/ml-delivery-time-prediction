# Standard Imports
from pathlib import Path

# Paths
PROJECT_PATH = Path(__file__).parent.parent.parent.resolve()
DATA_PATH = PROJECT_PATH.joinpath("data")
DATA_RAW_PATH = DATA_PATH.joinpath("raw")
DATA_PROCESSED_PATH = DATA_PATH.joinpath("processed")
RESOURCES_PATH = PROJECT_PATH.joinpath("resources")
