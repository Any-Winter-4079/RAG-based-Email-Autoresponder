import modal

PYTHON_VERSION = "3.11"
SCALEDOWN_WINDOW = 60 # seconds
MODAL_TIMEOUT = 3600 # seconds
MIN_CONTAINERS = 0 # 0 to make sure we don't pay 24/7

PACKAGES = [
    "fastembed",
    "qdrant-client>=1.14.2"
]

image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .pip_install(*PACKAGES)
    .add_local_python_source("config", "helpers")
)
