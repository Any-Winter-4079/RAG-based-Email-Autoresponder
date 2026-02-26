from config.decoder import image as _base_image

GPU = "L40S"
SCALEDOWN_WINDOW = 60 # seconds
MODAL_TIMEOUT = 900 # seconds
MIN_CONTAINERS = 0 # 0 to make sure we don't pay 24/7

image = _base_image.pip_install(
    "llama-index",
    "fastembed-gpu",
    "qdrant-client>=1.14.2",
    "huggingface_hub"
)
