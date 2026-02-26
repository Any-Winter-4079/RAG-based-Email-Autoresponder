import modal
from config.general import modal_secret, rag_volume, VOLUME_PATH
from config.encoder_gpu import (
    image,
    GPU,
    MODAL_TIMEOUT,
    SCALEDOWN_WINDOW,
    MIN_CONTAINERS
)

# Modal
app = modal.App("encoder-gpu-upserter")

@app.function(
    image=image,
    gpu=GPU,
    secrets=[modal_secret],
    timeout=MODAL_TIMEOUT,
    scaledown_window=SCALEDOWN_WINDOW,
    min_containers=MIN_CONTAINERS,
    volumes={VOLUME_PATH: rag_volume},
)
def run_encoder_gpu_upserter(variant, timestamp, start_index, batch_size, encoder):
    from helpers.encoder import run_encoder_upserter

    # run upserter on GPU
    run_encoder_upserter(
        variant=variant,
        timestamp=timestamp,
        start_index=start_index,
        batch_size=batch_size,
        encoder=encoder,
        worker_name="run_encoder_gpu_upserter",
    )
