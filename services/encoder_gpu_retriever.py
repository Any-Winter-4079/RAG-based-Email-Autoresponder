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
app = modal.App("encoder-gpu-retriever")

@app.function(
    image=image,
    gpu=GPU,
    secrets=[modal_secret],
    timeout=MODAL_TIMEOUT,
    scaledown_window=SCALEDOWN_WINDOW,
    min_containers=MIN_CONTAINERS,
    volumes={VOLUME_PATH: rag_volume},
)
def run_encoder_gpu_retriever(query_text, variant, encoder_name, top_k):
    from helpers.encoder import run_encoder_retriever

    # run retriever on GPU
    return run_encoder_retriever(
        query_text=query_text,
        variant=variant,
        encoder_name=encoder_name,
        top_k=top_k,
        worker_name="run_encoder_gpu_retriever",
    )
