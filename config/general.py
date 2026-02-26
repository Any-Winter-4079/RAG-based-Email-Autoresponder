import modal

SECRET_NAME = "MUIA-SECRET"

modal_secret = modal.Secret.from_name(SECRET_NAME)

VOLUME_NAME = "muia-rag-volume"
VOLUME_PATH = "/root/volume"
QDRANT_PATH = f"{VOLUME_PATH}/qdrant"

rag_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

LEGACY_COLLECTIONS = [
    "raw_chunks_bm25",
    "manually_cleaned_chunks_bm25",
    "lm_cleaned_text_chunks_bm25",
    "lm_summary_chunks_bm25",
    "lm_q_and_a_chunks_bm25",
    "lm_q_and_a_for_q_only_chunks_bm25",
]
