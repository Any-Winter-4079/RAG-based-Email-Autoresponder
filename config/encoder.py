# all encoders
ENCODERS = {
    "bm25": {
        "model_name": "Qdrant/bm25",
        "service": "encoder_cpu_upserter",
        "function": "run_encoder_cpu_upserter",
        "fastembed_kind": "sparse",
        "modifier": "idf",
    },
    "splade": {
        "model_name": "prithivida/Splade_PP_en_v1",
        "service": "encoder_gpu_upserter",
        "function": "run_encoder_gpu_upserter",
        "fastembed_kind": "sparse",
    },
    "colbert": {
        "model_name": "colbert-ir/colbertv2.0",
        "service": "encoder_gpu_upserter",
        "function": "run_encoder_gpu_upserter",
        "fastembed_kind": "late",
        "vector_size": 128,
        "distance": "cosine",
        "max_recommended_input_size": 256,
    },
    "bge_small": {
        "model_name": "BAAI/bge-small-en-v1.5",
        "service": "encoder_gpu_upserter",
        "function": "run_encoder_gpu_upserter",
        "fastembed_kind": "dense",
        "vector_size": 384,
        "distance": "cosine",
        "max_recommended_input_size": 8192,
    },
}
