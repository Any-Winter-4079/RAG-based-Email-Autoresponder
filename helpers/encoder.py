##################################
# Helper 1: Run encoder upserter #
##################################
def run_encoder_upserter(variant, timestamp, start_index, batch_size, encoder, worker_name):
    from config.encoder import ENCODERS
    from config.general import QDRANT_PATH
    from config.crawler_agent import (
        FILE_START,
        RAW_CHUNKS_PATH,
        MANUALLY_CLEANED_CHUNKS_PATH,
        LM_CLEANED_TEXT_SUBCHUNKS_PATH,
        LM_SUMMARY_SUBCHUNKS_PATH,
        LM_Q_AND_A_VALID_CHUNKS_PATH,
        LM_Q_AND_A_FOR_Q_ONLY_VALID_CHUNKS_PATH
    )
    import os
    import json
    from qdrant_client import QdrantClient, models
    from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding

    encode_paths = {
        "raw_chunks": RAW_CHUNKS_PATH,
        "manually_cleaned_chunks": MANUALLY_CLEANED_CHUNKS_PATH,
        "lm_cleaned_text_chunks": LM_CLEANED_TEXT_SUBCHUNKS_PATH,
        "lm_summary_chunks": LM_SUMMARY_SUBCHUNKS_PATH,
        "lm_q_and_a_chunks": LM_Q_AND_A_VALID_CHUNKS_PATH,
        "lm_q_and_a_for_q_only_chunks": LM_Q_AND_A_FOR_Q_ONLY_VALID_CHUNKS_PATH
    }

    collection_name = variant
    client = QdrantClient(path=QDRANT_PATH)
    if not client.collection_exists(collection_name=collection_name):
        print(f"{worker_name}: collection '{collection_name}' does not exist")
        return

    # load files to encode
    file_path = os.path.join(encode_paths[variant], f"{FILE_START}{timestamp}.jsonl")
    with open(file_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    batch = records[start_index:start_index + batch_size]
    print(f"{worker_name}: {variant}: loaded {len(batch)} records")

    # encode and upsert
    texts = []
    payloads = []
    point_ids = []
    if variant in ["lm_q_and_a_chunks", "lm_q_and_a_for_q_only_chunks"]:
        pair_start = sum(len(record["pairs"]) for record in records[:start_index])
        pair_offset = 0
        for record in batch:
            for pair_index, pair in enumerate(record["pairs"], start=1):
                if variant == "lm_q_and_a_chunks":
                    text = f"Q: {pair['question']}\nA: {pair['answer']}"
                else:
                    text = pair["question"]
                texts.append(text)
                payloads.append({
                    "variant": variant,
                    "timestamp": timestamp,
                    "url": record["url"],
                    "chunk_index": record["chunk_index"],
                    "pair_index": pair_index,
                    "question": pair["question"],
                    "answer": pair["answer"],
                    "decoder_token_count": pair["decoder_token_count"],
                    "encoder_token_count": pair["encoder_token_count"],
                })
                point_ids.append(pair_start + pair_offset)
                pair_offset += 1
    else:
        for record_offset, record in enumerate(batch):
            record_index = start_index + record_offset
            texts.append(record["text"])
            payloads.append({
                **record,
                "variant": variant,
                "timestamp": timestamp,
            })
            point_ids.append(record_index)

    encoder_config = ENCODERS[encoder]
    fastembed_kind = encoder_config["fastembed_kind"]
    if fastembed_kind == "sparse":
        encoder_model = SparseTextEmbedding(model_name=encoder_config["model_name"])
        encoder_embeddings = list(encoder_model.embed(texts))
    elif fastembed_kind == "late":
        encoder_model = LateInteractionTextEmbedding(model_name=encoder_config["model_name"])
        encoder_embeddings = list(encoder_model.embed(texts))
    elif fastembed_kind == "dense":
        encoder_model = TextEmbedding(model_name=encoder_config["model_name"])
        encoder_embeddings = list(encoder_model.embed(texts))
    else:
        raise ValueError(f"{worker_name} does not support fastembed_kind '{fastembed_kind}' for encoder '{encoder}'")

    points = []
    for i in range(len(texts)):
        embedding = encoder_embeddings[i]
        if fastembed_kind == "sparse":
            vectors = {
                encoder: models.SparseVector(
                    indices=embedding.indices.tolist(),
                    values=embedding.values.tolist()
                )
            }
        else:
            vector_value = embedding.tolist() if hasattr(embedding, "tolist") else embedding
            vectors = {
                encoder: vector_value
            }
        points.append(
            models.PointStruct(
                id=point_ids[i],
                payload=payloads[i],
                vector=vectors,
            )
        )
    client.upsert(collection_name=collection_name, points=points)
    print(f"{worker_name}: {variant}: upserted {len(points)} points into '{collection_name}'")

###################################
# Helper 2: Run encoder retriever #
###################################
def run_encoder_retriever(query_text, variant, encoder_name, top_k, worker_name):
    from config.encoder import ENCODERS
    from config.general import QDRANT_PATH
    from qdrant_client import QdrantClient, models
    from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding

    encoder_config = ENCODERS[encoder_name]
    fastembed_kind = encoder_config["fastembed_kind"]
    model_name = encoder_config["model_name"]
    client = QdrantClient(path=QDRANT_PATH)

    if fastembed_kind == "sparse":
        model = SparseTextEmbedding(model_name=model_name)
        query_embedding = list(model.query_embed([query_text]))[0]
        query = models.SparseVector(
            indices=query_embedding.indices.tolist(),
            values=query_embedding.values.tolist()
        )
    elif fastembed_kind == "late":
        model = LateInteractionTextEmbedding(model_name=model_name)
        query_embedding = list(model.query_embed([query_text]))[0]
        query = query_embedding.tolist() if hasattr(query_embedding, "tolist") else query_embedding
    elif fastembed_kind == "dense":
        model = TextEmbedding(model_name=model_name)
        query_embedding = list(model.query_embed([query_text]))[0]
        query = query_embedding.tolist() if hasattr(query_embedding, "tolist") else query_embedding
    else:
        raise ValueError(f"{worker_name} does not support fastembed_kind '{fastembed_kind}' for encoder '{encoder_name}'")

    # retrieve from collection
    results = client.query_points(
        collection_name=variant,
        query=query,
        using=encoder_name,
        limit=top_k,
        with_payload=True
    )

    return [
        {"score": point.score, "payload": point.payload or {}}
        for point in results.points
    ]
