import modal
from config.general import modal_secret, rag_volume, VOLUME_PATH
from config.crawler_agent import (
    image,
    MODAL_TIMEOUT,
    CRAWL_MINUTES,
    CRAWL_HOUR,
    CRAWL_DAY,
    CRAWL_MONTH
)

# Modal
app = modal.App("crawler-agent")

@app.function(
        image=image,
        secrets=[modal_secret],
        # with Cron format "Minute Hour Day Month DayOfWeek":
        # "{CRAWL_MINUTES} {CRAWL_HOUR} {CRAWL_DAY} {CRAWL_MONTH} *" -> 9:00 AM on the {CRAWL_DAY} of {CRAWL_MONTH}
        schedule=modal.Cron(f"{CRAWL_MINUTES} {CRAWL_HOUR} {CRAWL_DAY} {CRAWL_MONTH} *"),
        timeout=MODAL_TIMEOUT,
        volumes={VOLUME_PATH: rag_volume},
)
async def run_crawler_agent():
    import os
    import json
    import glob
    import asyncio
    import datetime
    from transformers import AutoTokenizer
    from helpers.crawler_agent import crawl
    from helpers.decoder import count_tokens
    from llama_index.core.node_parser import SentenceSplitter
    from config.decoder import MODEL_PROFILES as DECODER_MODEL_PROFILES, DATA_CLEANER_PROFILE, EMAIL_WRITER_PROFILE
    from config.encoder import ENCODERS
    from config.crawler_agent import (
        START_URL,
        ADDITIONAL_URLS,
        MAX_DEPTH,
        MAX_LINKS_PER_PAGE,
        EXCLUDED_URLS,
        GSFS_BASE_URL,
        ALLOWED_GSFS_URLS,
        JINA_FETCH_TIMEOUT,
        CHUNK_OVERLAP,
        REUSE_CRAWL,
        REUSE_CRAWL_PAST_CURRENT_YEAR,
        REUSE_TIMESTAMP,
        RECREATE_QDRANT_COLLECTIONS,
        FILE_START,
        RAW_PATH,
        MANUALLY_CLEANED_PATH,
        RAW_CHUNKS_PATH,
        MANUALLY_CLEANED_CHUNKS_PATH,
        LM_CLEANED_TEXT_CHUNKS_PATH,
        LM_ABSTRACT_CHUNKS_PATH,
        LM_SUMMARY_CHUNKS_PATH,
        LM_Q_AND_A_CHUNKS_PATH,
        LM_CLEANED_TEXT_SUBCHUNKS_PATH,
        LM_SUMMARY_SUBCHUNKS_PATH,
        LM_Q_AND_A_VALID_CHUNKS_PATH,
        LM_Q_AND_A_FOR_Q_ONLY_VALID_CHUNKS_PATH,
        ENCODE_VARIANTS
    )

    # worker function to process single URL
    # https://modal.com/docs/guide/async#async-functions
    async def process_single_url(url, content):
        local_results = {
            "url": url,
            "content": content,
            "raw_chunks": [],
            "manually_cleaned_chunks": [],
            "lm_cleaned_text_chunks": [],
            "lm_abstract_chunks": [],
            "lm_summary_chunks": [],
            "lm_q_and_a_chunks": []
        }

        # count raw version tokens for decoder
        raw_decoder_tokens = count_tokens(decoder_tokenizer, content["raw"]["text"])
        content["raw"]["decoder_token_count"] = raw_decoder_tokens

        # count raw version tokens for encoder
        raw_encoder_tokens = count_tokens(encoder_tokenizer, content["raw"]["text"])
        content["raw"]["encoder_token_count"] = raw_encoder_tokens
        
        # count manually cleaned tokens for decoder
        manually_cleaned_decoder_tokens = count_tokens(decoder_tokenizer, content["manually_cleaned"]["text"])
        content["manually_cleaned"]["decoder_token_count"] = manually_cleaned_decoder_tokens

        # count manually cleaned tokens for encoder
        manually_cleaned_encoder_tokens = count_tokens(encoder_tokenizer, content["manually_cleaned"]["text"])
        content["manually_cleaned"]["encoder_token_count"] = manually_cleaned_encoder_tokens

        # chunk raw content
        try:
            raw_text_chunks = embedding_splitter.split_text(content["raw"]["text"])
            for idx, chunk_text in enumerate(raw_text_chunks):
                local_results["raw_chunks"].append({
                    "url": url,
                    "chunk_index": idx,
                    "text": chunk_text,
                    "decoder_token_count": count_tokens(decoder_tokenizer, chunk_text),
                    "encoder_token_count": count_tokens(encoder_tokenizer, chunk_text)
                })
        except Exception as e:
            print(f"run_crawler_agent: error splitting raw {url}: {e}")
        
        # chunk manually cleaned content
        try:
            text_chunks = embedding_splitter.split_text(content["manually_cleaned"]["text"])
            for idx, chunk_text in enumerate(text_chunks):
                local_results["manually_cleaned_chunks"].append({
                    "url": url,
                    "chunk_index": idx,
                    "text": chunk_text,
                    "decoder_token_count": count_tokens(decoder_tokenizer, chunk_text),
                    "encoder_token_count": count_tokens(encoder_tokenizer, chunk_text)
                })
        except Exception as e:
            print(f"run_crawler_agent: error splitting manually cleaned {url}: {e}")

        # clean with lm
        try:
            print(f"run_crawler_agent: decoder worker is data cleaning: {url}")
            # reset page history, last cleaned_text for new page (for context)
            page_history = {}
            previous_cleaned_text = "None (Start of Document)"

            # chunk for decoder
            text_chunks = lm_splitter.split_text(content["manually_cleaned"]["text"])
            
            for idx, chunk_text in enumerate(text_chunks):
                # get current date and time
                current_date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") # e.g., "2026-02-02 09:00:00"

                # select decoder configuration for data cleaning
                model_config = DECODER_MODEL_PROFILES[DATA_CLEANER_PROFILE].copy()

                # pop (and save) "prompt_template" (run_qwen3_lm_or_vlm would not expect it as model_config)
                prompt_template = model_config.pop("prompt_template")

                # pop "max_chunk_size" (run_qwen3_lm_or_vlm would not expect it as model_config)
                model_config.pop("max_chunk_size")

                # get up to 20 chunks abstract, summary worth of context
                sorted_indices = sorted(page_history.keys())
                if len(sorted_indices) > 20:
                    selected_indices = sorted_indices[:5] + sorted_indices[-15:]
                else:
                    selected_indices = sorted_indices
                if selected_indices:
                    history_context = ""
                    for i in selected_indices:
                        entry = page_history[i]
                        history_context += f"- Chunk {i}:\n\tAbstract: {entry['abstract']}\n\tSummary: {entry['summary']}\n"
                else:
                    history_context = "None (Start of Document)"

                # construct prompt
                try:
                    prompt = prompt_template.format(
                        datetime=current_date_time,
                        page_history_context=history_context,
                        previous_chunk_context=previous_cleaned_text,
                        text=chunk_text
                    )
                except KeyError as e:
                    print(f"run_crawler_agent: error formatting prompt template: {e}")
                    continue

                # run decoder (without "template" in model_config)
                try:
                    lm_cleaned_content, prompt_text = await run_qwen3_lm_or_vlm.remote.aio(
                        context=[],
                        current_turn_input_text=prompt,
                        current_turn_image_in_bytes=None,
                        **model_config,
                        decoder_profile=DATA_CLEANER_PROFILE
                    )
                except Exception as e:
                    print(f"run_crawler_agent: decoder generation failed: {e}")
                    continue

                if DECODER_MODEL_PROFILES[DATA_CLEANER_PROFILE]["return_prompt_text"]:
                    print(f"{prompt_text}\n\n")

                if lm_cleaned_content and len(lm_cleaned_content) == 5:
                    abstract, summary, cleanedtext, questions, answers = lm_cleaned_content
                    
                    # update abstracts, summaries, last cleaned_text (for context)
                    if abstract and summary:
                        page_history[idx] = {
                             "abstract": abstract,
                             "summary": summary
                        }
                    if cleanedtext:
                        previous_cleaned_text = cleanedtext
                        
                    # process abstract
                    if abstract:
                        abstract_decoder_tokens = count_tokens(decoder_tokenizer, abstract)
                        abstract_encoder_tokens = count_tokens(encoder_tokenizer, abstract)
                        local_results["lm_abstract_chunks"].append({
                            "url": url,
                            "chunk_index": idx,
                            "text": abstract,
                            "decoder_token_count": abstract_decoder_tokens,
                            "encoder_token_count": abstract_encoder_tokens
                        })

                    # process summary
                    if summary:
                        summary_decoder_tokens = count_tokens(decoder_tokenizer, summary)
                        summary_encoder_tokens = count_tokens(encoder_tokenizer, summary)
                        local_results["lm_summary_chunks"].append({
                            "url": url,
                            "chunk_index": idx,
                            "text": summary,
                            "decoder_token_count": summary_decoder_tokens,
                            "encoder_token_count": summary_encoder_tokens
                        })

                    # process cleaned text
                    if cleanedtext:
                        cleaned_text_decoder_tokens = count_tokens(decoder_tokenizer, cleanedtext)
                        cleaned_text_encoder_tokens = count_tokens(encoder_tokenizer, cleanedtext)
                        local_results["lm_cleaned_text_chunks"].append({
                            "url": url,
                            "chunk_index": idx,
                            "text": cleanedtext,
                            "decoder_token_count": cleaned_text_decoder_tokens,
                            "encoder_token_count": cleaned_text_encoder_tokens
                        })

                    # process q&a
                    if questions and answers:
                        pairs = []
                        for question, answer in zip(questions, answers):
                            q_decoder_tokens = count_tokens(decoder_tokenizer, question)
                            q_encoder_tokens = count_tokens(encoder_tokenizer, question)          
                            a_decoder_tokens = count_tokens(decoder_tokenizer, answer)
                            a_encoder_tokens = count_tokens(encoder_tokenizer, answer)
                            # calculate max
                            q_and_a_decoder_tokens = max(q_decoder_tokens, a_decoder_tokens)
                            q_and_a_encoder_tokens = max(q_encoder_tokens, a_encoder_tokens)

                            pairs.append({
                                "question": question,
                                "answer": answer,
                                "decoder_token_count": q_and_a_decoder_tokens,
                                "encoder_token_count": q_and_a_encoder_tokens
                            })
                        local_results["lm_q_and_a_chunks"].append({
                            "url": url,
                            "chunk_index": idx,
                            "pairs": pairs,
                        })

        except Exception as e:
            print(f"run_crawler_agent: error processing {url}: {e}")
        
        return local_results

    # load encoder tokenizer of the model with the smallest max recommended input size
    encoder_sizes = {
        name: config["max_recommended_input_size"]
        for name, config in ENCODERS.items()
        if "max_recommended_input_size" in config
    }
    if not encoder_sizes:
        print("run_crawler_agent: no encoder max_recommended_input_size found")
        return
    chunking_encoder = min(encoder_sizes, key=encoder_sizes.get)
    encoder_path = ENCODERS[chunking_encoder]["model_name"]
    decoder_path = DECODER_MODEL_PROFILES[EMAIL_WRITER_PROFILE]["model_path"]
    encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_path, trust_remote_code=True)
    decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_path, trust_remote_code=True)

    embedding_chunk_size = encoder_sizes[chunking_encoder]
    embedding_splitter = SentenceSplitter(
        chunk_size=embedding_chunk_size,
        chunk_overlap=CHUNK_OVERLAP,
        tokenizer=lambda text: encoder_tokenizer.encode(text, add_special_tokens=False)
    )

    # save chunk-level data helper
    def save_chunks(chunk_list, json_path, txt_path, label):
        with open(json_path, "w", encoding="utf-8") as f_json, open(txt_path, "w", encoding="utf-8") as f_txt:
            for i, chunk in enumerate(chunk_list):
                separator = "=" * 150
                f_json.write(json.dumps(chunk) + "\n")

                if "text" in chunk:
                    content = chunk["text"]
                    token_info = f"Tokens {decoder_path}: {chunk['decoder_token_count']:,} | Tokens {encoder_path}: {chunk['encoder_token_count']:,}"
                elif "pairs" in chunk:
                    content = "\n".join([f"Q: {pair['question']}\nA: {pair['answer']}" for pair in chunk["pairs"]])
                    max_decoder_tokens = max((pair["decoder_token_count"] for pair in chunk["pairs"]))
                    max_encoder_tokens = max((pair["encoder_token_count"] for pair in chunk["pairs"]))
                    token_info = f"Pairs: {len(chunk['pairs'])} | Tokens (max) {decoder_path}: {max_decoder_tokens:,} | Tokens (max) {encoder_path}: {max_encoder_tokens:,}"
                else:
                    content = ""
                    token_info = ""

                if "subchunk_index" in chunk:
                    chunk_label = f"{chunk['chunk_index']}.{chunk['subchunk_index']}"
                else:
                    chunk_label = str(i + 1)
                header = f"{label} CHUNK {chunk_label} [Source: {chunk['url']}] | {token_info}"
                f_txt.write(f"\n{separator}\n{header}\n{separator}\n{content}\n")

    variant_file_paths = {}
    variants_to_encode = {}
    encode_timestamp = None
    current_year = datetime.datetime.now().year
    if not ENCODE_VARIANTS:
        print("run_crawler_agent: ENCODE_VARIANTS is empty")
        return
    variant_paths = {}
    for variant in ENCODE_VARIANTS.keys():
        if variant == "raw_chunks":
            variant_paths[variant] = RAW_CHUNKS_PATH
        elif variant == "manually_cleaned_chunks":
            variant_paths[variant] = MANUALLY_CLEANED_CHUNKS_PATH
        elif variant == "lm_cleaned_text_chunks":
            variant_paths[variant] = LM_CLEANED_TEXT_CHUNKS_PATH
        elif variant == "lm_abstract_chunks":
            variant_paths[variant] = LM_ABSTRACT_CHUNKS_PATH
        elif variant == "lm_summary_chunks":
            variant_paths[variant] = LM_SUMMARY_CHUNKS_PATH
        elif variant == "lm_q_and_a_chunks":
            variant_paths[variant] = LM_Q_AND_A_CHUNKS_PATH
        elif variant == "lm_q_and_a_for_q_only_chunks":
            variant_paths[variant] = LM_Q_AND_A_CHUNKS_PATH
        else:
            print(f"run_crawler_agent: invalid ENCODE_VARIANTS entry '{variant}'")
            return

    if REUSE_CRAWL:
        reuse_timestamp = str(REUSE_TIMESTAMP).strip() if REUSE_TIMESTAMP else ""

        # if timestamp set:
        if reuse_timestamp:
            try:
                timestamp_year = int(reuse_timestamp[:4])
            except Exception as e:
                # if the timestamp has the wrong format, error out and return
                print(f"run_crawler_agent: invalid REUSE_TIMESTAMP '{reuse_timestamp}'. Valid example: '20260203_161009'")
                return
            # if the files exist and we are fine reusing them even past >1 year, or they are <1 year old, use the files
            if REUSE_CRAWL_PAST_CURRENT_YEAR or timestamp_year == current_year:
                for variant in ENCODE_VARIANTS.keys():
                    candidate_file = os.path.join(
                        variant_paths[variant],
                        f"{FILE_START}{reuse_timestamp}.jsonl"
                    )
                    if not os.path.exists(candidate_file):
                        print(f"run_crawler_agent: '{candidate_file}' not found. Are you sure it exists on Modal?")
                        return
                    variant_file_paths[variant] = candidate_file
                encode_timestamp = reuse_timestamp
                anchor_variant = next(iter(ENCODE_VARIANTS))
                print(f"run_crawler_agent: reusing crawl {os.path.basename(variant_file_paths[anchor_variant])}")
            # else, discard, crawl and use a new version
            else:
                print(f"run_crawler_agent: REUSE_TIMESTAMP '{reuse_timestamp}' is outside current year ({current_year}): crawling fresh data")

        # else (we want to reuse but no timestamp is set):
        else:
            year_filter = "*" if REUSE_CRAWL_PAST_CURRENT_YEAR else f"{current_year}*"
            anchor_variant = next(iter(ENCODE_VARIANTS))
            existing_text_files = glob.glob(
                os.path.join(variant_paths[anchor_variant], f"{FILE_START}{year_filter}.jsonl")
            )
            # if no file exists (or no file within the current year), error out and return
            if not existing_text_files:
                print(f"run_crawler_agent: REUSE_CRAWL is enabled but no '{anchor_variant}' file was found. Are you sure it exists on Modal?")
                return

            # if they exist, use them
            anchor_file = max(existing_text_files, key=os.path.getctime)
            anchor_basename = os.path.basename(anchor_file)
            encode_timestamp = anchor_basename[len(FILE_START):-len(".jsonl")]
            for variant in ENCODE_VARIANTS.keys():
                candidate_file = os.path.join(variant_paths[variant], os.path.basename(anchor_file))
                if not os.path.exists(candidate_file):
                    print(f"run_crawler_agent: '{candidate_file}' not found. Are you sure it exists on Modal?")
                    return
                variant_file_paths[variant] = candidate_file
            print(f"run_crawler_agent: reusing latest eligible crawl: {os.path.basename(anchor_file)}")

    # if a reusable crawl is available, load it (from the volume)
    if variant_file_paths:
        for variant, file_path in variant_file_paths.items():
            with open(file_path, "r", encoding="utf-8") as f:
                variants_to_encode[variant] = [json.loads(line) for line in f]
    
    # else, fetch, clean up, and store the data on the volume
    else:
        try:
            # create text splitter using the decoder tokenizer and a max chunk size
            lm_chunk_size = DECODER_MODEL_PROFILES[DATA_CLEANER_PROFILE]["max_chunk_size"]
            lm_splitter = SentenceSplitter(
                chunk_size=lm_chunk_size,
                chunk_overlap=CHUNK_OVERLAP,
                tokenizer=lambda text: decoder_tokenizer.encode(text, add_special_tokens=False)
            )
        
        except Exception as e:
            print(f"run_crawler_agent: error loading tokenizers or creating splitter: {e}")
            return

        # find decoder service
        try:
            run_qwen3_lm_or_vlm = modal.Function.from_name("decoder", "run_qwen3_lm_or_vlm")
        except Exception as e:
            print(f"run_crawler_agent: failed to find decoder service. Is it deployed? Error: {e}")
            return
    
        # crawl
        url_content_dict = crawl(
            start_url=START_URL,
            additional_urls=ADDITIONAL_URLS,
            max_depth=MAX_DEPTH,
            max_links_per_page=MAX_LINKS_PER_PAGE,
            timeout=JINA_FETCH_TIMEOUT,
            excluded_urls=EXCLUDED_URLS,
            gsfs_base_url=GSFS_BASE_URL,
            allowed_gsfs_urls=ALLOWED_GSFS_URLS
        )

        if not url_content_dict:
            print("run_crawler_agent: no content found")
            return
        else:
            print(f"run_crawler_agent: {len(url_content_dict)} URLs to process")

        # post-process
        raw_chunks = []
        manually_cleaned_chunks = []
        lm_cleaned_text_chunks = []
        lm_abstract_chunks = []
        lm_summary_chunks = []
        lm_q_and_a_chunks = []

        max_token_lengths = {
            "raw": {
                "decoder": 0,
                "encoder": 0
            },
            "manually_cleaned": {
                "decoder": 0,
                "encoder": 0
            },
            "lm_cleaned": {
                "cleaned_text": {
                    "decoder": 0,
                    "encoder": 0
                },
                "abstract": {
                    "decoder": 0,
                    "encoder": 0
                },
                "summary": {
                    "decoder": 0,
                    "encoder": 0
                },
                "q_and_a": {
                    "decoder": 0,
                    "encoder": 0
                },
            }
        }

        # create tasks and run in parallel
        tasks = [process_single_url(url, content) for url, content in url_content_dict.items()]
        results = await asyncio.gather(*tasks)

        # aggregate results
        for result in results:
            url = result["url"]
            content = result["content"]

            # update original dict
            url_content_dict[url] = content

            # update max_token_lengths
            if content["raw"]["decoder_token_count"] > max_token_lengths["raw"]["decoder"]:
                max_token_lengths["raw"]["decoder"] = content["raw"]["decoder_token_count"]
            if content["raw"]["encoder_token_count"] > max_token_lengths["raw"]["encoder"]:
                max_token_lengths["raw"]["encoder"] = content["raw"]["encoder_token_count"]

            if content["manually_cleaned"]["decoder_token_count"] > max_token_lengths["manually_cleaned"]["decoder"]:
                max_token_lengths["manually_cleaned"]["decoder"] = content["manually_cleaned"]["decoder_token_count"]
            if content["manually_cleaned"]["encoder_token_count"] > max_token_lengths["manually_cleaned"]["encoder"]:
                max_token_lengths["manually_cleaned"]["encoder"] = content["manually_cleaned"]["encoder_token_count"]

            # extend chunk lists and update max_token_lengths
            raw_chunks.extend(result["raw_chunks"])
            manually_cleaned_chunks.extend(result["manually_cleaned_chunks"])

            lm_cleaned_text_chunks.extend(result["lm_cleaned_text_chunks"])
            for chunk in result["lm_cleaned_text_chunks"]:
                if chunk["decoder_token_count"] > max_token_lengths["lm_cleaned"]["cleaned_text"]["decoder"]:
                    max_token_lengths["lm_cleaned"]["cleaned_text"]["decoder"] = chunk["decoder_token_count"]
                if chunk["encoder_token_count"] > max_token_lengths["lm_cleaned"]["cleaned_text"]["encoder"]:
                    max_token_lengths["lm_cleaned"]["cleaned_text"]["encoder"] = chunk["encoder_token_count"]

            lm_abstract_chunks.extend(result["lm_abstract_chunks"])
            for chunk in result["lm_abstract_chunks"]:
                if chunk["decoder_token_count"] > max_token_lengths["lm_cleaned"]["abstract"]["decoder"]:
                    max_token_lengths["lm_cleaned"]["abstract"]["decoder"] = chunk["decoder_token_count"]
                if chunk["encoder_token_count"] > max_token_lengths["lm_cleaned"]["abstract"]["encoder"]:
                    max_token_lengths["lm_cleaned"]["abstract"]["encoder"] = chunk["encoder_token_count"]

            lm_summary_chunks.extend(result["lm_summary_chunks"])
            for chunk in result["lm_summary_chunks"]:
                if chunk["decoder_token_count"] > max_token_lengths["lm_cleaned"]["summary"]["decoder"]:
                    max_token_lengths["lm_cleaned"]["summary"]["decoder"] = chunk["decoder_token_count"]
                if chunk["encoder_token_count"] > max_token_lengths["lm_cleaned"]["summary"]["encoder"]:
                    max_token_lengths["lm_cleaned"]["summary"]["encoder"] = chunk["encoder_token_count"]

            lm_q_and_a_chunks.extend(result["lm_q_and_a_chunks"])
            for chunk in result["lm_q_and_a_chunks"]:
                for pair in chunk["pairs"]:
                    if pair["decoder_token_count"] > max_token_lengths["lm_cleaned"]["q_and_a"]["decoder"]:
                        max_token_lengths["lm_cleaned"]["q_and_a"]["decoder"] = pair["decoder_token_count"]
                    if pair["encoder_token_count"] > max_token_lengths["lm_cleaned"]["q_and_a"]["encoder"]:
                        max_token_lengths["lm_cleaned"]["q_and_a"]["encoder"] = pair["encoder_token_count"]
        
        # store all versions (raw, manually cleaned, lm cleaned, unchuncked and chunked) on file
        for path in [RAW_PATH, MANUALLY_CLEANED_PATH, RAW_CHUNKS_PATH, MANUALLY_CLEANED_CHUNKS_PATH,
                    LM_CLEANED_TEXT_CHUNKS_PATH, LM_ABSTRACT_CHUNKS_PATH, LM_SUMMARY_CHUNKS_PATH, LM_Q_AND_A_CHUNKS_PATH]:
            os.makedirs(path, exist_ok=True)

        now = datetime.datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        encode_timestamp = timestamp

        files = {
            "raw": {
                "json": os.path.join(RAW_PATH, f"{FILE_START}{timestamp}.jsonl"),
                "txt": os.path.join(RAW_PATH, f"{FILE_START}{timestamp}.txt")
            },
            "manually_cleaned": {
                "json": os.path.join(MANUALLY_CLEANED_PATH, f"{FILE_START}{timestamp}.jsonl"),
                "txt": os.path.join(MANUALLY_CLEANED_PATH, f"{FILE_START}{timestamp}.txt")
            },
            "raw_chunks": {
                "json": os.path.join(RAW_CHUNKS_PATH, f"{FILE_START}{timestamp}.jsonl"),
                "txt": os.path.join(RAW_CHUNKS_PATH, f"{FILE_START}{timestamp}.txt")
            },
            "manually_cleaned_chunks": {
                "json": os.path.join(MANUALLY_CLEANED_CHUNKS_PATH, f"{FILE_START}{timestamp}.jsonl"),
                "txt": os.path.join(MANUALLY_CLEANED_CHUNKS_PATH, f"{FILE_START}{timestamp}.txt")
            },
            "lm_cleaned_text_chunks": {
                "json": os.path.join(LM_CLEANED_TEXT_CHUNKS_PATH, f"{FILE_START}{timestamp}.jsonl"),
                "txt": os.path.join(LM_CLEANED_TEXT_CHUNKS_PATH, f"{FILE_START}{timestamp}.txt")
            },
            "lm_abstract_chunks": {
                "json": os.path.join(LM_ABSTRACT_CHUNKS_PATH, f"{FILE_START}{timestamp}.jsonl"),
                "txt": os.path.join(LM_ABSTRACT_CHUNKS_PATH, f"{FILE_START}{timestamp}.txt")
            },
            "lm_summary_chunks": {
                "json": os.path.join(LM_SUMMARY_CHUNKS_PATH, f"{FILE_START}{timestamp}.jsonl"),
                "txt": os.path.join(LM_SUMMARY_CHUNKS_PATH, f"{FILE_START}{timestamp}.txt")
            },
            "lm_q_and_a_chunks": {
                "json": os.path.join(LM_Q_AND_A_CHUNKS_PATH, f"{FILE_START}{timestamp}.jsonl"),
                "txt": os.path.join(LM_Q_AND_A_CHUNKS_PATH, f"{FILE_START}{timestamp}.txt")
            }
        }

        try:
            # https://modal.com/docs/reference/modal.Volume
            with open(files["raw"]["json"], "w", encoding="utf-8") as f_raw_json, \
                open(files["raw"]["txt"], "w", encoding="utf-8") as f_raw_txt, \
                open(files["manually_cleaned"]["json"], "w", encoding="utf-8") as f_manually_cleaned_json, \
                open(files["manually_cleaned"]["txt"], "w", encoding="utf-8") as f_manually_cleaned_txt:

                for i, (url, data) in enumerate(url_content_dict.items()):
                    separator = "=" * 150

                    # raw
                    raw_data = data["raw"]
                    f_raw_json.write(json.dumps({"url": url, "data": raw_data}) + "\n")
                    header_raw = f"RAW PAGE {i+1}: {url} | Tokens {decoder_path}: {raw_data['decoder_token_count']:,} | Tokens {encoder_path}: {raw_data['encoder_token_count']:,}"
                    f_raw_txt.write(f"\n{separator}\n{header_raw}\n{separator}\n{raw_data['text']}\n")

                    # manually cleaned
                    manually_cleaned_data = data["manually_cleaned"]
                    f_manually_cleaned_json.write(json.dumps({"url": url, "data": manually_cleaned_data}) + "\n")
                    header_manually_cleaned = f"MANUALLY CLEANED PAGE {i+1}: {url} | Tokens {decoder_path}: {manually_cleaned_data['decoder_token_count']:,} | Tokens {encoder_path}: {manually_cleaned_data['encoder_token_count']:,}"
                    f_manually_cleaned_txt.write(f"\n{separator}\n{header_manually_cleaned}\n{separator}\n{manually_cleaned_data['text']}\n")

            # save all chunk types
            save_chunks(raw_chunks, files["raw_chunks"]["json"], files["raw_chunks"]["txt"], "RAW")
            save_chunks(manually_cleaned_chunks, files["manually_cleaned_chunks"]["json"], files["manually_cleaned_chunks"]["txt"], "MANUALLY CLEANED")
            save_chunks(lm_cleaned_text_chunks, files["lm_cleaned_text_chunks"]["json"], files["lm_cleaned_text_chunks"]["txt"], "LM CLEANED TEXT")
            save_chunks(lm_abstract_chunks, files["lm_abstract_chunks"]["json"], files["lm_abstract_chunks"]["txt"], "LM ABSTRACT")
            save_chunks(lm_summary_chunks, files["lm_summary_chunks"]["json"], files["lm_summary_chunks"]["txt"], "LM SUMMARY")
            save_chunks(lm_q_and_a_chunks, files["lm_q_and_a_chunks"]["json"], files["lm_q_and_a_chunks"]["txt"], "LM Q&A")

            rag_volume.commit()
            print(f"run_crawler_agent: data saved and volume committed")
            print(f"run_crawler_agent: max token lengths:\n{max_token_lengths}")

        except Exception as e:
            print(f"run_crawler_agent: error saving files: {e}")

        # select variants to encode
        if "raw_chunks" in ENCODE_VARIANTS:
            variants_to_encode["raw_chunks"] = raw_chunks
        if "manually_cleaned_chunks" in ENCODE_VARIANTS:
            variants_to_encode["manually_cleaned_chunks"] = manually_cleaned_chunks
        if "lm_cleaned_text_chunks" in ENCODE_VARIANTS:
            variants_to_encode["lm_cleaned_text_chunks"] = lm_cleaned_text_chunks
        if "lm_abstract_chunks" in ENCODE_VARIANTS:
            variants_to_encode["lm_abstract_chunks"] = lm_abstract_chunks
        if "lm_summary_chunks" in ENCODE_VARIANTS:
            variants_to_encode["lm_summary_chunks"] = lm_summary_chunks
        if "lm_q_and_a_chunks" in ENCODE_VARIANTS:
            variants_to_encode["lm_q_and_a_chunks"] = lm_q_and_a_chunks
        if "lm_q_and_a_for_q_only_chunks" in ENCODE_VARIANTS:
            variants_to_encode["lm_q_and_a_for_q_only_chunks"] = lm_q_and_a_chunks

    if not variants_to_encode:
        print("run_crawler_agent: no variants selected to encode")
        return

    # process data before encoding:
    # if raw data is selected:              keep raw data as is
    # if manually cleaned data is selected: keep manually cleaned data as is
    # if lm cleaned data is selected:       subchunk LM cleaned data
    # if lm summaries is selected:          subchunk LM summaries
    # if lm Q&A is selected:                remove Q&A pairs over max size
    # write any data to encode (that isn't already in a file) to file
    encode_stats = []
    for variant, chunks in variants_to_encode.items():
        items_to_encode = 0

        if variant in ["lm_q_and_a_chunks", "lm_q_and_a_for_q_only_chunks"]:
            skip_label = "Q&A pairs" if variant == "lm_q_and_a_chunks" else "questions"
            print(f"run_crawler_agent: processing {variant}; skipping overlength {skip_label}")
            skipped_example = None
            pair_count = 0
            valid_chunks = []
            for chunk in chunks:
                valid_pairs = []
                for pair in chunk["pairs"]:
                    question = pair["question"].strip()
                    answer = pair["answer"].strip()
                    if variant == "lm_q_and_a_chunks":
                        text = f"Q: {question}\nA: {answer}".strip()
                    else:
                        text = question
                    pair_count += 1
                    # for Q&A, splitting probably does not make sense, so if we surpass
                    # the max_recommended_input_size, we simply skip the Q&A pair
                    token_ids = encoder_tokenizer.encode(text, add_special_tokens=False)
                    if len(token_ids) > embedding_chunk_size:
                        if skipped_example is None:
                            preview_limit = embedding_chunk_size * 3
                            skipped_example = text[:preview_limit] + "..." if len(text) > preview_limit else text
                        continue
                    q_decoder_tokens = count_tokens(decoder_tokenizer, question)
                    a_decoder_tokens = count_tokens(decoder_tokenizer, answer)
                    q_encoder_tokens = count_tokens(encoder_tokenizer, question)
                    a_encoder_tokens = count_tokens(encoder_tokenizer, answer)
                    pair["decoder_token_count"] = max(q_decoder_tokens, a_decoder_tokens)
                    pair["encoder_token_count"] = max(q_encoder_tokens, a_encoder_tokens)
                    valid_pairs.append(pair)
                    items_to_encode += 1
                if valid_pairs:
                    valid_chunks.append({
                        "url": chunk["url"],
                        "chunk_index": chunk["chunk_index"],
                        "pairs": valid_pairs
                    })
            if variant == "lm_q_and_a_chunks":
                os.makedirs(LM_Q_AND_A_VALID_CHUNKS_PATH, exist_ok=True)
                lm_q_and_a_json = os.path.join(LM_Q_AND_A_VALID_CHUNKS_PATH, f"{FILE_START}{encode_timestamp}.jsonl")
                lm_q_and_a_txt = os.path.join(LM_Q_AND_A_VALID_CHUNKS_PATH, f"{FILE_START}{encode_timestamp}.txt")
                save_chunks(valid_chunks, lm_q_and_a_json, lm_q_and_a_txt, "LM Q&A")
            else:
                os.makedirs(LM_Q_AND_A_FOR_Q_ONLY_VALID_CHUNKS_PATH, exist_ok=True)
                lm_q_only_json = os.path.join(LM_Q_AND_A_FOR_Q_ONLY_VALID_CHUNKS_PATH, f"{FILE_START}{encode_timestamp}.jsonl")
                lm_q_only_txt = os.path.join(LM_Q_AND_A_FOR_Q_ONLY_VALID_CHUNKS_PATH, f"{FILE_START}{encode_timestamp}.txt")
                save_chunks(valid_chunks, lm_q_only_json, lm_q_only_txt, "LM Q&A for Q only")
        else:
            if variant in ["raw_chunks", "manually_cleaned_chunks"]:
                print(f"run_crawler_agent: skipping re-splitting for {variant} (already chunked)")
                items_to_encode = len(chunks)
            else:
                print(f"run_crawler_agent: splitting for {variant}")
                prepared_chunks = []
                for chunk_index, chunk in enumerate(chunks, start=1):
                    text = chunk["text"].strip()
                    # for summaries, etc., we can split and keep meaning
                    split_texts = embedding_splitter.split_text(text)
                    for split_index, split_text in enumerate(split_texts):
                        items_to_encode += 1
                        prepared_chunks.append({
                            "url": chunk["url"],
                            "chunk_index": chunk_index,
                            "subchunk_index": split_index + 1,
                            "text": split_text,
                            "decoder_token_count": count_tokens(decoder_tokenizer, split_text),
                            "encoder_token_count": count_tokens(encoder_tokenizer, split_text)
                        })
                if variant == "lm_cleaned_text_chunks":
                    os.makedirs(LM_CLEANED_TEXT_SUBCHUNKS_PATH, exist_ok=True)
                    lm_cleaned_json = os.path.join(LM_CLEANED_TEXT_SUBCHUNKS_PATH, f"{FILE_START}{encode_timestamp}.jsonl")
                    lm_cleaned_txt = os.path.join(LM_CLEANED_TEXT_SUBCHUNKS_PATH, f"{FILE_START}{encode_timestamp}.txt")
                    save_chunks(prepared_chunks, lm_cleaned_json, lm_cleaned_txt, "LM CLEANED TEXT")
                if variant == "lm_summary_chunks":
                    os.makedirs(LM_SUMMARY_SUBCHUNKS_PATH, exist_ok=True)
                    lm_summary_json = os.path.join(LM_SUMMARY_SUBCHUNKS_PATH, f"{FILE_START}{encode_timestamp}.jsonl")
                    lm_summary_txt = os.path.join(LM_SUMMARY_SUBCHUNKS_PATH, f"{FILE_START}{encode_timestamp}.txt")
                    save_chunks(prepared_chunks, lm_summary_json, lm_summary_txt, "LM SUMMARY")

        encode_stats.append({
            "variant": variant,
            "chunk_count": len(chunks),
            "items_to_encode": items_to_encode,
            "q_and_a_pair_count": pair_count if variant in ["lm_q_and_a_chunks", "lm_q_and_a_for_q_only_chunks"] else None,
            "skipped_example": skipped_example if variant in ["lm_q_and_a_chunks", "lm_q_and_a_for_q_only_chunks"] else None
        })

    print("run_crawler_agent: summary:")
    for stats in encode_stats:
        variant = stats["variant"]
        chunk_count = stats["chunk_count"]
        items_to_encode = stats["items_to_encode"]
        if variant in ["lm_q_and_a_chunks", "lm_q_and_a_for_q_only_chunks"]:
            pair_count = stats["q_and_a_pair_count"]
            skipped_pairs = pair_count - items_to_encode
            print(f"run_crawler_agent: {variant}: ({chunk_count} chunks) {pair_count} pairs: {items_to_encode} items to encode")
            if skipped_pairs:
                skipped_label = "questions" if variant == "lm_q_and_a_for_q_only_chunks" else "Q&A pairs"
                print(f"run_crawler_agent: skipped {skipped_pairs} {skipped_label} due to exceeding embedding chunk size")
                if stats["skipped_example"]:
                    print(f"run_crawler_agent: skipped {skipped_label} example: {stats['skipped_example']}")
        else:
            print(f"run_crawler_agent: {variant}: {chunk_count} chunks: {items_to_encode} items to encode")

    if any(variant in variants_to_encode for variant in ["lm_cleaned_text_chunks", "lm_summary_chunks", "lm_q_and_a_chunks", "lm_q_and_a_for_q_only_chunks"]):
        rag_volume.commit()
        print("run_crawler_agent: volume committed")

    encode_file_paths = {
        "raw_chunks": RAW_CHUNKS_PATH,
        "manually_cleaned_chunks": MANUALLY_CLEANED_CHUNKS_PATH,
        "lm_cleaned_text_chunks": LM_CLEANED_TEXT_SUBCHUNKS_PATH,
        "lm_summary_chunks": LM_SUMMARY_SUBCHUNKS_PATH,
        "lm_q_and_a_chunks": LM_Q_AND_A_VALID_CHUNKS_PATH,
        "lm_q_and_a_for_q_only_chunks": LM_Q_AND_A_FOR_Q_ONLY_VALID_CHUNKS_PATH,
    }

    # create collection
    encoder_functions = {}
    try:
        create_collections = modal.Function.from_name("collection-handler", "create_collections")
    except Exception as e:
        print(f"run_crawler_agent: failed to find collection-handler.create_collections. Is it deployed? Error: {e}")
        return
    await create_collections.remote.aio(list(variants_to_encode.keys()), RECREATE_QDRANT_COLLECTIONS)

    for variant in variants_to_encode.keys():
        variant_config = ENCODE_VARIANTS[variant]
        encoder_variant_config = variant_config["encoders"]
        file_path = os.path.join(encode_file_paths[variant], f"{FILE_START}{encode_timestamp}.jsonl")
        if not os.path.exists(file_path):
            print(f"run_crawler_agent: missing encode file '{file_path}'")
            return
        with open(file_path, "r", encoding="utf-8") as f:
            total_records = sum(1 for _ in f)
        for encoder_name, encoder_config_for_variant in encoder_variant_config.items():
            encoder_config = ENCODERS[encoder_name]
            service_name = encoder_config["service"]
            function_name = encoder_config["function"]
            batch_size = encoder_config_for_variant["batch_size"]
            service_key = (service_name, function_name)
            if service_key not in encoder_functions:
                try:
                    encoder_functions[service_key] = modal.Function.from_name(service_name, function_name)
                except Exception as e:
                    print(f"run_crawler_agent: failed to find {service_name}.{function_name}. Is it deployed? Error: {e}")
                    return
            run_encoder = encoder_functions[service_key]
            tasks = [
                run_encoder.remote.aio(
                    variant,
                    encode_timestamp,
                    start,
                    batch_size,
                    encoder_name
                )
                for start in range(0, total_records, batch_size)
            ]
            await asyncio.gather(*tasks)

    print("run_crawler_agent: encoder dispatch complete")

    return
