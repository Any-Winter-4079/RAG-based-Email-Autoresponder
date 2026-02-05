import modal
from config.general import modal_secret
from config.crawler_agent import (
    image,
    MODAL_TIMEOUT,
    CRAWL_MINUTES,
    CRAWL_HOUR,
    CRAWL_DAY,
    CRAWL_MONTH,
    VOLUME_PATH,
    rag_volume
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
    from config.decoder import MODEL_PROFILES as DECODER_MODEL_PROFILES
    from config.encoder import EMBEDDING_MODEL, MODEL_PROFILES as ENCODER_MODEL_PROFILES
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
        FILE_START,
        RAW_PATH,
        MANUALLY_CLEANED_PATH,
        RAW_CHUNKS_PATH,
        MANUALLY_CLEANED_CHUNKS_PATH,
        LM_CLEANED_TEXT_CHUNKS_PATH,
        LM_ABSTRACT_CHUNKS_PATH,
        LM_SUMMARY_CHUNKS_PATH,
        LM_Q_AND_A_CHUNKS_PATH
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
                model_config = DECODER_MODEL_PROFILES["data_cleaner"].copy()

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
                        **model_config,
                        is_email_writer=False
                    )
                except Exception as e:
                    print(f"run_crawler_agent: decoder generation failed: {e}")
                    continue

                if DECODER_MODEL_PROFILES["data_cleaner"]["return_prompt_text"]:
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

    # load encoder tokenizer to count page tokens (for embedding)
    encoder_path = ENCODER_MODEL_PROFILES[EMBEDDING_MODEL]["model_path"]
    encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_path, trust_remote_code=True)

    # create text splitter using the encoder tokenizer and recommended max size
    embedding_chunk_size = ENCODER_MODEL_PROFILES[EMBEDDING_MODEL]["max_recommended_input_size"] 
    embedding_splitter = SentenceSplitter(
        chunk_size=embedding_chunk_size,
        chunk_overlap=CHUNK_OVERLAP,
        tokenizer=lambda text: encoder_tokenizer.encode(text, add_special_tokens=False)
    )

    reuse_text_file = ""
    reuse_q_and_a_file = ""
    current_year = datetime.datetime.now().year

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
            candidate_text_file = os.path.join(LM_CLEANED_TEXT_CHUNKS_PATH, f"{FILE_START}{reuse_timestamp}.jsonl")
            candidate_q_and_a_file = os.path.join(LM_Q_AND_A_CHUNKS_PATH, f"{FILE_START}{reuse_timestamp}.jsonl")

            # if the timestamp is correct but the file does not exist, error out and return
            if not os.path.exists(candidate_text_file):
                print(f"run_crawler_agent: '{candidate_text_file}' not found. Are you sure it exists on Modal?")
                return

            # if the files exist and we are fine reusing them even past >1 year, or they are <1 year old, use the files
            if REUSE_CRAWL_PAST_CURRENT_YEAR or timestamp_year == current_year:
                reuse_text_file = candidate_text_file
                if os.path.exists(candidate_q_and_a_file):
                    reuse_q_and_a_file = candidate_q_and_a_file
                # unless not all files are available (then error out and return)
                else:
                    print(f"run_crawler_agent: '{candidate_q_and_a_file}' not found. Are you sure it exists on Modal?")
                    return
                print(f"run_crawler_agent: reusing crawl {os.path.basename(candidate_text_file)}")
            # else, discard, crawl and use a new version
            else:
                print(f"run_crawler_agent: REUSE_TIMESTAMP '{reuse_timestamp}' is outside current year ({current_year}): crawling fresh data")

        # else (we want to reuse but no timestamp is set):
        else:
            year_filter = "*" if REUSE_CRAWL_PAST_CURRENT_YEAR else f"{current_year}*"
            existing_text_files = glob.glob(os.path.join(LM_CLEANED_TEXT_CHUNKS_PATH, f"{FILE_START}{year_filter}.jsonl"))
            # if no file exists (or no file within the current year), error out and return
            if not existing_text_files:
                print(f"run_crawler_agent: REUSE_CRAWL is enabled but no lm_cleaned file was found. Are you sure it exists on Modal?")
                return

            # if they exist, use them
            reuse_text_file = max(existing_text_files, key=os.path.getctime)
            matching_q_and_a_file = os.path.join(LM_Q_AND_A_CHUNKS_PATH, os.path.basename(reuse_text_file))
            if os.path.exists(matching_q_and_a_file):
                reuse_q_and_a_file = matching_q_and_a_file
            # unless not all files are available (then error out and return)
            else:
                print(f"run_crawler_agent: '{matching_q_and_a_file}' not found. Are you sure it exists on Modal?")
                return
            print(f"run_crawler_agent: reusing latest eligible crawl: {os.path.basename(reuse_text_file)}")

    # if a reusable crawl is available, load it (from the volume)
    if reuse_text_file:
        with open(reuse_text_file, "r", encoding="utf-8") as f:
            lm_cleaned_text_chunks = [json.loads(line) for line in f]
        with open(reuse_q_and_a_file, "r", encoding="utf-8") as f:
            lm_q_and_a_chunks = [json.loads(line) for line in f]
    
    # else, fetch, clean up, and store the data on the volume
    else:
        try:
            # load decoder tokenizer to count page tokens (for generation)
            decoder_path = DECODER_MODEL_PROFILES["email_writer"]["model_path"]
            decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_path, trust_remote_code=True)

            # create text splitter using the decoder tokenizer and a max chunk size
            lm_chunk_size = DECODER_MODEL_PROFILES["data_cleaner"]["max_chunk_size"] 
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

                        header = f"{label} CHUNK {i+1} [Source: {chunk['url']}] | {token_info}"
                        f_txt.write(f"\n{separator}\n{header}\n{separator}\n{content}\n")

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

    # encode
    print("run_crawler_agent: encode to begin shortly")

    return
