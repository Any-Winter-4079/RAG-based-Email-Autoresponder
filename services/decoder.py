import modal

from config.general import modal_secret
from config.decoder import (
    image,
    GPU,
    TIMEOUT,
    SCALEDOWN_WINDOW,
    MIN_CONTAINERS,
    EMAIL_WRITER_PROFILE,
    THREAD_GROUPER_PROFILE,
    DATA_CLEANER_PROFILE
)

# Modal
app = modal.App("decoder")

@app.function(
        image=image,
        gpu=GPU,
        secrets=[modal_secret],
        timeout=TIMEOUT,
        scaledown_window=SCALEDOWN_WINDOW,
        min_containers=MIN_CONTAINERS
        )
def run_qwen3_lm_or_vlm(
    context,
    current_turn_input_text,
    model_path,
    is_vision_model,
    current_turn_image_in_bytes,
    system_prompt,
    max_new_tokens,
    temperature,
    top_p,
    top_k,
    use_flash_attention_2,
    enable_thinking,
    return_prompt_text=False,
    decoder_profile=EMAIL_WRITER_PROFILE
    ):
    import io
    import torch
    from PIL import Image
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, Qwen3VLForConditionalGeneration
    from helpers.decoder import remove_think_tokens, extract_message_content, extract_lm_cleaned_content, extract_thread_content
    from config.decoder import (
        NO_MESSAGE_OPENING_TAG,
        NO_MESSAGE_CLOSING_TAG,
        MESSAGE_OPENING_TAG,
        MESSAGE_CLOSING_TAG,
        ABSTRACT_OPENING_TAG,
        ABSTRACT_CLOSING_TAG,
        SUMMARY_OPENING_TAG,
        SUMMARY_CLOSING_TAG,
        CLEANED_TEXT_OPENING_TAG,
        CLEANED_TEXT_CLOSING_TAG,
        THREAD_OPENING_TAG,
        THREAD_CLOSING_TAG,
        THREAD_MESSAGE_OPENING_TAG,
        THREAD_MESSAGE_CLOSING_TAG,
        THREAD_FROM_OPENING_TAG,
        THREAD_FROM_CLOSING_TAG,
        THREAD_TO_OPENING_TAG,
        THREAD_TO_CLOSING_TAG,
        THREAD_SUBJECT_OPENING_TAG,
        THREAD_SUBJECT_CLOSING_TAG,
        THREAD_BODY_OPENING_TAG,
        THREAD_BODY_CLOSING_TAG,
        QUESTION_OPENING_TAG,
        QUESTION_CLOSING_TAG,
        ANSWER_OPENING_TAG,
        ANSWER_CLOSING_TAG
    )
    
    #######################################
    # Cache model (if not cached already) #
    #######################################
    if (not hasattr(run_qwen3_lm_or_vlm, "model") or
        getattr(run_qwen3_lm_or_vlm, "model_path", None) != model_path or
        getattr(run_qwen3_lm_or_vlm, "is_vision_model", None) != is_vision_model):

        print(f"run_qwen3_lm_or_vlm: loading model from {model_path} (is_vision_model: {is_vision_model})...")
        
        ##################################################################
        # Model could be LoRA adapter (to load on top of the base model) #
        ##################################################################
        try:
            run_qwen3_lm_or_vlm.model = AutoPeftModelForCausalLM.from_pretrained(
                model_path,
                dtype="auto",
                device_map="auto",
                attn_implementation="sdpa" if not use_flash_attention_2 else "flash_attention_2"
            )
        #################
        # Or base model #
        #################
        except Exception:
            if is_vision_model:
                run_qwen3_lm_or_vlm.model = Qwen3VLForConditionalGeneration.from_pretrained(
                    model_path,
                    dtype="auto",
                    device_map="auto",
                    attn_implementation="sdpa" if not use_flash_attention_2 else "flash_attention_2"
                )
            else:
                run_qwen3_lm_or_vlm.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    dtype="auto",
                    device_map="auto",
                    attn_implementation="sdpa" if not use_flash_attention_2 else "flash_attention_2"
                )
        if is_vision_model:
            run_qwen3_lm_or_vlm.processor = AutoProcessor.from_pretrained(model_path)
        else:
            run_qwen3_lm_or_vlm.tokenizer = AutoTokenizer.from_pretrained(model_path)
        run_qwen3_lm_or_vlm.model_path = model_path
        run_qwen3_lm_or_vlm.is_vision_model = is_vision_model
        
        print("run_qwen3_lm_or_vlm: model and processor loaded and cached")

    ################################################
    # Use the cached model and processor/tokenizer #
    ################################################
    model = run_qwen3_lm_or_vlm.model
    processor = run_qwen3_lm_or_vlm.processor if is_vision_model else run_qwen3_lm_or_vlm.tokenizer

    messages = []

    def form_vlm_input_turn_content(input_text, input_image_in_bytes):
        content = [{"type": "text", "text": input_text}]
        if input_image_in_bytes:
            context_image_pil = Image.open(io.BytesIO(input_image_in_bytes))
        else:
            context_image_pil = None
        if context_image_pil:
            content.insert(0, {"type": "image", "image": context_image_pil})
        return content
    
    #####################
    # Add system prompt #
    #####################
    messages.append({
        "role": "system", 
        "content": [{"type": "text", "text": system_prompt}] if is_vision_model else system_prompt
    })

    #######################################
    # Add context (both input and output) #
    #######################################
    for context_turn in context:
        if is_vision_model:
            context_input_turn_content = form_vlm_input_turn_content(context_turn["input_text"], context_turn.get("input_image"))
        messages.append({
            "role": "user",
            "content": context_input_turn_content if is_vision_model else context_turn["input_text"]
        })
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": context_turn["output_text"]}] if is_vision_model else context_turn["output_text"]
        })
    
    ##########################
    # Add current turn input #
    ##########################
    if is_vision_model:
        current_input_turn_content = form_vlm_input_turn_content(current_turn_input_text, current_turn_image_in_bytes)
    messages.append({
        "role": "user",
        "content": current_input_turn_content if is_vision_model else current_turn_input_text
    })

    ###################################################################
    # Store entire prompt before tokenization (if return_prompt_text) #
    ###################################################################
    if return_prompt_text:
        if is_vision_model:
            prompt_text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking
            )
    
    ##################################################################
    # Apply chat template to prompt, tokenize and move ids to device #
    ##################################################################
    if is_vision_model:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
    else:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=enable_thinking
        )
    inputs = inputs.to(model.device)

    ##################################
    # Generate response as token ids #
    ##################################
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            use_cache=True, 
            temperature=temperature, 
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
        )

    ############################
    # Decode token ids to text #
    ############################
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    output_text = output_text[0] if output_text else None

    print(f"{output_text}\n\n")

    ########################################
    # Remove think tokens if thinking mode #
    ########################################
    if enable_thinking:
        output_text = remove_think_tokens(output_text)    

    ##########################################################
    # Extract reply if LM think it has enough info to answer #
    ##########################################################
    if decoder_profile == EMAIL_WRITER_PROFILE:
        output_text = extract_message_content(
            output_text,
            NO_MESSAGE_OPENING_TAG,
            NO_MESSAGE_CLOSING_TAG,
            MESSAGE_OPENING_TAG,
            MESSAGE_CLOSING_TAG
        )
    elif decoder_profile == THREAD_GROUPER_PROFILE:
        output_text = extract_thread_content(
            output_text,
            THREAD_OPENING_TAG,
            THREAD_CLOSING_TAG,
            THREAD_MESSAGE_OPENING_TAG,
            THREAD_MESSAGE_CLOSING_TAG,
            THREAD_FROM_OPENING_TAG,
            THREAD_FROM_CLOSING_TAG,
            THREAD_TO_OPENING_TAG,
            THREAD_TO_CLOSING_TAG,
            THREAD_SUBJECT_OPENING_TAG,
            THREAD_SUBJECT_CLOSING_TAG,
            THREAD_BODY_OPENING_TAG,
            THREAD_BODY_CLOSING_TAG
        )
    elif decoder_profile == DATA_CLEANER_PROFILE:
        output_text = extract_lm_cleaned_content(
            output_text,
            ABSTRACT_OPENING_TAG,
            ABSTRACT_CLOSING_TAG,
            SUMMARY_OPENING_TAG,
            SUMMARY_CLOSING_TAG,
            CLEANED_TEXT_OPENING_TAG,
            CLEANED_TEXT_CLOSING_TAG,
            QUESTION_OPENING_TAG,
            QUESTION_CLOSING_TAG,
            ANSWER_OPENING_TAG,
            ANSWER_CLOSING_TAG
        )
    else:
        print(f"run_qwen3_lm_or_vlm: unknown decoder_profile '{decoder_profile}'")
        output_text = None
    
    return (output_text, prompt_text) if return_prompt_text else (output_text, None)
