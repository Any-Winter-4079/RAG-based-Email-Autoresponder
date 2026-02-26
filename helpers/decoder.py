##########################################
# Helper 1: Extract content between tags #
##########################################
def extract_matched_content(response, opening_tag, closing_tag):
    import re

    if response is None:
        return None

    # match text within tags (we use () to return content inside the match)
    # . -> any character
    # * -> zero or more times
    # ? -> stop as soon as possible (at the 1st closing tag match, vs at the last one)
    matches = re.findall(f"{opening_tag}(.*?){closing_tag}", response, flags=re.DOTALL)
    return [match.strip() for match in matches]

##############################################
# Helper 2: Remove <think>...</think> tokens #
##############################################
def remove_think_tokens(response):
    import re

    # if no response, do not reply
    if response is None:
        return None
    
    # if response, remove <think>...</think> tokens
    return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)

###################################################
# Helper 3: Extract <message>...</message> tokens #
###################################################
def extract_message_content(
        response,
        no_message_opening_tag,
        no_message_closing_tag,
        message_opening_tag,
        message_closing_tag
        ):
    # if no response, do not reply
    if response is None:
        return None

    # if LM thinks it does not have enough info to answer, we do not reply
    no_message = extract_matched_content(response, no_message_opening_tag, no_message_closing_tag)
    if no_message:
        return None
    
    # if LM thinks it has enough info to answer, use the message
    message = extract_matched_content(response, message_opening_tag, message_closing_tag)
    if message:
        # return message (there should be a single message per response)
        return message[0]

    # if neither tag, we do not reply (LM did not follow intructions or instructions were not correct)
    return None

#################################################################################
# Helper 4: Extract <from>...</from>, <to>...</to>, <subject>...</subject>      #
#                   <body>...</body>                                            # 
#################################################################################
def extract_thread_content(
        response,
        thread_opening_tag,
        thread_closing_tag,
        message_opening_tag,
        message_closing_tag,
        from_opening_tag,
        from_closing_tag,
        to_opening_tag,
        to_closing_tag,
        subject_opening_tag,
        subject_closing_tag,
        body_opening_tag,
        body_closing_tag
        ):
    # if no response, do not return content
    if response is None:
        return None

    threads = extract_matched_content(response, thread_opening_tag, thread_closing_tag)
    if not threads:
        return None

    parsed_threads = []
    for thread_text in threads:
        messages = extract_matched_content(thread_text, message_opening_tag, message_closing_tag)
        parsed_messages = []
        for message_text in messages:
            from_value = extract_matched_content(message_text, from_opening_tag, from_closing_tag)
            to_value = extract_matched_content(message_text, to_opening_tag, to_closing_tag)
            subject_value = extract_matched_content(message_text, subject_opening_tag, subject_closing_tag)
            body_value = extract_matched_content(message_text, body_opening_tag, body_closing_tag)

            parsed_messages.append({
                "from": from_value[0] if from_value else None,
                "to": to_value[0] if to_value else None,
                "subject": subject_value[0] if subject_value else None,
                "body": body_value[0] if body_value else None
            })
        parsed_threads.append({"messages": parsed_messages})

    return parsed_threads

#################################################################################
# Helper 5: Extract <abstract>...</abstract>, <summary>...</summary>,           # 
#                   <cleanedtext>...</cleanedtext>, <question>...</question>    #
#                   <answer>...</answer>                                        #
#################################################################################
def extract_lm_cleaned_content(
        response,
        abstract_opening_tag,
        abstract_closing_tag,
        summary_opening_tag,
        summary_closing_tag,
        cleanedtext_opening_tag,
        cleanedtext_closing_tag,
        question_opening_tag,
        question_closing_tag,
        answer_opening_tag,
        answer_closing_tag
        ):

    # if no response, do not return content
    if response is None:
        return None
    
    # extract abstract, summary, cleanedtext, questions, answers
    abstract = extract_matched_content(response, abstract_opening_tag, abstract_closing_tag)
    if abstract:
        # there should be a single abstract per response
        abstract = abstract[0]
    summary = extract_matched_content(response, summary_opening_tag, summary_closing_tag)
    if summary:
        # there should be a single summary per response
        summary = summary[0]
    cleanedtext = extract_matched_content(response, cleanedtext_opening_tag, cleanedtext_closing_tag)
    if cleanedtext:
        # there should be a single cleanedtext per response
        cleanedtext = cleanedtext[0]
    questions = extract_matched_content(response, question_opening_tag, question_closing_tag)
    answers = extract_matched_content(response, answer_opening_tag, answer_closing_tag)

    return [abstract, summary, cleanedtext, questions, answers]

##########################
# Helper 5: Count tokens #
##########################
def count_tokens(tokenizer, text):
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception as e:
        print(f"count_tokens: error counting tokens: {e}")
        return 0

################################
# Helper 6: Truncate to tokens #
################################
def truncate_to_tokens(tokenizer, text, max_tokens):
    try:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
    except Exception as e:
        print(f"truncate_to_tokens: error encoding text: {e}")
        return None
    if len(token_ids) <= max_tokens:
        return text
    ellipsis = "..."
    ellipsis_tokens = len(tokenizer.encode(ellipsis, add_special_tokens=False))
    if max_tokens <= ellipsis_tokens:
        return ellipsis
    truncated = tokenizer.decode(
        token_ids[:max_tokens - ellipsis_tokens],
        skip_special_tokens=True
    ).strip()
    return f"{truncated}{ellipsis}" if truncated else ellipsis
