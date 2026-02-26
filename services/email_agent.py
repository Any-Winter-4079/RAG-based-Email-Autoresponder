import modal
from config.general import modal_secret
from config.email_agent import (
    image,
    MODAL_TIMEOUT,
    EMAIL_HOUR,
    EMAIL_MINUTE
)

# Modal
app = modal.App("email-agent")

@app.function(
        image=image,
        # with Cron format "Minute Hour Day Month DayOfWeek":
        schedule=modal.Cron(f"{EMAIL_MINUTE} {EMAIL_HOUR} * * *", timezone="Europe/Madrid"),
        secrets=[modal_secret],
        timeout=MODAL_TIMEOUT,
        # https://modal.com/docs/guide/region-selection: price multiplier = 1.25x
        region="eu-south-2" # "spaincentral": AZR Madrid / "eu-south-2": AWS Spain
)
def run_email_agent():
    import os
    from datetime import datetime
    from transformers import AutoTokenizer
    from helpers.decoder import count_tokens, truncate_to_tokens

    from helpers.data import (
        assign_thread_ids_by_subject_and_participant_overlap_for_production,
        get_unquoted_text
    )
    from config.decoder import MODEL_PROFILES, EMAIL_WRITER_PROFILE
    from config.email_agent import (
        MAX_EMAILS,
        CONTEXT_EMAILS_PER_FOLDER,
        MAX_UNQUOTED_TOKENS_PER_CURRENT_EMAIL,
        MAX_UNQUOTED_TOKENS_PER_CONTEXT_EMAIL,
        MAX_QUOTED_TOKENS_PER_CURRENT_EMAIL,
        MAX_QUOTED_TOKENS_PER_CONTEXT_EMAIL,
        INBOX_FOLDER,
        SENT_FOLDER,
        UNREAD_ONLY,
        LEAVE_UNREAD,
        LAST_N_DAYS,
        SEND_TO_SELF,
        SAVE_AS_DRAFT,
        DRAFTS_FOLDER
    )
    from helpers.email_agent import (
        transform_env_csv_into_list,
        read_latest_emails,
        format_response_quoting_original_body,
        compact_email_body_for_decoder,
        send_emails,
        save_drafts,
        mark_emails_as_read
    )

    # required env vars
    imap_server = os.getenv("IMAP_SERVER")
    imap_port_str = os.getenv("IMAP_PORT")
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port_str = os.getenv("SMTP_PORT")
    imap_email = os.getenv("IMAP_EMAIL")
    smtp_email = os.getenv("SMTP_EMAIL")
    password = os.getenv("PASSWORD")
    my_email_addresses = os.getenv("MY_EMAIL_ADDRESSES")
    my_name = os.getenv("MY_NAME")
    my_description = os.getenv("MY_DESCRIPTION")

    missing_env_vars = []
    if not imap_server: missing_env_vars.append("IMAP_SERVER")
    if not imap_port_str: missing_env_vars.append("IMAP_PORT")
    if not smtp_server: missing_env_vars.append("SMTP_SERVER")
    if not smtp_port_str: missing_env_vars.append("SMTP_PORT")
    if not imap_email: missing_env_vars.append("IMAP_EMAIL")
    if not smtp_email: missing_env_vars.append("SMTP_EMAIL")
    if not password: missing_env_vars.append("PASSWORD")
    if not my_email_addresses: missing_env_vars.append("MY_EMAIL_ADDRESSES")
    if not my_name: missing_env_vars.append("MY_NAME")
    if not my_description: missing_env_vars.append("MY_DESCRIPTION")

    if missing_env_vars:
        print(f"run_email_agent: missing required environment variables: {', '.join(missing_env_vars)}")
        return

    # optional env vars
    blacklisted_emails = transform_env_csv_into_list(os.getenv("BLACKLISTED_EMAILS", ""))
    blacklisted_emails.append(smtp_email.lower()) # add self
    blacklisted_emails = list(set(blacklisted_emails))
    blacklisted_domains = transform_env_csv_into_list(os.getenv("BLACKLISTED_DOMAINS", ""))
    my_email_addresses = transform_env_csv_into_list(my_email_addresses)
    if not my_email_addresses:
        print("run_email_agent: MY_EMAIL_ADDRESSES must include at least one email")
        return
    
    # find decoder service
    try:
        run_qwen3_lm_or_vlm = modal.Function.from_name("decoder", "run_qwen3_lm_or_vlm")
    except Exception as e:
        print(f"run_email_agent: failed to find decoder service. Is it deployed? Error: {e}")
        return

    # read latest emails
    emails = read_latest_emails(
        max_emails=MAX_EMAILS,
        folder=INBOX_FOLDER,
        last_n_days=LAST_N_DAYS,
        imap_email=imap_email,
        password=password,
        unread_only=UNREAD_ONLY,
        imap_server=imap_server,
        imap_port=int(imap_port_str),
        blacklisted_emails=blacklisted_emails,
        blacklisted_domains=blacklisted_domains
    )
    if not emails:
        print("run_email_agent: no new emails to process")
        return
    else:
        print(f"run_email_agent: {len(emails)} new emails to process")

    # load additional context emails from inbox and sent folders
    context_inbox_emails = read_latest_emails(
        max_emails=CONTEXT_EMAILS_PER_FOLDER,
        folder=INBOX_FOLDER,
        last_n_days=LAST_N_DAYS,
        imap_email=imap_email,
        password=password,
        unread_only=False,
        imap_server=imap_server,
        imap_port=int(imap_port_str),
        blacklisted_emails=blacklisted_emails,
        blacklisted_domains=blacklisted_domains
    )
    # reverse to match config/decoder's oldest to newest
    context_inbox_emails = list(reversed(context_inbox_emails))
    context_sent_emails = read_latest_emails(
        max_emails=CONTEXT_EMAILS_PER_FOLDER,
        folder=SENT_FOLDER,
        last_n_days=LAST_N_DAYS,
        imap_email=imap_email,
        password=password,
        unread_only=False,
        imap_server=imap_server,
        imap_port=int(imap_port_str),
        blacklisted_emails=blacklisted_emails,
        blacklisted_domains=blacklisted_domains
    )
    # reverse to match config/decoder's oldest to newest
    context_sent_emails = list(reversed(context_sent_emails))
    print(
        "run_email_agent: loaded "
        f"{len(context_inbox_emails)} inbox context emails and "
        f"{len(context_sent_emails)} sent context emails"
    )

    # total context emails are the "set" of inbox emails, sent emails, and emails to answer
    # because we could have (rare albeit possible):
    # MAX_EMAILS = 1 (email 20)
    # CONTEXT_EMAILS_PER_FOLDER = 20 (emails 0-19)
    # normalized_subject(email_20) != normalized subjects of emails 0-19 so it becomes contextless
    combined_context_emails = context_inbox_emails + context_sent_emails + list(reversed(emails))
    # we require inbox+sent+emails (to reply to) to form the thread ids, despite later
    # separating again 
    unique_context_emails = []
    seen_context_ids = set()
    for email in combined_context_emails:
        email_id = email.get("id")
        if email_id in seen_context_ids:
            continue
        seen_context_ids.add(email_id)
        unique_context_emails.append(email)

    # add thread ids (by normalized subject + participant overlap)
    combined_context_emails = assign_thread_ids_by_subject_and_participant_overlap_for_production(
        unique_context_emails,
        my_email_addresses
    )

    # select decoder configuration for email writing
    email_writer_profile_config = MODEL_PROFILES[EMAIL_WRITER_PROFILE].copy()

    # pop (and save) "prompt_template" and "max_context_tokens" (run_qwen3_lm_or_vlm would not expect them)
    prompt_template = email_writer_profile_config.pop("prompt_template")
    max_context_tokens = email_writer_profile_config.pop("max_context_tokens")

    # calculate max new tokens budget based on thinking mode, max context tokens
    enable_thinking = email_writer_profile_config.get("enable_thinking", False)
    input_token_budget = max_context_tokens // 3 if enable_thinking else max_context_tokens // 2
    email_writer_profile_config["max_new_tokens"] = max_context_tokens - input_token_budget

    # get decoder tokenizer 
    decoder_path = email_writer_profile_config["model_path"]
    decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_path, trust_remote_code=True)

    # map email id to thread id and thread id to emails
    email_id_to_thread_id = {}
    thread_id_to_emails = {}
    for thread_email in combined_context_emails:
        thread_id = thread_email.get("threadID")
        if thread_id is None:
            continue
        email_id = thread_email.get("id")
        if email_id is not None:
            email_id_to_thread_id[email_id] = thread_id
        if thread_id not in thread_id_to_emails:
            thread_id_to_emails[thread_id] = []
        thread_id_to_emails[thread_id].append(thread_email)

    reply_bodies, original_subjects, recipient_emails, processed_email_ids = [], [], [], []

    # for each email:
    for email in emails:
        # get subject and original body and sender
        original_subject = email.get("subject")
        original_body = email.get("message_body")
        original_sender = email.get("from")
        # if message is incomplete, skip
        if not original_subject or not original_body or not original_sender:
            print(
                "run_email_agent: skipping email because missing data "
                f"(subject={original_subject!r}, body={original_body!r}, sender={original_sender!r})"
            )
            continue
        # if we are the author (message to self), skip
        # if any(email in original_sender.lower() for email in my_email_addresses):
        #     print(f"run_email_agent: skipping email because sender is one of {', '.join(my_email_addresses)}")
        #     continue
        print(f"run_email_agent: generating reply for '{original_subject}' from {original_sender}")

        # if current email already includes quoted history, skip thread context
        _, original_quoted_body = get_unquoted_text(original_body, return_quoted=True)
        skip_thread_context = bool((original_quoted_body or "").strip())

        # get thread context emails for this email
        email_id = email.get("id")
        thread_id = email_id_to_thread_id.get(email_id)
        thread_context_emails = [
            context_email
            for context_email in thread_id_to_emails.get(thread_id, [])
            if context_email.get("id") != email_id
        ]
        # sort from most recent/latest to original (high to low datetime)
        thread_context_emails = sorted(
            thread_context_emails,
            key=lambda context_email: context_email.get("date") or datetime.min,
            reverse=True
        )
        # get original email (that started the thread) and the rest (latest to oldest)
        if thread_context_emails:
            first_email = thread_context_emails[-1]
            other_emails_latest_first = thread_context_emails[:-1]
        else:
            first_email = None
            other_emails_latest_first = []

        # get unquoted/quoted body text for the current email and truncate them
        original_body_compacted = compact_email_body_for_decoder(
            decoder_tokenizer,
            original_body,
            MAX_UNQUOTED_TOKENS_PER_CURRENT_EMAIL,
            MAX_QUOTED_TOKENS_PER_CURRENT_EMAIL,
            "[text omitted: body missing]",
            unquoted_fail_placeholder=None,
            quoted_fail_placeholder="[quoted text omitted: tokenization failed]",
            log_prefix="run_email_agent: current email"
        )
        if original_body_compacted is None:
            continue

        # make sure base prompt fits, clamp email if needed
        try:
            base_prompt = prompt_template.format(
                my_name=my_name,
                my_description=my_description,
                subject=original_subject,
                sender=original_sender,
                body="",
                thread_context="",
                rag_context=""
            )
        except KeyError as e:
            print(f"run_email_agent: error formatting email writer prompt template (without body): {e}")
            continue
        base_prompt_tokens = count_tokens(decoder_tokenizer, base_prompt)
        available_body_tokens = input_token_budget - base_prompt_tokens
        if available_body_tokens <= 0:
            print("run_email_agent: skipping email because no token budget left for body")
            continue
        body_tokens = count_tokens(decoder_tokenizer, original_body_compacted)
        if body_tokens > available_body_tokens:
            original_body_compacted = truncate_to_tokens(
                decoder_tokenizer,
                original_body_compacted,
                available_body_tokens
            )
            if original_body_compacted is None:
                print("run_email_agent: skipping email because body truncation failed")
                continue

        # build prompt with email body
        try:
            prompt = prompt_template.format(
                my_name=my_name,
                my_description=my_description,
                subject=original_subject,
                sender=original_sender,
                body=original_body_compacted,
                thread_context="",
                rag_context=""
            )
        except KeyError as e:
            print(f"run_email_agent: error formatting email writer prompt template (with body): {e}")
            continue
        prompt_tokens = count_tokens(decoder_tokenizer, prompt)
        if prompt_tokens > input_token_budget:
            print("run_email_agent: skipping email because base prompt exceeds input token budget")
            continue

        # add context emails (if tokens fit) starting with first_email, then most recent to oldest, 
        # if email to reply to doesn't already contain quoted text
        # NOTE: this is a simplification, given email could have quoted text but not be the full
        # thread. It can also happen CONTEXT_EMAILS_PER_FOLDER aren't enough to reconstruct as
        # much as the quoted text in the email to answer, so we can't rely on it either. We'd have
        # to semantically or format-aware check if quoted text in the email to reply to contains
        # the full thread or they complement each other, but this is a fair approximation for
        # a 1st prototype
        thread_context = ""
        message_separator = "\n[END MESSAGE]\n"
        if first_email and not skip_thread_context:
            first_email_body = compact_email_body_for_decoder(
                decoder_tokenizer,
                first_email.get("message_body"),
                MAX_UNQUOTED_TOKENS_PER_CONTEXT_EMAIL,
                MAX_QUOTED_TOKENS_PER_CONTEXT_EMAIL,
                "[text omitted: body missing]",
                unquoted_fail_placeholder="[text omitted: tokenization failed]",
                quoted_fail_placeholder="[quoted text omitted: tokenization failed]",
                log_prefix="run_email_agent: context email"
            )
            first_email_from = (first_email.get("from") or "").strip()
            first_email_to = (first_email.get("to") or "").strip()
            first_email_subject = (first_email.get("subject") or "").strip()
            first_email_date = first_email.get("date")
            first_email_date_text = str(first_email_date) if first_email_date else ""
            block_header = (
                "From: " + first_email_from + "\n"
                "To: " + first_email_to + "\n"
                "Date: " + first_email_date_text + "\n"
                "Subject: " + first_email_subject + "\n"
                "Body:\n"
            )
            # form thread context with first email
            thread_context = f"{block_header}{first_email_body}{message_separator}".strip()
            # if first email fits, add latest emails until budget is reached
            thread_context_tokens = count_tokens(decoder_tokenizer, thread_context)
            if prompt_tokens + thread_context_tokens <= input_token_budget:
                for other_email in other_emails_latest_first:
                    other_email_body = other_email.get("message_body")
                    if not other_email_body:
                        print("run_email_agent: skipping context email because body is missing")
                        continue
                    other_email_body = compact_email_body_for_decoder(
                        decoder_tokenizer,
                        other_email_body,
                        MAX_UNQUOTED_TOKENS_PER_CONTEXT_EMAIL,
                        MAX_QUOTED_TOKENS_PER_CONTEXT_EMAIL,
                        "[text omitted: body missing]",
                        unquoted_fail_placeholder="[text omitted: tokenization failed]",
                        quoted_fail_placeholder="[quoted text omitted: tokenization failed]",
                        log_prefix="run_email_agent: context email"
                    )
                    other_email_from = (other_email.get("from") or "").strip()
                    other_email_to = (other_email.get("to") or "").strip()
                    other_email_subject = (other_email.get("subject") or "").strip()
                    other_email_date = other_email.get("date")
                    other_email_date_text = str(other_email_date) if other_email_date else ""
                    other_block_header = (
                        "From: " + other_email_from + "\n"
                        "To: " + other_email_to + "\n"
                        "Date: " + other_email_date_text + "\n"
                        "Subject: " + other_email_subject + "\n"
                        "Body:\n"
                    )
                    candidate_context = f"{thread_context}\n{other_block_header}{other_email_body}{message_separator}".strip()
                    candidate_tokens = count_tokens(decoder_tokenizer, candidate_context)
                    if prompt_tokens + candidate_tokens > input_token_budget:
                        break
                    thread_context = candidate_context
            else:
                print("run_email_agent: skipping first email because base && first email prompt exceeds input token budget")
                thread_context = ""

        if not thread_context:
            thread_context = (
                "(conversation already quoted in the current email)"
                if skip_thread_context
                else "(no prior messages found)"
            )

        # construct prompt
        try:
            prompt = prompt_template.format(
                my_name=my_name,
                my_description=my_description,
                subject=original_subject,
                sender=original_sender,
                body=original_body_compacted,
                thread_context=thread_context,
                rag_context=""
            )
        except KeyError as e:
            print(f"run_email_agent: error formatting email writer prompt template (with body and context): {e}")
            continue

        # run decoder (without "template" in email_writer_profile_config)
        try:
            proposed_reply, prompt_text = run_qwen3_lm_or_vlm.remote(
                context=[],
                current_turn_input_text=prompt,
                current_turn_image_in_bytes=None,
                **email_writer_profile_config
            )
        except Exception as e:
            print(f"run_email_agent: decoder generation failed: {e}")
            continue

        if MODEL_PROFILES[EMAIL_WRITER_PROFILE]["return_prompt_text"]:
            print(f"{prompt_text}\n\n")

        # if LM thinks it does not have enough info to answer or fails to use <message>...</message>, skip email reply
        if proposed_reply is None:
            continue

        # format reply quoting original inquiry and append it to reply bodies list
        reply_body = format_response_quoting_original_body(proposed_reply, original_body)
        reply_bodies.append(reply_body)
        processed_email_ids.append(email["id"])

        # append subject to subjects list
        original_subjects.append(original_subject)
        
        # set recipient email and append it to recipient emails list
        recipient_email = smtp_email if (SEND_TO_SELF and not SAVE_AS_DRAFT) else original_sender
        recipient_emails.append(recipient_email)

    # save drafts
    if SAVE_AS_DRAFT:
        success, error = save_drafts(
            reply_bodies=reply_bodies,
            original_subjects=original_subjects,
            imap_email=imap_email,
            smtp_email=smtp_email,
            recipient_emails=recipient_emails,
            password=password,
            imap_server=imap_server,
            imap_port=int(imap_port_str),
            drafts_folder=DRAFTS_FOLDER
        )
        action_performed = f"saved {len(original_subjects)} drafts"
    # or send replies
    else:
        success, error = send_emails(
            reply_bodies=reply_bodies,
            original_subjects=original_subjects,
            smtp_email=smtp_email,
            recipient_emails=recipient_emails,
            password=password,
            smtp_server=smtp_server,
            smtp_port=int(smtp_port_str)
        )
        action_performed = f"sent {len(original_subjects)} emails"

    if success:
        print(f"run_email_agent: {action_performed} successfully")
        if not LEAVE_UNREAD and processed_email_ids:
            mark_success, mark_error = mark_emails_as_read(
                email_ids=processed_email_ids,
                imap_email=imap_email,
                password=password,
                imap_server=imap_server,
                imap_port=int(imap_port_str)
            )
            if mark_success:
                print(f"run_email_agent: marked {len(processed_email_ids)} emails as read")
            else:
                print(f"run_email_agent: emails were replied to, but marking as read failed: {mark_error}")
    else:
        print(f"run_email_agent: error: {error}")
