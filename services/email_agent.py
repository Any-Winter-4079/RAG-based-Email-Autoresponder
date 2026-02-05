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
    from config.email_agent import (
        MAX_EMAILS,
        UNREAD_ONLY,
        LEAVE_UNREAD,
        LAST_N_DAYS,
        SEND_TO_SELF,
        SAVE_AS_DRAFT,
        DRAFTS_FOLDER
    )
    from config.decoder import MODEL_PROFILES
    from helpers.email_agent import (
        transform_env_csv_into_list,
        read_latest_emails,
        format_response_quoting_original_body,
        send_emails,
        save_drafts,
        mark_emails_as_read
    )
    import os

    # required env vars
    imap_server = os.getenv("IMAP_SERVER")
    imap_port_str = os.getenv("IMAP_PORT")
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port_str = os.getenv("SMTP_PORT")
    imap_email = os.getenv("IMAP_EMAIL")
    smtp_email = os.getenv("SMTP_EMAIL")
    password = os.getenv("PASSWORD")
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
    
    # find decoder service
    try:
        run_qwen3_lm_or_vlm = modal.Function.from_name("decoder", "run_qwen3_lm_or_vlm")
    except Exception as e:
        print(f"run_email_agent: failed to find decoder service. Is it deployed? Error: {e}")
        return

    # read latest emails
    emails = read_latest_emails(
        max_emails=MAX_EMAILS,
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

    # select decoder configuration for emails
    model_config = MODEL_PROFILES["email_writer"].copy()

    # pop (and save) "prompt_template" (run_qwen3_lm_or_vlm would not expect it as model_config)
    prompt_template = model_config.pop("prompt_template")

    reply_bodies, original_subjects, recipient_emails, processed_email_ids = [], [], [], []

    # for each email:
    for email in emails:
        # get subject and original body and sender
        original_subject = email["subject"]
        original_body = email["message_body"]
        original_sender = email["from"]
        print(f"run_email_agent: generating reply for '{original_subject}' from {original_sender}")

        # construct prompt
        try:
            prompt = prompt_template.format(
                my_name=my_name,
                my_description=my_description,
                sender=original_sender,
                body=original_body,
                rag_context=""
            )
        except KeyError as e:
            print(f"run_email_agent: error formatting prompt template: {e}")
            continue

        # run decoder (without "template" in model_config)
        try:
            proposed_reply, prompt_text = run_qwen3_lm_or_vlm.remote(
                context=[],
                current_turn_input_text=prompt,
                **model_config
            )
        except Exception as e:
            print(f"run_email_agent: decoder generation failed: {e}")
            continue

        if MODEL_PROFILES["email_writer"]["return_prompt_text"]:
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
