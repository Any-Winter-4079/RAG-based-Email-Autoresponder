################################################
# Helper 1: Transform env csv into Python list #
################################################
def transform_env_csv_into_list(env_list):
    # e.g., handle@domain1.com,handle@domain2.com -> ["handle@domain1.com", "handle@domain2.com"]
    return [item.strip().lower() for item in env_list.split(",") if item.strip()]

###################################################################
# Helper 2: Check whether email address or domain are blacklisted #
###################################################################
def is_blacklisted(third_party_email, blacklisted_emails, blacklisted_domains):
    third_party_email = third_party_email.lower()
    # *endswith* to match various handles, e.g., <messages-noreply@linkedin.com>
    # with linkedin.com
    return (
        any(third_party_email == email.lower() for email in blacklisted_emails) or
        any(third_party_email.endswith(f"@{domain.lower()}") for domain in blacklisted_domains)
    )

#################################
# Helper 3: Decode email header #
#################################
def decode_email_header(header_value):
    # source: https://dnmtechs.com/decoding-utf-8-email-headers-in-python-3/
    from email.header import decode_header
    
    decoded_header = decode_header(header_value)
    decoded_parts = []

    for part, encoding in decoded_header:
        if isinstance(part, bytes):
            decoded_parts.append(part.decode(encoding or "utf-8"))
        else:
            decoded_parts.append(part)
    
    return " ".join(decoded_parts)

############################################################
# Helper 4: Format reply as answer to quoted original body #
############################################################
def format_response_quoting_original_body(proposed_reply, original_body):
    # e.g.,
    # This is the language model reply.
    # ...
    # > This is the original question.
    # > ...
    quoted_lines = [f"> {line}" for line in original_body.strip().split("\n")]
    quoted_text = "\n".join(quoted_lines)
    return f"""{proposed_reply}

{quoted_text}"""

############################################
# Helper 5: Compact email body for decoder #
############################################
def compact_email_body_for_decoder(
        tokenizer,
        body,
        max_unquoted_tokens,
        max_quoted_tokens,
        missing_body_placeholder,
        unquoted_fail_placeholder=None,
        quoted_fail_placeholder=None,
        log_prefix=""
        ):
    from helpers.data import get_unquoted_text
    from helpers.decoder import truncate_to_tokens

    if not body:
        body = missing_body_placeholder

    unquoted_body, quoted_body = get_unquoted_text(body, return_quoted=True)
    if unquoted_body:
        if max_unquoted_tokens >= 0:
            unquoted_body = truncate_to_tokens(tokenizer, unquoted_body, max_unquoted_tokens)
            if unquoted_body is None:
                if log_prefix:
                    print(f"{log_prefix} unquoted_body tokenization in truncate_to_tokens failed")
                if unquoted_fail_placeholder is None:
                    return None
                unquoted_body = unquoted_fail_placeholder
    if quoted_body:
        if max_quoted_tokens == 0:
            quoted_body = ""
        elif max_quoted_tokens > 0:
            quoted_body = truncate_to_tokens(tokenizer, quoted_body, max_quoted_tokens)
            if quoted_body is None:
                if log_prefix:
                    print(f"{log_prefix} quoted_body tokenization in truncate_to_tokens failed")
                quoted_body = quoted_fail_placeholder if quoted_fail_placeholder is not None else ""

    return f"{unquoted_body}\n\n{quoted_body}".strip()

################################
# Helper 6: Read latest emails #
################################
def read_latest_emails(
        max_emails,
        folder,
        last_n_days,
        imap_email,
        password,
        unread_only,
        imap_server,
        imap_port,
        blacklisted_emails,
        blacklisted_domains
        ):
    from config.email_agent import INBOX_FOLDER, SENT_FOLDER
    # https://docs.python.org/3/library/imaplib.html
    import email
    from imaplib import IMAP4_SSL
    from email.utils import parseaddr, parsedate_to_datetime
    from datetime import datetime, timedelta, timezone
    
    try:
        # getting a hold of the imap server with context manager:
        with IMAP4_SSL(imap_server, imap_port) as imap:
            # log in
            imap.login(imap_email, password)

            # select folder
            imap.select(folder)

            # search for either unseen or seen and unseen emails
            use_unread_filter = unread_only and folder == INBOX_FOLDER
            search_criteria = "(UNSEEN)" if use_unread_filter else "ALL"

            # get all messages that fit the (above) criteria
            retcode, messages = imap.search(None, search_criteria)

            # get message ids
            email_ids = messages[0].split()

            # set up a list to hold ids, senders, dates and bodies
            emails_contents = []

            # calculate cutoff date
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=last_n_days)

            # from latest to the oldest id:
            for email_id in reversed(email_ids):
                body = ""
                ignore_message = False
                try:
                    # BODY.PEEK[] keeps emails unread while they are being processed
                    fetch_command = "(BODY.PEEK[])"

                    # get message data
                    status, message_data = imap.fetch(email_id, fetch_command)
                    message = email.message_from_bytes(message_data[0][1])

                    # compare dates
                    try:
                        email_date = message.get("Date", "")
                        parsed_date = parsedate_to_datetime(email_date)
                        if parsed_date.tzinfo is None:
                            parsed_date = parsed_date.replace(tzinfo=timezone.utc)
                        if parsed_date < cutoff_date:
                            print(f"read_latest_emails: email date {email_date} is older than {last_n_days} days: stopping")
                            break
                        email_date = parsed_date # use parsed date only if parsing succeeds
                    except Exception as e:
                        print(f"read_latest_emails: warning: could not parse date '{email_date}' for email {email_id}: {e}")
                    
                    # get sender
                    raw_from = decode_email_header(message.get("From", ""))
                    _, from_ = parseaddr(raw_from)
                    # get recipients
                    to_ = decode_email_header(message.get("To", ""))

                    # ignore if sender is blacklisted (skip for sent folder)
                    if folder != SENT_FOLDER and is_blacklisted(from_, blacklisted_emails, blacklisted_domains):
                        print(f"read_latest_emails: email '{from_}' is blacklisted: skipping")
                        continue

                    # get subject
                    subject = decode_email_header(message.get("Subject", ""))

                    # get message
                    if message.is_multipart():
                        # walking if multipart (e.g., HTML, attachments, plain text) to find plain text
                        for part in message.walk():

                            # ignoring the full message if attachments are present
                            content_disposition = str(part.get("Content-Disposition"))
                            if "attachment" in content_disposition:
                                ignore_message = True
                            
                            # and using the plain text when reached (discarding other parts)
                            if not ignore_message and part.get_content_type() == "text/plain":
                                try:
                                    payload = part.get_payload(decode=True)
                                    charset = part.get_content_charset()
                                    body = payload.decode(charset or "utf-8")
                                except Exception as e:
                                    print(f"read_latest_emails: error extracting body: {e}")
                    
                    # or decoding content if it is plain text
                    else:
                        try:
                            payload = message.get_payload(decode=True)
                            charset = message.get_content_charset()
                            body = payload.decode(charset or "utf-8")
                        except Exception as e:
                            print(f"read_latest_emails: error extracting body: {e}")

                    # append email id, sender, date, subject and message body
                    if not ignore_message:
                        emails_contents.append({"id": email_id, "from": from_, "to": to_, "date": email_date, "subject": subject, "message_body": body})
                        if len(emails_contents) == 1:
                            print(f"read_latest_emails: sample email format: {emails_contents[0]}")
                    ignore_message = False
                    
                    # break upon reaching max_emails
                    if len(emails_contents) >= max_emails:
                        break

                except Exception as e:
                    ignore_message = False
                    print(f"read_latest_emails: error processing email {email_id}: {e}")
                    continue

        # return email contents
        return emails_contents

    except Exception as e:
        print(f"read_latest_emails: error reading emails: {e}")
        return []
    
#########################
# Helper 7: Save drafts #
#########################
def save_drafts(
        reply_bodies,
        original_subjects,
        imap_email,
        smtp_email,
        recipient_emails,
        password,
        imap_server,
        imap_port,
        drafts_folder
        ):
    # docs.python.org/3/library/smtplib.html
    # https://docs.python.org/3/library/imaplib.html
    import time
    import imaplib
    from imaplib import IMAP4_SSL
    from email.utils import formatdate
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    try:
        # getting a hold of the imap server with context manager:
        with IMAP4_SSL(imap_server, imap_port) as imap:
            # log in
            imap.login(imap_email, password)

            # for each email to save a draft for:
            for i in range(len(original_subjects)):
                # get reply body, subject, recipient email
                reply_body = reply_bodies[i]
                original_subject = original_subjects[i]
                recipient_email = recipient_emails[i]

                # create message container
                message = MIMEMultipart()
                message["From"] = smtp_email
                message["To"] = recipient_email
                message["Subject"] = f"Re: {original_subject}"
                message["Date"] = formatdate(localtime=True)

                # attach body
                message.attach(MIMEText(reply_body, "plain", "utf-8"))

                # select folder
                status, _ = imap.select(drafts_folder)
                if status != "OK":
                    available = imap.list()[1]
                    return False, f"save_drafts: folder '{drafts_folder}' not found. Server lists: {available}"

                # save
                imap.append(drafts_folder, "\\Draft", imaplib.Time2Internaldate(time.time()), message.as_bytes())

                # courtesy wait
                time.sleep(1)

        return True, ""
    
    except Exception as e:
        return False, str(e)

#########################
# Helper 8: Send emails #
#########################
def send_emails(
        reply_bodies,
        original_subjects,
        smtp_email,
        recipient_emails,
        password,
        smtp_server,
        smtp_port
        ):
    # docs.python.org/3/library/smtplib.html
    import time
    from smtplib import SMTP
    from email.utils import formatdate
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    try:
        # getting a hold of the smtp server with context manager:
        with SMTP(smtp_server, smtp_port) as smtp:
            # set debug output level
            smtp.set_debuglevel(1)
            # put SMTP connection in TLS (Transport Layer Security) mode
            smtp.starttls()
            # log in on SMTP server that requires authentication
            smtp.login(smtp_email, password)

            # for each email to send a reply to:
            for i in range(len(original_subjects)):
                # get reply body, subject, recipient email
                reply_body = reply_bodies[i]
                original_subject = original_subjects[i]
                recipient_email = recipient_emails[i]

                # create message container
                message = MIMEMultipart()
                message["From"] = smtp_email
                message["To"] = recipient_email
                message["Subject"] = f"Re: {original_subject}"
                message["Date"] = formatdate(localtime=True)

                # attach body
                message.attach(MIMEText(reply_body, "plain", "utf-8"))

                # send mail
                smtp.sendmail(smtp_email, recipient_email, message.as_string())

                # courtesy wait
                time.sleep(1)

        return True, ""
    
    except Exception as e:
        return False, str(e)

#################################
# Helper 9: Mark emails as read #
#################################
def mark_emails_as_read(
        email_ids,
        imap_email,
        password,
        imap_server,
        imap_port
        ):
    # https://docs.python.org/3/library/imaplib.html
    from imaplib import IMAP4_SSL

    try:
        # getting a hold of the imap server with context manager:
        with IMAP4_SSL(imap_server, imap_port) as imap:
            # log in
            imap.login(imap_email, password)

            # select inbox
            imap.select("INBOX")

            # for each processed email, add Seen flag
            for email_id in email_ids:
                imap.store(email_id, "+FLAGS", "\\Seen")

        return True, ""

    except Exception as e:
        return False, str(e)
