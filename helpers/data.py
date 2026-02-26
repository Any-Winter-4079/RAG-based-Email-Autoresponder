###################################################
# Helper 1: Check if email is from/to UPM domains #
###################################################
def is_upm_internal(author, recipients, upm_domains):
    import re
    # if all participants (author and recipients) aren't students / external people (e.g., they are professors):
    # return True to remove from the dataset
    author_email = re.findall(r'[\w\.-]+@[\w\.-]+', author.lower()) # list
    recipient_emails = re.findall(r'[\w\.-]+@[\w\.-]+', recipients.lower()) # list
    all_emails = author_email + recipient_emails
    return all(any(f"@{domain}" in email for domain in upm_domains) for email in all_emails) and len(all_emails) > 0

##########################################
# Helper 2: Normalize email subject text #
##########################################
def normalize_subject(subject):
    # convert to lowercase and remove reply/forward prefixes
    normalized = subject.strip().lower()
    prefixes = ["re:", "fw:", "fwd:"]
    prefix_removed = True
    while prefix_removed:
        prefix_removed = False
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):].strip()
                prefix_removed = True
    return normalized

#######################################
# Helper 3: Normalize email body text #
#######################################
def normalize_email_body(body):
    return body.replace("\n", " ").replace("\r", " ").strip().lower()

#####################################
# Helper 4: Plot distribution chart #
#####################################
def plot_distribution(labels, sizes, title):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 8))
    plt.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%', # one decimal place
        counterclock=False
    )
    plt.title(title)
    plt.axis("equal")
    plt.show()

################################################
# Helper 5: Print and return folder URI counts #
################################################
def get_and_print_folder_uri_counts(folder_uri_column, title, previous_counts=None):
    from collections import Counter

    folder_uri_counts = Counter(folder_uri_column)
    print(f"\n{title}")
    # print Alumnos/Seminarios/... counts from folderURI
    for uri, count in folder_uri_counts.items():
        diff_text = ""
        if previous_counts is not None:
            diff = count - previous_counts.get(uri, 0)
            diff_text = " (==)" if diff == 0 else f" ({diff:+d})"
        print(f"\t{uri.split('/')[-1]}: {count} messages{diff_text}")
    total_count = sum(folder_uri_counts.values())
    total_diff_text = ""
    if previous_counts is not None:
        total_diff = total_count - sum(previous_counts.values())
        total_diff_text = " (==)" if total_diff == 0 else f" ({total_diff:+d})"
    print(f"\tTotal: {total_count} messages{total_diff_text}")
    return folder_uri_counts

###########################################################
# Helper 6: Assign thread ids by contiguous same-subjects #
###########################################################
def assign_thread_ids_by_subject_blocks(emails):
    if not emails:
        return []

    emails_to_process = list(emails)

    thread_id = 0
    first_subject = emails_to_process[0].get("subject") or ""
    current_subject = normalize_subject(first_subject)
    current_block_emails = []
    emails_with_threads = []

    def flush_block():
        nonlocal thread_id, current_block_emails
        thread_id += 1
        for block_email in current_block_emails:
            email_with_thread = block_email.copy()
            email_with_thread["threadID"] = thread_id
            emails_with_threads.append(email_with_thread)
        current_block_emails = []

    for email in emails_to_process:
        subject = email.get("subject") or ""
        normalized_subject = normalize_subject(subject)
        if normalized_subject != current_subject:
            flush_block()
            current_subject = normalized_subject

        current_block_emails.append(email)

    if current_block_emails:
        flush_block()

    return emails_with_threads

#########################################################################
# Helper 7: Assign thread ids by contiguous same-subjects (for dataset) #
#########################################################################
def assign_thread_ids_by_subject_blocks_for_dataset(rows):
    # insert 'threadID' as 2nd column on 1st row
    rows_with_threads = [rows[0][:1] + ["threadID"] + rows[0][1:]]

    data_rows = rows[1:]
    email_subjects = [{"subject": row[1]} for row in data_rows]
    emails_with_threads = assign_thread_ids_by_subject_blocks(email_subjects)

    for row, email in zip(data_rows, emails_with_threads):
        rows_with_threads.append(row[:1] + [email["threadID"]] + row[1:])

    return rows_with_threads

###################################################################################
# Helper 8: Assign thread ids by subject and participant overlap (for production) #
###################################################################################
def assign_thread_ids_by_subject_and_participant_overlap_for_production(emails, my_email_addresses):
    import re
    if not emails:
        return []

    my_email_addresses = set(
        email.lower()
        for email in (my_email_addresses or [])
        if email
    )

    emails_with_threads = {}

    def extract_participants(from_, to_):
        participants = set()
        for text in [from_, to_]:
            if not text:
                continue
            for email in re.findall(r'[\w\.-]+@[\w\.-]+', text.lower()):
                participants.add(email)
        return participants

    for email in emails:
        email_subject = email.get("subject") or ""
        email_normalized_subject = normalize_subject(email_subject)
        email_participants = extract_participants(email.get("from"), email.get("to"))
        email_participants = {email for email in email_participants if email not in my_email_addresses}
        if not email_participants:
            continue
        email_participants_key = tuple(sorted(email_participants))

        found_key_match = False
        # if it's the first email, add
        if len(emails_with_threads) == 0:
            emails_with_threads[(email_normalized_subject, email_participants_key)] = [email]
        # otherwise:
        else:
            # for what we've seen so far
            for key in list(emails_with_threads.keys()):
                thread_normalized_subject = key[0]
                thread_participants = key[1]
                # if our current email's normalized subject match and participants intersect:
                if email_normalized_subject == thread_normalized_subject and set(thread_participants).intersection(email_participants):
                    # calculate the new key (extending participants)
                    new_thread_participants = email_participants.union(thread_participants)
                    new_thread_participants_key = tuple(sorted(new_thread_participants))
                    # set as new emails for the (normalized subject, extended set) the old emails (popping the key) and new email
                    thread_emails = emails_with_threads.pop(key)
                    thread_emails.append(email)
                    emails_with_threads[(thread_normalized_subject, new_thread_participants_key)] = thread_emails
                    found_key_match = True
            # and if no match (with a normalized subject and participants), add as new key/value
            if not found_key_match:
                emails_with_threads[(email_normalized_subject, email_participants_key)] = [email]

    # add thread ids
    emails_with_threads_list = []
    for thread_id, thread_emails in enumerate(emails_with_threads.values(), start=1):
        for thread_email in thread_emails:
            email_with_thread = thread_email.copy()
            email_with_thread["threadID"] = thread_id
            emails_with_threads_list.append(email_with_thread)

    return emails_with_threads_list

#############################################
# Helper 9: Split quoted messages from body #
#############################################
def get_unquoted_text(body, return_quoted=False):
    import re
    # get 1st match
    match = re.search(
        r"\s*(en .*?\b\d{4}\b.* escribió:|en .*?<[^>]*@[^>]*>.* escribió:|on .*?\b\d{4}\b.* wrote[:：]|on .*?<[^>]*@[^>]*>.* wrote[:：]|de:.*<[^>]*@[^>]*>.*enviado.*para:.*<[^>]*@[^>]*>.*asunto:|de:.*enviado:.*para:.*asunto:)",
        body.lower(),
        flags=re.IGNORECASE | re.DOTALL
    )
    if match:
        unquoted = body[:match.start()]
        quoted = body[match.start():]
    else:
        unquoted = body
        quoted = ""
    unquoted = unquoted.strip()
    quoted = quoted.strip()
    return (unquoted, quoted) if return_quoted else unquoted

#########################################################
# Helper 10: Check template matches in unquoted content #
#########################################################
def has_template_in_unquoted(body, templates):
    unquoted = normalize_email_body(get_unquoted_text(body))
    return any(template in unquoted for template in templates)

###########################
# Helper 11: Save dataset #
###########################
def save_dataset(rows, output_path, delimiter=";"):
    import csv
    with open(output_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=delimiter)
        writer.writerows(rows)
    print(f"Saved dataset to {output_path}")
