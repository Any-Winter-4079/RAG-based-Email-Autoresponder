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
def assign_thread_ids_by_subject_blocks(rows):
    # insert 'threadID' as 2nd column on 1st row
    rows_with_threads = [rows[0][:1] + ["threadID"] + rows[0][1:]]
    if len(rows) <= 1:
        return rows_with_threads

    thread_id = 0
    current_subject = normalize_subject(rows[1][1])
    current_block_rows = []

    def flush_block():
        nonlocal thread_id, current_block_rows
        thread_id += 1
        for block_row in current_block_rows:
            rows_with_threads.append(block_row[:1] + [thread_id] + block_row[1:])
        current_block_rows = []

    for row in rows[1:]:
        normalized_subject = normalize_subject(row[1])
        if normalized_subject != current_subject:
            flush_block()
            current_subject = normalized_subject

        current_block_rows.append(row)

    if current_block_rows:
        flush_block()

    return rows_with_threads

##############################################
# Helper 7: Remove quoted messages from body #
##############################################
def get_unquoted_text(body):
    import re
    match = re.search(
        r"\s*(en .*?\b\d{4}\b.* escribió:|en .*?<[^>]*@[^>]*>.* escribió:|on .*?\b\d{4}\b.* wrote[:：]|on .*?<[^>]*@[^>]*>.* wrote[:：]|de:.*<[^>]*@[^>]*>.*enviado.*para:.*<[^>]*@[^>]*>.*asunto:|de:.*enviado:.*para:.*asunto:)",
        body.lower(),
        flags=re.IGNORECASE | re.DOTALL
    )
    if match:
        return body[:match.start()]
    return body

########################################################
# Helper 8: Check template matches in unquoted content #
########################################################
def has_template_in_unquoted(body, templates):
    unquoted = normalize_email_body(get_unquoted_text(body))
    return any(template in unquoted for template in templates)

##########################
# Helper 9: Save dataset #
##########################
def save_dataset(rows, output_path, delimiter=";"):
    import csv
    with open(output_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=delimiter)
        writer.writerows(rows)
    print(f"\nSaved dataset to {output_path}")
