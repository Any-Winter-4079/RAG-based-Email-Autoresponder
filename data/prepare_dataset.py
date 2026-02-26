import sys
from os.path import dirname, abspath
project_root = abspath(dirname(dirname(__file__)))
sys.path.insert(0, project_root)

import os
import csv
from cycler import cycler
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from helpers.email_agent import transform_env_csv_into_list
from config.data import (
    DATASET_PATH,
    TRAIN_SPLIT_PATH,
    MESSAGES_WITH_THREADS_DATASET_PATH,
    AUTOMATED_OUTBOUND_TEMPLATES,
    PRE_ENROLLMENT_TEMPLATES,
)
from helpers.data import (
    normalize_subject,
    normalize_email_body,
    plot_distribution,
    get_and_print_folder_uri_counts,
    assign_thread_ids_by_subject_blocks_for_dataset,
    save_dataset,
    get_unquoted_text,
    has_template_in_unquoted
)

load_dotenv()
my_email_addresses = transform_env_csv_into_list(os.getenv("MY_EMAIL_ADDRESSES", ""))

custom_colors = [
    "#FFAF00", "#F46920", "#F53255", "#F857C1",
     "#29BDFD", "#00CBBF", "#01C159", "#9DCA1C"
]
plt.rcParams["axes.prop_cycle"] = cycler(color=custom_colors)

def main():
    with open(DATASET_PATH, newline='') as csv_file:
        rows = list(csv.reader(csv_file, delimiter=';'))
        header_row = rows[0]
        previous_folder_uri_counts = None

        print("\n"+"="*50)
        print("Step 0")
        print("="*50)
        print(f"\nTotal data columns: {len(header_row)}")
        print(f"Column names: {header_row}")
        print(f"Total data rows (excluding header with column names): {len(rows[1:])}")
        outbound_count = 0
        inbound_to_director_count = 0
        for row in rows[1:]:
            author = row[5].lower()
            recipients = row[6].lower()
            is_outbound = any(email in author for email in my_email_addresses)
            is_inbound_to_director = any(email in recipients for email in my_email_addresses)
            if is_outbound:
                outbound_count += 1
            if is_inbound_to_director:
                inbound_to_director_count += 1
        print(f"\nUsing director emails: {my_email_addresses}")
        print("Author distribution:")
        print(f"\tOutbound (author is director): {outbound_count} messages")
        print(f"\tInbound to director (recipient includes director): {inbound_to_director_count} messages")

        ########################################################
        # Remove id column (due to being unique, not relevant) #
        ########################################################
        print("\n"+"="*50)
        print("Step 1: removing id column")
        print("="*50)
        # obtain the column id values by getting the 1st value in each row
        id_column = [row[0] for row in rows[1:]] # skipping header
        # drop id if values they are unique
        if len(set(id_column)) == len(id_column):
            rows_no_id = [row[1:] for row in rows]
        else:
            raise ValueError("Duplicate IDs found")
        print(f"\nColumn names ({len(rows_no_id[0])} total): {rows_no_id[0]}")

        ###################################################################################
        # Remove folderID column (due to matching folderURI while being less descriptive) #
        ###################################################################################
        print("\n"+"="*50)
        print("Step 2: removing folderID column")
        print("="*50)
        # obtain the folderID values by getting the (now) 1st value in each row
        folder_id_column = [row[0] for row in rows_no_id[1:]] # skipping header
        unique_folder_ids = set(folder_id_column)
        print(f"\nUnique folderID values: {len(unique_folder_ids)}")

        # obtain the folderURI values by getting the (now) 2st value in each row
        folder_uri_column = [row[1] for row in rows_no_id[1:]] # skipping header
        unique_folder_uris = set(folder_uri_column)
        print(f"Unique folderURI values: {len(unique_folder_uris)}")

        # drop folderID column (first column) from each row
        rows_no_id_and_no_folder_id = [row[1:] for row in rows_no_id]
        print(f"Column names ({len(rows_no_id_and_no_folder_id[0])} total): {rows_no_id_and_no_folder_id[0]}")

        # show distribution
        previous_folder_uri_counts = get_and_print_folder_uri_counts(
            [row[0] for row in rows_no_id_and_no_folder_id[1:]], # folderURI is the 1st value and we skip the header
            title="Distribution of messages by folderURI:"
        )
        
        #################################
        # Remove rows with empty c0body #
        #################################
        print("\n"+"="*50)
        print("Step 3: removing rows with empty c0body")
        print("="*50)
        non_empty_c0body_rows = [rows_no_id_and_no_folder_id[0]] # keep header
        for row in rows_no_id_and_no_folder_id[1:]:
            if row[2].strip(): # check that the body is non-empty
                non_empty_c0body_rows.append(row)

        # show updated distribution
        previous_folder_uri_counts = get_and_print_folder_uri_counts(
            [row[0] for row in non_empty_c0body_rows[1:]], # folderURI is the 1st value and we skip the header
            title="Distribution of messages by folderURI (post empty body removal):",
            previous_counts=previous_folder_uri_counts
        )

        #####################################################################################################
        # Remove pre-enrollment template messages (due to being messages not suited for automated response) #
        #####################################################################################################
        print("\n"+"="*50)
        print("Step 4: removing pre-enrollment template messages")
        print("="*50)
        normalized_pre_enrollment_templates = [template.lower() for template in PRE_ENROLLMENT_TEMPLATES]
        pre_enrollment_counts = {template: 0 for template in PRE_ENROLLMENT_TEMPLATES}
        for template, normalized_template in zip(PRE_ENROLLMENT_TEMPLATES, normalized_pre_enrollment_templates):
            pre_enrollment_counts[template] = sum(
                1 for row in non_empty_c0body_rows[1:]
                if normalized_template in normalize_email_body(get_unquoted_text(row[2]))
            )
        print("\nPre-enrollment template instance occurrences:")
        for template in PRE_ENROLLMENT_TEMPLATES:
            print(f"\ttemplate '{template}': {pre_enrollment_counts[template]} messages")

        pre_enrollment_filtered_rows = [non_empty_c0body_rows[0]] # keep header
        for row in non_empty_c0body_rows[1:]: # skipping header
            if not has_template_in_unquoted(row[2], normalized_pre_enrollment_templates):
                pre_enrollment_filtered_rows.append(row)

        pre_enrollment_match_counts = {matches: 0 for matches in range(0, len(normalized_pre_enrollment_templates) + 1)}
        for row in non_empty_c0body_rows[1:]:
            matches = sum(1 for template in normalized_pre_enrollment_templates if template in normalize_email_body(get_unquoted_text(row[2])))
            pre_enrollment_match_counts[matches] += 1
        print("\n\trows by pre-enrollment template match count:")
        for matches in range(0, len(normalized_pre_enrollment_templates) + 1):
            print(f"\t\t{matches} {'template: ' if matches == 1 else 'templates:'} {pre_enrollment_match_counts[matches]} rows")

        # show updated distribution
        previous_folder_uri_counts = get_and_print_folder_uri_counts(
            [row[0] for row in pre_enrollment_filtered_rows[1:]], # folderURI is the 1st value and we skip the header
            title="Distribution of messages by folderURI (post pre-enrollment filtering):",
            previous_counts=previous_folder_uri_counts
        )

        #############################################################################################
        # Remove (admission/rejection) template messages (due to being automated outbound messages) #
        #############################################################################################
        # NOTE: keeping an instance per template for train split / knowledge base
        print("\n"+"="*50)
        print("Step 5: removing admission/rejection template messages")
        print("="*50)
        normalized_templates = [message.lower() for message in AUTOMATED_OUTBOUND_TEMPLATES]
        template_sample_rows = [pre_enrollment_filtered_rows[0]]
        collected_templates = set()
        # count ocurrences of messages that are templates and not handwritten
        print("\nAdmission/rejection template instance occurrences:")
        for message in AUTOMATED_OUTBOUND_TEMPLATES:
            count = sum(
                1 for row in pre_enrollment_filtered_rows[1:]
                if message.lower() in normalize_email_body(get_unquoted_text(row[2]))
            )
            print(f"\ttemplate '{message}': {count} messages")
        multi_template_rows = sum(
            1 for row in pre_enrollment_filtered_rows[1:]
            if sum(1 for template in normalized_templates if template in normalize_email_body(get_unquoted_text(row[2]))) > 1
        )
        print(f"\n\trows with >1 template match: {multi_template_rows}")
        match_counts = {matches: 0 for matches in range(0, len(normalized_templates) + 1)}
        for row in pre_enrollment_filtered_rows[1:]:
            matches = sum(1 for template in normalized_templates if template in normalize_email_body(get_unquoted_text(row[2])))
            match_counts[matches] += 1
        print("\n\trows by template match count:")
        for matches in range(0, len(normalized_templates) + 1):
            print(f"\t\t{matches} {'template: ' if matches == 1 else 'templates:'} {match_counts[matches]} rows")

        # keep one instance per admission/rejection template for train split
        for row in pre_enrollment_filtered_rows[1:]:
            normalized_body = normalize_email_body(get_unquoted_text(row[2]))
            for template in normalized_templates:
                if template in normalized_body and template not in collected_templates:
                    template_sample_rows.append(row)
                    collected_templates.add(template)
                    break
        if template_sample_rows[1:]:
            print(f"\nCollected {len(template_sample_rows) - 1} template samples for train split")

        # filter out rows with admission / rejection to MUIA messages
        filtered_rows = [pre_enrollment_filtered_rows[0]] # keep header
        for row in pre_enrollment_filtered_rows[1:]: # skipping header
            if not has_template_in_unquoted(row[2], normalized_templates):
                filtered_rows.append(row)

        # show updated distribution
        previous_folder_uri_counts = get_and_print_folder_uri_counts(
            [row[0] for row in filtered_rows[1:]], # folderURI is the 1st value and we skip the header
            title="Distribution of messages by folderURI (post admission/rejection filtering):",
            previous_counts=previous_folder_uri_counts
        )

        ##################################################################
        # Remove duplicate rows (same body, subject, author, recipients) #
        ##################################################################
        print("\n"+"="*50)
        print("Step 6: removing duplicate rows (spam, etc.)")
        print("="*50)
        seen_messages = set()
        dedup_filtered_rows = [filtered_rows[0]] # keep header
        for row in filtered_rows[1:]:
            normalized_subject = normalize_subject(row[1]) # c1subject is the 2nd column after id and folderID removal
            normalized_body = normalize_email_body(row[2]) # c0body is the 3rd column after id and folderID removal
            author = row[3].strip().lower()
            recipients = row[4].strip().lower()
            message_key = (normalized_body, normalized_subject, author, recipients)
            if message_key not in seen_messages:
                seen_messages.add(message_key)
                dedup_filtered_rows.append(row)
        
        # show updated distribution
        previous_folder_uri_counts = get_and_print_folder_uri_counts(
            [row[0] for row in dedup_filtered_rows[1:]], # folderURI is the 1st value and we skip the header
            title="Distribution of messages by folderURI (post deduplication):",
            previous_counts=previous_folder_uri_counts
        )

        ##########################
        # Group emails by thread #
        ##########################
        # NOTE: adding extra column with unique id per thread
        print("\n"+"="*50)
        print("Step 7: grouping emails by thread")
        print("="*50)

        dedup_filtered_rows_with_threads = assign_thread_ids_by_subject_blocks_for_dataset(dedup_filtered_rows)
        thread_ids = {row[-1] for row in dedup_filtered_rows_with_threads[1:]}
        print(f"\nThread count: {len(thread_ids)} (rows: {len(dedup_filtered_rows_with_threads) - 1})")

        # get folderURI counts and show updated distribution
        folder_uri_counts = get_and_print_folder_uri_counts(
            [row[0] for row in dedup_filtered_rows_with_threads[1:]], # folderURI is the 1st value and we skip the header
            title="Distribution of messages by folderURI (post thread grouping):",
            previous_counts=previous_folder_uri_counts
        )

        train_sample_rows_with_threads = [template_sample_rows[0][:1] + ["threadID"] + template_sample_rows[0][1:]]
        negative_thread_id = -1
        for row in template_sample_rows[1:]:
            train_sample_rows_with_threads.append(row[:1] + [negative_thread_id] + row[1:])
            negative_thread_id -= 1

        #################################################################
        # Save csv files (one with train samples, another with threads) #
        #################################################################
        print()
        save_dataset(train_sample_rows_with_threads, TRAIN_SPLIT_PATH)
        save_dataset(dedup_filtered_rows_with_threads, MESSAGES_WITH_THREADS_DATASET_PATH)

        ######################
        # Plot final results #
        ######################
        # plot folderURI distributions
        plot_distribution(
            [uri.split('/')[-1] for uri in folder_uri_counts.keys()],
            list(folder_uri_counts.values()),
            title="Distribution of messages by folderURI (after data cleaning)"
        )
        count = sum(
            1 for row in dedup_filtered_rows_with_threads[1:] # skip the header
            if any(email in row[3] for email in my_email_addresses) # c3author
        )
        plot_distribution(
            ["Authored by director", "Authored by another"],
            [count, len(dedup_filtered_rows_with_threads) - 1 - count], # skip the header
            title="Distribution of messages by author (after data cleaning)"
        )

if __name__ == "__main__":
    main()
