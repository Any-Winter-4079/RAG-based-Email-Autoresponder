import modal

MAX_EMAILS = 2
CONTEXT_EMAILS_PER_FOLDER = 20
# < 0 to keep all tokens
MAX_UNQUOTED_TOKENS_PER_CURRENT_EMAIL = 1500
MAX_UNQUOTED_TOKENS_PER_CONTEXT_EMAIL = 1000
MAX_QUOTED_TOKENS_PER_CURRENT_EMAIL = -1
MAX_QUOTED_TOKENS_PER_CONTEXT_EMAIL = 0
INBOX_FOLDER = "Inbox"
SENT_FOLDER = "Sent"
UNREAD_ONLY = True
LEAVE_UNREAD = False # NOTE: if True, replies may be rewritten if they are not read before next execution
LAST_N_DAYS = 120
SEND_TO_SELF = True
SAVE_AS_DRAFT = True
DRAFTS_FOLDER = "Drafts"

EMAIL_HOUR = 9
EMAIL_MINUTE = 0

PYTHON_VERSION = "3.11"

PACKAGES = [
    "transformers==4.57.0"
]

MODAL_TIMEOUT = 600 # seconds

image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .pip_install(*PACKAGES)
    .add_local_python_source("config", "helpers")
)
