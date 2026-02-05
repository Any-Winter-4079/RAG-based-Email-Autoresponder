import modal

MAX_EMAILS = 2
UNREAD_ONLY = True
LEAVE_UNREAD = False # NOTE: if True, replies may be rewritten if they are not read before next execution
LAST_N_DAYS = 120
SEND_TO_SELF = True
SAVE_AS_DRAFT = True
DRAFTS_FOLDER = "Drafts"

EMAIL_HOUR = 9
EMAIL_MINUTE = 0

PYTHON_VERSION = "3.11"

MODAL_TIMEOUT = 600 # seconds

image = modal.Image.debian_slim(python_version=PYTHON_VERSION).add_local_python_source("config", "helpers")