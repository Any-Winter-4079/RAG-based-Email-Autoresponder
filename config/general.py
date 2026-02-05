import modal

SECRET_NAME = "MUIA-SECRET"

modal_secret = modal.Secret.from_name(SECRET_NAME)