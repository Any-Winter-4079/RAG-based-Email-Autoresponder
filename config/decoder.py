import modal

COMMON_PACKAGES = [
    "torchvision",
    "transformers==4.57.0",
    "accelerate",
    "peft==0.17.1",
    "Pillow",
    "requests",
    "hf_transfer",
]
GPU = "L40S"
SCALEDOWN_WINDOW = 60 # seconds
TIMEOUT = 900 # seconds
MIN_CONTAINERS = 0 # 0 to make sure we don't pay 24/7
FLASH_ATTENTION_RELEASE = "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
FLASH_ATTENTION_IMAGE = "anywinter4079/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04-runpod-clone"
FLASH_ATTENTION_RUN_COMMANDS = ("python -m pip install --upgrade pip && "
                                "pip config set global.extra-index-url https://download.pytorch.org/whl/cu128")
FLASH_ATTENTION_TORCH_VERSION = "torch==2.8.0+cu128"
USE_FLASH_ATTENTION_IMAGE = False # TODO: fix flash_attn image (should be faster than image without flash_attn)
NO_FLASH_ATTENTION_PYTHON_VERSION = "3.11"
NO_FLASH_ATTENTION_TORCH_VERSION = "torch==2.8.0"

NO_MESSAGE_OPENING_TAG = "<nomessage>"
NO_MESSAGE_CLOSING_TAG = "</nomessage>"
MESSAGE_OPENING_TAG = "<message>"
MESSAGE_CLOSING_TAG = "</message>"

ABSTRACT_OPENING_TAG = "<abstract>"
ABSTRACT_CLOSING_TAG = "</abstract>"

SUMMARY_OPENING_TAG = "<summary>"
SUMMARY_CLOSING_TAG = "</summary>"

CLEANED_TEXT_OPENING_TAG = "<cleanedtext>"
CLEANED_TEXT_CLOSING_TAG = "</cleanedtext>"

QUESTIONS_OPENING_TAG = "<questions>"
QUESTIONS_CLOSING_TAG = "</questions>"
QUESTION_OPENING_TAG = "<question>"
QUESTION_CLOSING_TAG = "</question>"
ANSWER_OPENING_TAG = "<answer>"
ANSWER_CLOSING_TAG = "</answer>"

THREAD_OPENING_TAG = "<thread>"
THREAD_CLOSING_TAG = "</thread>"
THREAD_MESSAGE_OPENING_TAG = "<message>"
THREAD_MESSAGE_CLOSING_TAG = "</message>"
THREAD_FROM_OPENING_TAG = "<from>"
THREAD_FROM_CLOSING_TAG = "</from>"
THREAD_TO_OPENING_TAG = "<to>"
THREAD_TO_CLOSING_TAG = "</to>"
THREAD_SUBJECT_OPENING_TAG = "<subject>"
THREAD_SUBJECT_CLOSING_TAG = "</subject>"
THREAD_BODY_OPENING_TAG = "<body>"
THREAD_BODY_CLOSING_TAG = "</body>"

THREAD_GROUPER_MAX_EMAILS = 20
EMAIL_WRITER_PROFILE = "email_writer"
THREAD_GROUPER_PROFILE = "thread_grouper"
DATA_CLEANER_PROFILE = "data_cleaner"

DIRECTOR_EMAIL = "masteria.dia@fi.upm.es"
DIRECTOR_NAME = "Damiano Zanardini"
DEPARTMENT_PHONE = "+34 910672909"

# anonymized
EXAMPLE_STUDENT_NAME = "Marco Conti"
EXAMPLE_STUDENT_EMAIL = "marco.conti@uxg.edu"
EXAMPLE_STAFF_NAME = "Laura Medina"
EXAMPLE_STAFF_EMAIL = "laura.medina@fi.upm.es"
EXAMPLE_COLLEAGUE_NAME = "Alex Perez"
EXAMPLE_COLLEAGUE_EMAIL = "alex.perez@fi.upm.es"
EXAMPLE_STUDENT_REP_NAME = "Javier Ruiz"
EXAMPLE_STUDENT_REP_EMAIL = "javier.ruiz@alumnos.upm.es"
EXAMPLE_DIRECTOR_PEER_NAME = "Elena Torres"
EXAMPLE_DIRECTOR_PEER_EMAIL = "elena.torres@fi.upm.es"
EXAMPLE_PROF1_EMAIL = "carmen.santos@fi.upm.es"
EXAMPLE_PROF2_EMAIL = "luis.martin@fi.upm.es"

MODEL_PROFILES = {
    EMAIL_WRITER_PROFILE: {
        "model_path": "Qwen/Qwen3-8B-FP8",
        "is_vision_model": False,
        "system_prompt": "You are a concise, professional corporate email assistant.",
        "prompt_template": (
            "You are taking the role of {my_name}, {my_description}. You are reading an email sent to you.\n"
            "Your task is to write a professional, concise reply.\n"
            "IMPORTANT: Respond in the SAME LANGUAGE as the Original Email (e.g., Spanish -> Spanish, English -> English).\n\n"
            "### INSTRUCTIONS:\n"
            f"1. If you have enough information to reply, wrap your response in {MESSAGE_OPENING_TAG}...{MESSAGE_CLOSING_TAG} tags.\n"
            f"2. If you lack context or cannot reply, output {NO_MESSAGE_OPENING_TAG}I cannot reply because...{NO_MESSAGE_CLOSING_TAG}.\n"
            "3. Do not include subject lines or greetings/signatures outside the tags.\n\n"
            "### SAMPLE RESPONSE:\n"
            "Context from knowledge base:\n"
            "---\n"
            "Calendar available at 3 PM.\n"
            "---\n"
            "Conversation so far (oldest to newest):\n"
            "---\n"
            "(no prior messages)\n"
            "---\n"
            "Current email to answer:\n"
            "---\n"
            "Subject: Meeting request\n"
            "From: john.doe@example.com\n"
            "Body: Can we meet at 3 PM?\n"
            "---\n"
            "Output:\n"
            f"{MESSAGE_OPENING_TAG}\n"
            "Hi John,\n\n"
            "3 PM works for me. See you then.\n\n"
            "Best,\n"
            "{my_name}\n"
            f"{MESSAGE_CLOSING_TAG}\n\n"
            "### ACTUAL TASK:\n"
            "Context from knowledge base:\n"
            "---\n"
            "{rag_context}\n"
            "---\n"
            "Conversation so far (oldest to newest):\n"
            "---\n"
            "{thread_context}\n"
            "---\n"
            "Current email to answer:\n"
            "---\n"
            "Subject:\n"
            "{subject}\n"
            "From:\n"
            "{sender}\n"
            "Body:\n"
            "{body}\n"
            "---\n"
            "Output:\n"
        ),
        "max_context_tokens": 32768,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "use_flash_attention_2": USE_FLASH_ATTENTION_IMAGE,
        "enable_thinking": True,
        "return_prompt_text": True
    },
    THREAD_GROUPER_PROFILE: {
        "model_path": "Qwen/Qwen3-8B-FP8",
        "is_vision_model": False,
        "system_prompt": (
            "You are an expert email thread reconstruction assistant."
        ),
        "max_context_tokens": 32768,
        "production_task_description_start": (
            "You have received {inbox_count} inbox emails and {sent_count} sent emails "
            "from {my_name} ({my_description}) INBOX and SENT folders. "
            "Your task is to group all emails into threads and remove quoted text carefully. "
            f"All email addresses that belong to {{my_name}} are: {{my_email_addresses}}. "
            "Each email includes: 'id', 'threadID', 'from', 'to', 'date', 'subject', 'body'. "
            "'id' is the email server id and is not the thread id. "
            "'threadID' is not part of the original communication (and must not be written in the output XML). "
            "It is a weak hint produced by an automated subject-based grouping and can be wrong. "
            "Your task is to output XML, reconstructing the threads and removing quoted text when it is already part of another email."
        ),
        "production_example": (
            "Input emails:\n"
            "Inbox:\n"
            f"{{'id': b'440', 'threadID': 1, 'from': '{EXAMPLE_STUDENT_NAME} <{EXAMPLE_STUDENT_EMAIL}> undefined', 'to': '{EXAMPLE_PROF1_EMAIL}, \"[MUIA] {DIRECTOR_NAME}\" <{DIRECTOR_EMAIL}> undefined, {EXAMPLE_PROF2_EMAIL} undefined undefined undefined', 'date': datetime.datetime(2020, 5, 4, 9, 8, 59, tzinfo=datetime.timezone.utc), 'subject': 'Erasmus', 'body': \"Good morning. I am a student from Italy currently studying at my university. I would like to join the Erasmus program at Universidad Politécnica de Madrid. Would it be possible to take courses from both the MSc in Artificial Intelligence and the Máster Universitario en Automática y Robótica, or should I choose just one? Thank you\"}}\n"
            f"{{'id': b'448', 'threadID': 3, 'from': '{EXAMPLE_STAFF_NAME} <{EXAMPLE_STAFF_EMAIL}>', 'to': '\"[MUIA] {DIRECTOR_NAME}\" <{DIRECTOR_EMAIL}> undefined', 'date': datetime.datetime(2020, 6, 11, 13, 32, 15, tzinfo=datetime.timezone.utc), 'subject': 'Fwd: Fichero egresados', 'body': \"Hola, {DIRECTOR_NAME}. Te reenvío aquí lo último que hice yo. Es de mayo de 2020. No sé si {EXAMPLE_COLLEAGUE_NAME} hizo alguno posterior. ¿Te sirve? {EXAMPLE_STAFF_NAME}\"}}\n"
            f"{{'id': b'432', 'threadID': 4, 'from': '{EXAMPLE_STUDENT_REP_NAME} <{EXAMPLE_STUDENT_REP_EMAIL}>', 'to': '\"[MUIA] {DIRECTOR_NAME}\" <{DIRECTOR_EMAIL}> undefined', 'date': datetime.datetime(2020, 6, 12, 8, 55, 27, tzinfo=datetime.timezone.utc), 'subject': 'Orla/graduación de alumnos del máster en IA', 'body': \"Hola {DIRECTOR_NAME}, como delegado del máster en Inteligencia Artificial, me gustaría trasladarte la consulta de varios alumnos acerca de si se va a hacer orla / acto de graduación para los estudiantes del máster, o por si el contrario corre bajo nuestra cuenta hacerlo. ¡Un saludo! {EXAMPLE_STUDENT_REP_NAME} Máster Universitario en Inteligencia Artificial\"}}\n"
            "Sent:\n"
            f"{{'id': b'441', 'threadID': 1, 'from': '\"[MUIA] {DIRECTOR_NAME}\" <{DIRECTOR_EMAIL}> undefined', 'to': '{EXAMPLE_STUDENT_NAME} <{EXAMPLE_STUDENT_EMAIL}> undefined', 'date': datetime.datetime(2020, 5, 4, 11, 16, 0, tzinfo=datetime.timezone.utc), 'subject': 'Erasmus', 'body': \"I don't know about the other Degree; it is probably taught in another School, so that it can be tricky for you to attend both. Besides, I'm not sure you can follow courses from different School at the same time during your Erasmus. Please ask orex@fi.upm.es: they are in charge of the Erasmus Programme at Escuela Técnica Superior de Ingenieros Informáticos. regards -- {DIRECTOR_NAME} Coordinador del Máster Universitario en Inteligencia Artificial Universidad Politécnica de Madrid Escuela Técnica Superior de Ingenieros Informáticos Campus de Montegancedo S/N, 28660, Boadilla del Monte, Madrid SPAIN Departamento de Inteligencia Artificial {DEPARTMENT_PHONE} {DIRECTOR_EMAIL}\"}}\n"
            f"{{'id': b'3278', 'threadID': 2, 'from': '\"[MUIA] {DIRECTOR_NAME}\" <{DIRECTOR_EMAIL}> undefined', 'to': '{EXAMPLE_STAFF_EMAIL}', 'date': datetime.datetime(2020, 6, 10, 10, 39, 0, tzinfo=datetime.timezone.utc), 'subject': 'Estudio de egresados', 'body': \"Hola {EXAMPLE_STAFF_NAME}. Estoy con el informe de titulación. Cuál es el último informe de egresados que tenemos? Hay unos cuantos datos que estaría bien actualizar, como los ex-alumnos que están en el extranjero, los que están realizando un doctorado, etc. Además, dice aquí: El último estudio sobre empleabilidad realizado por el Observatorio Académico sobre titulaciones de postgrado de la ETSIINF es una encuesta fue realizada en el curso 2019/20. Tenemos uno más reciente? Gracias! -- {DIRECTOR_NAME} Coordinador del Máster Universitario en Inteligencia Artificial Universidad Politécnica de Madrid Escuela Técnica Superior de Ingenieros Informáticos Campus de Montegancedo S/N, 28660, Boadilla del Monte, Madrid SPAIN Departamento de Inteligencia Artificial {DEPARTMENT_PHONE} {DIRECTOR_EMAIL}\"}}\n"
            f"{{'id': b'3279', 'threadID': 3, 'from': '\"[MUIA] {DIRECTOR_NAME}\" <{DIRECTOR_EMAIL}> undefined', 'to': '{EXAMPLE_STAFF_EMAIL}', 'date': datetime.datetime(2020, 6, 11, 15, 0, 0, tzinfo=datetime.timezone.utc), 'subject': 'Fwd: Fichero egresados', 'body': \"Gracias {EXAMPLE_STAFF_NAME}. A ver lo que puedo sacar de aquí. Un saludo -- {DIRECTOR_NAME} Coordinador del Máster Universitario en Inteligencia Artificial Universidad Politécnica de Madrid Escuela Técnica Superior de Ingenieros Informáticos Campus de Montegancedo S/N, 28660, Boadilla del Monte, Madrid SPAIN Departamento de Inteligencia Artificial {DEPARTMENT_PHONE} {DIRECTOR_EMAIL}\"}}\n"
            f"{{'id': b'433', 'threadID': 4, 'from': '\"[MUIA] {DIRECTOR_NAME}\" <{DIRECTOR_EMAIL}> undefined', 'to': '{EXAMPLE_DIRECTOR_PEER_EMAIL}', 'date': datetime.datetime(2020, 6, 12, 9, 10, 0, tzinfo=datetime.timezone.utc), 'subject': 'Fwd: Orla/graduación de alumnos del máster en IA', 'body': \"Hola {EXAMPLE_DIRECTOR_PEER_NAME}. Este mensaje me pilla tan de sorpresa que no sé ni cómo empezar a contestar. No se supone que acabamos de estar en el Wanda? A ver si tú lo sabes interpretar...\"}}\n"
            "Output:\n"
            f"{THREAD_OPENING_TAG}\n"
            f"{THREAD_MESSAGE_OPENING_TAG}\n"
            f"{THREAD_FROM_OPENING_TAG}{EXAMPLE_STUDENT_NAME} <{EXAMPLE_STUDENT_EMAIL}> undefined{THREAD_FROM_CLOSING_TAG}\n"
            f"{THREAD_TO_OPENING_TAG}{EXAMPLE_PROF1_EMAIL}, \"[MUIA] {DIRECTOR_NAME}\" <{DIRECTOR_EMAIL}> undefined, {EXAMPLE_PROF2_EMAIL} undefined undefined undefined{THREAD_TO_CLOSING_TAG}\n"
            f"{THREAD_SUBJECT_OPENING_TAG}Erasmus{THREAD_SUBJECT_CLOSING_TAG}\n"
            f"{THREAD_BODY_OPENING_TAG}Good morning. I am a student from Italy currently studying at my university. I would like to join the Erasmus program at Universidad Politécnica de Madrid. Would it be possible to take courses from both the MSc in Artificial Intelligence and the Máster Universitario en Automática y Robótica, or should I choose just one? Thank you{THREAD_BODY_CLOSING_TAG}\n"
            f"{THREAD_MESSAGE_CLOSING_TAG}\n"
            f"{THREAD_MESSAGE_OPENING_TAG}\n"
            f"{THREAD_FROM_OPENING_TAG}\"[MUIA] {DIRECTOR_NAME}\" <{DIRECTOR_EMAIL}> undefined{THREAD_FROM_CLOSING_TAG}\n"
            f"{THREAD_TO_OPENING_TAG}{EXAMPLE_STUDENT_NAME} <{EXAMPLE_STUDENT_EMAIL}> undefined{THREAD_TO_CLOSING_TAG}\n"
            f"{THREAD_SUBJECT_OPENING_TAG}Erasmus{THREAD_SUBJECT_CLOSING_TAG}\n"
            f"{THREAD_BODY_OPENING_TAG}I don't know about the other Degree; it is probably taught in another School, so that it can be tricky for you to attend both. Besides, I'm not sure you can follow courses from different School at the same time during your Erasmus. Please ask orex@fi.upm.es: they are in charge of the Erasmus Programme at Escuela Técnica Superior de Ingenieros Informáticos. regards -- {DIRECTOR_NAME} Coordinador del Máster Universitario en Inteligencia Artificial Universidad Politécnica de Madrid Escuela Técnica Superior de Ingenieros Informáticos Campus de Montegancedo S/N, 28660, Boadilla del Monte, Madrid SPAIN Departamento de Inteligencia Artificial {DEPARTMENT_PHONE} {DIRECTOR_EMAIL}{THREAD_BODY_CLOSING_TAG}\n"
            f"{THREAD_MESSAGE_CLOSING_TAG}\n"
            f"{THREAD_CLOSING_TAG}\n"
            f"{THREAD_OPENING_TAG}\n"
            f"{THREAD_MESSAGE_OPENING_TAG}\n"
            f"{THREAD_FROM_OPENING_TAG}\"[MUIA] {DIRECTOR_NAME}\" <{DIRECTOR_EMAIL}> undefined{THREAD_FROM_CLOSING_TAG}\n"
            f"{THREAD_TO_OPENING_TAG}{EXAMPLE_STAFF_EMAIL}{THREAD_TO_CLOSING_TAG}\n"
            f"{THREAD_SUBJECT_OPENING_TAG}Estudio de egresados{THREAD_SUBJECT_CLOSING_TAG}\n"
            f"{THREAD_BODY_OPENING_TAG}Hola {EXAMPLE_STAFF_NAME}. Estoy con el informe de titulación. Cuál es el último informe de egresados que tenemos? Hay unos cuantos datos que estaría bien actualizar, como los ex-alumnos que están en el extranjero, los que están realizando un doctorado, etc. Además, dice aquí: El último estudio sobre empleabilidad realizado por el Observatorio Académico sobre titulaciones de postgrado de la ETSIINF es una encuesta fue realizada en el curso 2019/20. Tenemos uno más reciente? Gracias! -- {DIRECTOR_NAME} Coordinador del Máster Universitario en Inteligencia Artificial Universidad Politécnica de Madrid Escuela Técnica Superior de Ingenieros Informáticos Campus de Montegancedo S/N, 28660, Boadilla del Monte, Madrid SPAIN Departamento de Inteligencia Artificial {DEPARTMENT_PHONE} {DIRECTOR_EMAIL}{THREAD_BODY_CLOSING_TAG}\n"
            f"{THREAD_MESSAGE_CLOSING_TAG}\n"
            f"{THREAD_MESSAGE_OPENING_TAG}\n"
            f"{THREAD_FROM_OPENING_TAG}{EXAMPLE_STAFF_NAME} <{EXAMPLE_STAFF_EMAIL}>{THREAD_FROM_CLOSING_TAG}\n"
            f"{THREAD_TO_OPENING_TAG}\"[MUIA] {DIRECTOR_NAME}\" <{DIRECTOR_EMAIL}> undefined{THREAD_TO_CLOSING_TAG}\n"
            f"{THREAD_SUBJECT_OPENING_TAG}Fwd: Fichero egresados{THREAD_SUBJECT_CLOSING_TAG}\n"
            f"{THREAD_BODY_OPENING_TAG}Hola, {DIRECTOR_NAME}. Te reenvío aquí lo último que hice yo. Es de mayo de 2020. No sé si {EXAMPLE_COLLEAGUE_NAME} hizo alguno posterior. ¿Te sirve? {EXAMPLE_STAFF_NAME}{THREAD_BODY_CLOSING_TAG}\n"
            f"{THREAD_MESSAGE_CLOSING_TAG}\n"
            f"{THREAD_MESSAGE_OPENING_TAG}\n"
            f"{THREAD_FROM_OPENING_TAG}\"[MUIA] {DIRECTOR_NAME}\" <{DIRECTOR_EMAIL}> undefined{THREAD_FROM_CLOSING_TAG}\n"
            f"{THREAD_TO_OPENING_TAG}{EXAMPLE_STAFF_EMAIL}{THREAD_TO_CLOSING_TAG}\n"
            f"{THREAD_SUBJECT_OPENING_TAG}Fwd: Fichero egresados{THREAD_SUBJECT_CLOSING_TAG}\n"
            f"{THREAD_BODY_OPENING_TAG}Gracias {EXAMPLE_STAFF_NAME}. A ver lo que puedo sacar de aquí. Un saludo -- {DIRECTOR_NAME} Coordinador del Máster Universitario en Inteligencia Artificial Universidad Politécnica de Madrid Escuela Técnica Superior de Ingenieros Informáticos Campus de Montegancedo S/N, 28660, Boadilla del Monte, Madrid SPAIN Departamento de Inteligencia Artificial {DEPARTMENT_PHONE} {DIRECTOR_EMAIL}{THREAD_BODY_CLOSING_TAG}\n"
            f"{THREAD_MESSAGE_CLOSING_TAG}\n"
            f"{THREAD_CLOSING_TAG}\n"
            f"{THREAD_OPENING_TAG}\n"
            f"{THREAD_MESSAGE_OPENING_TAG}\n"
            f"{THREAD_FROM_OPENING_TAG}{EXAMPLE_STUDENT_REP_NAME} <{EXAMPLE_STUDENT_REP_EMAIL}>{THREAD_FROM_CLOSING_TAG}\n"
            f"{THREAD_TO_OPENING_TAG}\"[MUIA] {DIRECTOR_NAME}\" <{DIRECTOR_EMAIL}> undefined{THREAD_TO_CLOSING_TAG}\n"
            f"{THREAD_SUBJECT_OPENING_TAG}Orla/graduación de alumnos del máster en IA{THREAD_SUBJECT_CLOSING_TAG}\n"
            f"{THREAD_BODY_OPENING_TAG}Hola {DIRECTOR_NAME}, como delegado del máster en Inteligencia Artificial, me gustaría trasladarte la consulta de varios alumnos acerca de si se va a hacer orla / acto de graduación para los estudiantes del máster, o por si el contrario corre bajo nuestra cuenta hacerlo. ¡Un saludo! {EXAMPLE_STUDENT_REP_NAME} Máster Universitario en Inteligencia Artificial{THREAD_BODY_CLOSING_TAG}\n"
            f"{THREAD_MESSAGE_CLOSING_TAG}\n"
            f"{THREAD_MESSAGE_OPENING_TAG}\n"
            f"{THREAD_FROM_OPENING_TAG}\"[MUIA] {DIRECTOR_NAME}\" <{DIRECTOR_EMAIL}> undefined{THREAD_FROM_CLOSING_TAG}\n"
            f"{THREAD_TO_OPENING_TAG}{EXAMPLE_DIRECTOR_PEER_EMAIL}{THREAD_TO_CLOSING_TAG}\n"
            f"{THREAD_SUBJECT_OPENING_TAG}Fwd: Orla/graduación de alumnos del máster en IA{THREAD_SUBJECT_CLOSING_TAG}\n"
            f"{THREAD_BODY_OPENING_TAG}Hola {EXAMPLE_DIRECTOR_PEER_NAME}. Este mensaje me pilla tan de sorpresa que no sé ni cómo empezar a contestar. No se supone que acabamos de estar en el Wanda? A ver si tú lo sabes interpretar...{THREAD_BODY_CLOSING_TAG}\n"
            f"{THREAD_MESSAGE_CLOSING_TAG}\n"
            f"{THREAD_CLOSING_TAG}"
        ),
        "prompt_template": (
            "{task_description_start}\n\n"
            "### RULES:\n"
            "1. Preserve chronological order within each thread.\n"
            "2. Remove quoted text only when the quoted content appears elsewhere in the input as the same text "
            "with fewer or no quote markers. Keep the least-quoted instance.\n"
            "   - Example: if B contains \"> A\" and A appears elsewhere unquoted, remove \"> A\" from B.\n"
            "   - Example: if B contains only \"> A\" and A does not appear elsewhere, keep it.\n"
            "   - Example: if B has \"> A\" and C has \"> B\\n> A\", keep \"> A\" in B and remove the quoted part from C.\n"
            "   Reply headers such as 'En ... escribió:', 'On ... wrote:', 'De/Enviado/Para/Asunto' are a few indicators of quoted blocks.\n"
            f"3. Keep only the cleaned body inside {THREAD_BODY_OPENING_TAG}...{THREAD_BODY_CLOSING_TAG} tags.\n"
            "4. Do not hallucinate or add new messages.\n"
            "5. Output ONLY the thread XML, nothing else.\n\n"
            "### OUTPUT FORMAT:\n"
            f"{THREAD_OPENING_TAG}\n"
            f"{THREAD_MESSAGE_OPENING_TAG}\n"
            f"{THREAD_FROM_OPENING_TAG}...{THREAD_FROM_CLOSING_TAG}\n"
            f"{THREAD_TO_OPENING_TAG}...{THREAD_TO_CLOSING_TAG}\n"
            f"{THREAD_SUBJECT_OPENING_TAG}...{THREAD_SUBJECT_CLOSING_TAG}\n"
            f"{THREAD_BODY_OPENING_TAG}...{THREAD_BODY_CLOSING_TAG}\n"
            f"{THREAD_MESSAGE_CLOSING_TAG}\n"
            f"{THREAD_MESSAGE_OPENING_TAG}\n"
            "...\n"
            f"{THREAD_MESSAGE_CLOSING_TAG}\n"
            f"{THREAD_CLOSING_TAG}\n"
            f"{THREAD_OPENING_TAG}\n"
            "...\n"
            f"{THREAD_CLOSING_TAG}\n\n"
            "### SAMPLE RESPONSE:\n"
            "---\n"
            "{example_message}\n"
            "---\n\n"
            "### ACTUAL TASK:\n"
            "---\n"
            "Input emails:\n"
            "Inbox:\n"
            "{inbox_emails}\n"
            "Sent:\n"
            "{sent_emails}\n"
            "Output:\n"
        ),
        "max_new_tokens": 8192,
        "temperature": 0.3,
        "top_p": 0.8,
        "top_k": 20,
        "use_flash_attention_2": USE_FLASH_ATTENTION_IMAGE,
        "enable_thinking": True,
        "return_prompt_text": True
    },
    DATA_CLEANER_PROFILE: {
        "model_path": "Qwen/Qwen3-8B-FP8",
        "is_vision_model": False,
        "system_prompt": (
            "You are an expert Knowledge Curator for a RAG system. Your task is to transform the following raw text into a high-quality, English-language knowledge base entry.\n\n"
            "### CONTEXTUAL INFORMATION & RULES:\n"
            "1. **Origin**: This text was crawled for the MUIA (Master Universitario en Inteligencia Artificial) at Universidad Politécnica de Madrid (UPM).\n"
            "2. **Scope**: Content generally falls into two categories: specific MUIA details (subjects, seminars) or general UPM info (enrollment, other degrees). **Crucial**: You must explicitly note what the information belongs to (MUIA vs General UPM) in your outputs to preserve context.\n"
            "3. **Time Awareness**: You will receive the current date. Some resources may be outdated; if you detect dates in the text, note them and warn if content might be obsolete.\n"
            "4. **Context Inputs**: You will receive 'Page History' containing abstracts and summaries from previous chunks. Each entry includes its **Chunk Index** so you can determine its position in the document. **Note**: This history may be non-consecutive (e.g., Chunks 0-4 to provide you with beginning-of-document context followed by Chunks 35-40 to provide you with latest-chunks context).\n"
            "5. **Continuity**: You will receive the 'Previous Chunk Cleaned Text' to ensure grammatical continuity with the current input (for example, a question or link might have been cut off and this may give you the chance to reconstruct it).\n\n"
            "### OUTPUT INSTRUCTIONS:\n"
            "1. **Translate**: Translate ALL content into English, regardless of the source language.\n"
            f"2. **Clean & Format**: Fix broken sentences. Aggressively remove table-of-contents dots (e.g., ......15), page numbers, and residual formatting noise. Enclose the final cleaned text in {CLEANED_TEXT_OPENING_TAG} tags.\n"
            f"3. **Abstract**: Provide a concise 1-sentence overview inside {ABSTRACT_OPENING_TAG} tags. Mention if it applies to MUIA or UPM generally.\n"
            f"4. **Summary**: Provide a detailed, reorganized summary of the key facts inside {SUMMARY_OPENING_TAG} tags. Explicitly state if the info is MUIA-specific or general UPM.\n"
            f"5. **Augment**: Generate up to 20 Q&A pairs inside {QUESTIONS_OPENING_TAG} tags, using {QUESTION_OPENING_TAG} and {ANSWER_OPENING_TAG} for each pair. Extract every possible fact. **IMPORTANT**: Answers must contain specific details (names, dates, EXACT URLs) and explicitly mention if they refer to MUIA or UPM to ensure standalone context.\n\n"
            "### Q&A QUALITY GUIDELINES:\n"
            "**Negative Examples (Avoid These Mistakes):**\n\n"
            "Example 1 (Irrelevance):\n"
            f"{QUESTION_OPENING_TAG}What is the note regarding the currency and language of the ticket pricing page?{QUESTION_CLOSING_TAG}\n"
            f"{ANSWER_OPENING_TAG}The ticket pricing page is in Spanish.{ANSWER_CLOSING_TAG}\n"
            "Critique: Not relevant. Do not include meta-commentary about the page format.\n\n"
            "Example 2 (Outdated Information & Meta-pairs):\n"
            f"{QUESTION_OPENING_TAG}When was the MLAS event held that is referenced in the text?{QUESTION_CLOSING_TAG}\n"
            f"{ANSWER_OPENING_TAG}The 17th MLAS event was held in 2025, and the information is outdated as of 2026.{ANSWER_CLOSING_TAG}\n"
            f"{QUESTION_OPENING_TAG}What are the ticket prices for buses 591, 865, and the light rail?{QUESTION_CLOSING_TAG}\n"
            f"{ANSWER_OPENING_TAG}A single trip costs 2 euros, while a 10-ride ticket costs 12.20 euros.{ANSWER_CLOSING_TAG}\n"
            "Critique: Good intention, but the fact that information may be outdated should be embedded in each specific pair, not as a separate meta-pair.\n"
            "**Better Version:**\n"
            f"{QUESTION_OPENING_TAG}What are the ticket prices for buses 591, 865, and the light rail?{QUESTION_CLOSING_TAG}\n"
            f"{ANSWER_OPENING_TAG}As of 2025 (17th MLAS event), a single trip costs 2 euros, while a 10-ride ticket costs 12.20 euros.{ANSWER_CLOSING_TAG}\n\n"
            "Example 3 (Missing Context):\n"
            f"{QUESTION_OPENING_TAG}What is the recommended route after exiting the light-rail station?{QUESTION_CLOSING_TAG}\n"
            f"{ANSWER_OPENING_TAG}Participants should turn right and walk down Avda. Montepríncipe, as indicated by the Google Maps link.{ANSWER_CLOSING_TAG}\n"
            "Critique: Participants of what? Recommended route to where? Each pair must be standalone.\n"
            "**Better Version:**\n"
            f"{QUESTION_OPENING_TAG}What is the recommended route for MLAS participants after exiting the light-rail station?{QUESTION_CLOSING_TAG}\n"
            f"{ANSWER_OPENING_TAG}MLAS participants should turn right after exiting the light-rail station and walk down Avda. Montepríncipe.{ANSWER_CLOSING_TAG}\n\n"
            "Example 4 (Ambiguous Entities):\n"
            f"{QUESTION_OPENING_TAG}Which metro lines connect to the campus via bus 865?{QUESTION_CLOSING_TAG}\n"
            f"{ANSWER_OPENING_TAG}Bus 865 connects to the campus from Moncloa (metro lines 6 and 3).{ANSWER_CLOSING_TAG}\n"
            "Critique: Which campus? Questions and answers must be specific.\n"
            "**Better Version:**\n"
            f"{QUESTION_OPENING_TAG}Which metro lines connect to the Montegancedo Campus via bus 865?{QUESTION_CLOSING_TAG}\n"
            f"{ANSWER_OPENING_TAG}Bus 865 connects to the Montegancedo Campus from Moncloa (metro lines 6 and 3).{ANSWER_CLOSING_TAG}\n\n"
            "Example 5 (False Specificity):\n"
            f"{QUESTION_OPENING_TAG}Which group has collaborated with companies like Progenika Biopharma and Panda Security?{QUESTION_CLOSING_TAG}\n"
            f"{ANSWER_OPENING_TAG}The Computational Intelligence Group (CIG) has collaborated with companies such as Progenika Biopharma and Panda Security.{ANSWER_CLOSING_TAG}\n"
            "Critique: Asking 'Which group...' implies CIG is the only one. It is safer to invert the question.\n"
            "### SAMPLE RESPONSE:\n"
            "**Page History Context**:\n"
            "[list of previous abstracts/summaries with chunk indices...]\n"
            "**Previous Chunk Cleaned Text**:\n"
            "[cleaned English text from previous chunk would be here...]\n"
            "**Input Text**:\n"
            "-----\n\n"
            "Reconocido en el ranking de mejores másteres de España en informática especializada, publicado por el periódico El Mundo, entre los tres mejores durante catorce ediciones.\n"
            "-----\n"
            "#### 0\n\n"
            "### Año académico\n\n"
            "#### 0\n\n"
            "### Créditos ECTS\n\n"
            "#### 0\n\n"
            "### Plazas\n\n"
            "Un máster con\n\n"
            "orientación\n\n"
            "investigadora\n"
            "-----\n"
            "Organizado e impartido por el [Departamento de Inteligencia Artificial (DIA)](https://dia.fi.upm.es/), forma parte de la oferta de postgrado de la Escuela Técnica Superior de Ingenieros Informáticos de la Universidad Politécnica de Madrid. Con una orientación investigadora, comenzó a impartirse en el curso académico 2010/11, y proporciona una formación de calidad en diversos campos de investigación actuales en inteligencia artificial.\n\n"
            "**Output**:\n"
            f"{ABSTRACT_OPENING_TAG}\n"
            "Overview of the research-oriented Master in Artificial Intelligence (MUIA) and its national recognition.\n"
            f"{ABSTRACT_CLOSING_TAG}\n"
            f"{SUMMARY_OPENING_TAG}\n"
            "The Master in Artificial Intelligence (MUIA) is a research-oriented program organized by the Department of Artificial Intelligence (DIA) at the Technical University of Madrid (UPM). It has been consistently ranked as one of the top three masters in specialized informatics in Spain. It began in the 2010/11 academic year.\n"
            f"{SUMMARY_CLOSING_TAG}\n"
            f"{CLEANED_TEXT_OPENING_TAG}\n"
            "Recognized in the ranking of the best masters in Spain in specialized informatics, published by the newspaper El Mundo, among the top three for fourteen editions.\n\n"
            "A master's degree with a research orientation.\n\n"
            "Organized and taught by the [Department of Artificial Intelligence (DIA)](https://dia.fi.upm.es/), it is part of the postgraduate offer of the School of Computer Engineering of the Technical University of Madrid (UPM). With a research orientation, it began in the 2010/11 academic year, and provides quality training in various current research fields in artificial intelligence.\n"
            f"{CLEANED_TEXT_CLOSING_TAG}\n"
            f"{QUESTIONS_OPENING_TAG}\n"
            f"{QUESTION_OPENING_TAG}What is the primary orientation of the MUIA master's degree?{QUESTION_CLOSING_TAG}\n"
            f"{ANSWER_OPENING_TAG}The MUIA program is research-oriented.{ANSWER_CLOSING_TAG}\n"
            f"{QUESTION_OPENING_TAG}Which specific department organizes the MUIA program?{QUESTION_CLOSING_TAG}\n"
            f"{ANSWER_OPENING_TAG}The Department of Artificial Intelligence (DIA) at UPM.{ANSWER_CLOSING_TAG}\n"
            f"{QUESTION_OPENING_TAG}Where is the MUIA master's degree taught?{QUESTION_CLOSING_TAG}\n"
            f"{ANSWER_OPENING_TAG}It is taught at the School of Computer Engineering of the Technical University of Madrid (UPM).{ANSWER_CLOSING_TAG}\n"
            f"{QUESTION_OPENING_TAG}How has the MUIA program been recognized in national rankings?{QUESTION_CLOSING_TAG}\n"
            f"{ANSWER_OPENING_TAG}It has been recognized as one of the best masters in specialized informatics in Spain by El Mundo newspaper (among the top three for fourteen editions).{ANSWER_CLOSING_TAG}\n"
            f"{QUESTION_OPENING_TAG}When did the MUIA program begin?{QUESTION_CLOSING_TAG}\n"
            f"{ANSWER_OPENING_TAG}It began in the 2010/11 academic year.{ANSWER_CLOSING_TAG}\n"
            f"{QUESTION_OPENING_TAG}Where can the Department of Artificial Intelligence (DIA) be accessed at?{QUESTION_CLOSING_TAG}\n"
            f"{ANSWER_OPENING_TAG}It can be accessed at https://dia.fi.upm.es/{ANSWER_CLOSING_TAG}\n"
            f"{QUESTIONS_CLOSING_TAG}"
        ),
        "prompt_template": (
            "### CURRENT TASK:\n"
            "**Current Date**: {datetime}\n"
            "**Page History Context**:\n{page_history_context}\n"
            "**Previous Chunk Cleaned Text**:\n{previous_chunk_context}\n"
            "**Input Text**:\n{text}\n\n"
            "**Output**:\n"
        ),
        "max_chunk_size": 1024,
        "max_new_tokens": 8192,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "use_flash_attention_2": USE_FLASH_ATTENTION_IMAGE,
        "enable_thinking": True,
        "return_prompt_text": True
    }
}

_image_flash_attention = (
    modal.Image.from_registry(FLASH_ATTENTION_IMAGE)
    .run_commands(FLASH_ATTENTION_RUN_COMMANDS)
    .pip_install(
        FLASH_ATTENTION_TORCH_VERSION,
        *COMMON_PACKAGES
        )
    .pip_install(FLASH_ATTENTION_RELEASE)
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    # Run `image.add_local_*` commands last in your image build to avoid rebuilding images with every local file change.
    .add_local_python_source("config", "helpers")
)
_image_no_flash_attention = (
    modal.Image.debian_slim(python_version=NO_FLASH_ATTENTION_PYTHON_VERSION)
    .pip_install(
        NO_FLASH_ATTENTION_TORCH_VERSION,
        *COMMON_PACKAGES
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    # Run `image.add_local_*` commands last in your image build to avoid rebuilding images with every local file change.
    .add_local_python_source("config", "helpers")
)
image = _image_flash_attention if USE_FLASH_ATTENTION_IMAGE else _image_no_flash_attention
