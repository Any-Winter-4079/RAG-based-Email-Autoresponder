DATASET_PATH = "data/messages.csv"
TRAIN_SPLIT_PATH = "data/messages_train.csv"
MESSAGES_WITH_THREADS_DATASET_PATH = "data/messages_with_threads.csv"
UPM_DOMAINS = [
    "upm.es",
    "fi.upm.es",
    "etsii.upm.es",
    "relay.fi.upm.es"
]
AUTOMATED_OUTBOUND_TEMPLATES = [
    "Nos es grato comunicarte que has sido admitido",
    "Nos es grato comunicarte que has sido admitida",
    "le comunicamos que le hemos dado un acceso condicionado a la finalización de sus estudios",
    "Como sabrás, tu admisión está condicionada a la realización de",
    "el número de alumnos que se preinscriben es mucho mayor que el número de plazas disponibles",
    "En primer lugar, deseo expresarte mi pesar por haber tenido que denegar"
]
PRE_ENROLLMENT_TEMPLATES = [
    "El documento que recibe a continuación es la solicitud de preinscripción al Master/Doctorado que usted coordina/gestiona."
]
REMOVE_INTERNAL_UPM_MESSAGES = True
TRAIN_SPLIT_PCT = 0.5
DEV_SPLIT_PCT = 0.25
VAL_SPLIT_PCT = 0.15
TEST_SPLIT_PCT = 0.1
