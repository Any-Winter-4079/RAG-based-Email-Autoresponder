import modal

START_URL = "https://muia.dia.fi.upm.es/es/"
MAX_DEPTH = 3
MAX_LINKS_PER_PAGE = 30
EXCLUDED_URLS = [
    "https://web.upm.es/hrs4r",
    "https://sede.upm.es/",
    "https://www.upm.es/Portal_inv"
]
GSFS_BASE_URL = "https://www.upm.es/gsfs/"
ALLOWED_GSFS_URLS = [
    "https://www.upm.es/gsfs/SFS18800",
    "https://www.upm.es/gsfs/SFS04242"
]
ADDITIONAL_URLS = [
    # becas, ayudas y premios
    "https://www.upm.es/Estudiantes/BecasAyudasPremios/AyudasAlumnosDobleTitulacion",
    "https://www.upm.es/gsfs/SFS18800",
    # matrícula
    "https://www.upm.es/Estudiantes/Estudios_Titulaciones/Estudios_Master/Preguntas_frecuentes",
    "https://www.upm.es/Estudiantes/OrdenacionAcademica/Matricula/PreguntasFrecuentes&vgnextchannel=15b77da9f889c410VgnVCM10000009c7648aRCRD",
    "https://www.upm.es/Estudiantes/Estudios_Titulaciones/Estudios_Master/Calendario",
    "https://www.upm.es/gsfs/SFS04242",
    "https://www.upm.es/Estudiantes/Estudios_Titulaciones/Estudios_Master/?fmt=detail&id=CON06502",
    "https://www.upm.es/Estudiantes/Estudios_Titulaciones/Estudios_Master/Matricula",
    "https://www.upm.es/Estudiantes/OrdenacionAcademica/Matricula",
    "https://www.upm.es/Estudiantes/Estudios_Titulaciones/Estudios_Master/Admision/UniversidadesEEES",
    "https://www.upm.es/Estudiantes/Estudios_Titulaciones/Estudios_Master/Admision/UniversidadesNoEEES",
    "https://www.upm.es/Estudiantes/Estudios_Titulaciones/Estudios_Master/Admision/UniversidadesEspanolas",
    "https://www.upm.es/Estudiantes/Estudios_Titulaciones/Estudios_Master/Programas?id=10.9&fmt=detail",
]

JINA_FETCH_TIMEOUT = 30 # seconds
MODAL_TIMEOUT = 86400 # seconds

CHUNK_OVERLAP = 0 # if not decoder chunking (e.g., if SentenceSplitter)

CRAWL_MINUTES = 0
CRAWL_HOUR = 9
CRAWL_DAY = 10
CRAWL_MONTH = 9

REUSE_CRAWL = True
REUSE_CRAWL_PAST_CURRENT_YEAR = False
REUSE_TIMESTAMP = "20260203_161009" # reuse this within the same year if REUSE_CRAWL, forever if REUSE_CRAWL_PAST_CURRENT_YEAR

VOLUME_NAME = "muia-rag-volume"
VOLUME_PATH = "/root/volume"
QDRANT_PATH = f"{VOLUME_PATH}/qdrant"

FILE_START = "crawl_"
RAW_PATH = f"{VOLUME_PATH}/raw"
MANUALLY_CLEANED_PATH = f"{VOLUME_PATH}/manually_cleaned"
LM_CLEANED_PATH = f"{VOLUME_PATH}/lm_cleaned"

RAW_CHUNKS_PATH = f"{VOLUME_PATH}/raw_chunks"
MANUALLY_CLEANED_CHUNKS_PATH = f"{VOLUME_PATH}/manually_cleaned_chunks"
LM_CLEANED_TEXT_CHUNKS_PATH = f"{VOLUME_PATH}/lm_cleaned_text_chunks"
LM_ABSTRACT_CHUNKS_PATH = f"{VOLUME_PATH}/lm_abstract_chunks"
LM_SUMMARY_CHUNKS_PATH = f"{VOLUME_PATH}/lm_summary_chunks"
LM_Q_AND_A_CHUNKS_PATH = f"{VOLUME_PATH}/lm_q_and_a_chunks"

PYTHON_VERSION = "3.11"

image = (modal.Image.debian_slim(python_version=PYTHON_VERSION)
         .pip_install(
             "requests",
             "transformers",
             "torch",
             "llama-index-core"
         )
         .add_local_python_source("config", "helpers")
)
rag_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
