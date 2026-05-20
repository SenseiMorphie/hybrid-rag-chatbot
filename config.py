import os
from dotenv import load_dotenv
from pydantic import SecretStr

load_dotenv()

_openai_api_key = os.getenv("OPENAI_API_KEY")
OPENAI_API_KEY = SecretStr(_openai_api_key) if _openai_api_key is not None else None
CHAT_MODEL      = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
TEMPERATURE     = 0


WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")   

FAISS_K          = 6
BM25_K           = 6
ENSEMBLE_WEIGHTS = [0.4, 0.6]
MAX_SOURCES_SHOW = 5

SEMANTIC_THRESHOLD_TYPE   = "percentile"
SEMANTIC_THRESHOLD_AMOUNT = 90
FALLBACK_CHUNK_SIZE       = 1000
FALLBACK_CHUNK_OVERLAP    = 200

MAX_TOKENS               = 1000
MAX_REQUESTS_PER_SESSION = 50
MAX_CHUNKS_PER_SOURCE    = 100
MAX_SOURCES_PER_SESSION  = 10

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")