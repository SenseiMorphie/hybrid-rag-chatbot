

import os
from re import M
from dotenv import load_dotenv

load_dotenv()  


OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
CHAT_MODEL       = "gpt-4o-mini"  
EMBEDDING_MODEL  = "text-embedding-3-small"
TEMPERATURE      = 0
MAX_TOKENS       = 1000


FAISS_K          = 6    
BM25_K           = 6   
ENSEMBLE_WEIGHTS = [0.4, 0.6]   
MAX_SOURCES_SHOW = 5   


SEMANTIC_THRESHOLD_TYPE   = "percentile"   
SEMANTIC_THRESHOLD_AMOUNT = 90
FALLBACK_CHUNK_SIZE       = 1000
FALLBACK_CHUNK_OVERLAP    = 200