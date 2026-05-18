

from typing import List

from langchain_classic.schema import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    SEMANTIC_THRESHOLD_TYPE,
    SEMANTIC_THRESHOLD_AMOUNT,
    FALLBACK_CHUNK_SIZE,
    FALLBACK_CHUNK_OVERLAP,
)


def chunk_documents(docs: List[Document], embeddings) -> List[Document]:
    """
    Split documents into chunks using SemanticChunker.
    Falls back to RecursiveCharacterTextSplitter if semantic chunking fails
    (e.g. document is too short or embeddings call fails).

    Args:
        docs:       List of LangChain Documents to split.
        embeddings: OpenAIEmbeddings instance (used to find semantic boundaries).

    Returns:
        List of non-empty Document chunks.
    """
    try:
        splitter = SemanticChunker(
            embeddings,
            breakpoint_threshold_type=SEMANTIC_THRESHOLD_TYPE,
            breakpoint_threshold_amount=SEMANTIC_THRESHOLD_AMOUNT,
        )
        chunks = splitter.split_documents(docs)
        chunks = [c for c in chunks if c.page_content.strip()]
        if chunks:
            return chunks
        raise ValueError("SemanticChunker returned empty result.")

    except Exception as e:
        print(f"[chunker] SemanticChunker fell back to recursive splitter: {e}")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=FALLBACK_CHUNK_SIZE,
            chunk_overlap=FALLBACK_CHUNK_OVERLAP,
        )
        chunks = splitter.split_documents(docs)
        return [c for c in chunks if c.page_content.strip()]