# retriever.py

from typing import List, Tuple

from langchain_classic.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import (
    EnsembleRetriever,
    MultiQueryRetriever,
    ContextualCompressionRetriever,
)
from langchain_classic.retrievers.document_compressors import LLMChainExtractor

from config import FAISS_K, BM25_K, ENSEMBLE_WEIGHTS


def _build_pipeline(faiss_retriever, chunks: List[Document], llm):
    """
    Shared pipeline builder used by both build_retriever() and
    build_retriever_from_saved(). Takes a ready FAISS retriever
    and builds BM25 → Ensemble → MultiQuery → Compression on top.
    """
    # BM25
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = BM25_K

    # Ensemble
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=ENSEMBLE_WEIGHTS,
    )

    # MultiQuery
    multiquery_retriever = MultiQueryRetriever.from_llm(
        retriever=ensemble_retriever,
        llm=llm,
    )

    # Contextual Compression
    compressor = LLMChainExtractor.from_llm(llm)
    final_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=multiquery_retriever,
    )

    return final_retriever


def build_retriever(
    chunks: List[Document],
    embeddings,
    llm,
) -> Tuple[ContextualCompressionRetriever, FAISS]:
    """
    Builds retriever from scratch — creates a new FAISS index.
    Called when documents are first uploaded.
    """
    vectorstore = FAISS.from_documents(chunks, embeddings)
    faiss_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": FAISS_K},
    )
    final_retriever = _build_pipeline(faiss_retriever, chunks, llm)
    return final_retriever, vectorstore


def build_retriever_from_saved(
    chunks: List[Document],
    vectorstore: FAISS,
    llm,
) -> Tuple[ContextualCompressionRetriever, FAISS]:
    """
    Rebuilds retriever from a pre-loaded FAISS vectorstore.
    Called on page reload — skips re-embedding so no API calls needed.

    Args:
        chunks:      Restored Document chunks (from JSON).
        vectorstore: FAISS instance loaded from disk.
        llm:         ChatOpenAI instance.

    Returns:
        (final_retriever, vectorstore)
    """
    faiss_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": FAISS_K},
    )
    final_retriever = _build_pipeline(faiss_retriever, chunks, llm)
    return final_retriever, vectorstore