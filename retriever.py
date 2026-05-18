
# FAISS + BM25 → EnsembleRetriever → MultiQueryRetriever → ContextualCompression for besttttest results 🤯 boom

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


def build_retriever(
    chunks: List[Document],
    embeddings,
    llm,
) -> Tuple[ContextualCompressionRetriever, FAISS]:
    """
    Builds the full 4-stage retriever pipeline.

    Stage 1 — FAISS         : dense semantic similarity search
    Stage 2 — BM25          : sparse keyword search
    Stage 3 — Ensemble      : weighted fusion of FAISS + BM25
    Stage 4 — MultiQuery    : generates 3 query variants to widen recall
    Stage 5 — Compression   : LLM strips irrelevant passages from results

    Args:
        chunks:     All document chunks in the knowledge base.
        embeddings: OpenAIEmbeddings instance.
        llm:        ChatOpenAI instance.

    Returns:
        (final_retriever, vectorstore)
        vectorstore is returned so it can be reused or inspected.
    """

   
    vectorstore = FAISS.from_documents(chunks, embeddings)
    faiss_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": FAISS_K},
    )

    
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = BM25_K

    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=ENSEMBLE_WEIGHTS,   # [BM25, FAISS]
    )

    
    multiquery_retriever = MultiQueryRetriever.from_llm(
        retriever=ensemble_retriever,
        llm=llm,
    )

    
    compressor = LLMChainExtractor.from_llm(llm)
    final_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=multiquery_retriever,
    )

    return final_retriever, vectorstore