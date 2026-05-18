

from typing import List, Tuple

from langchain_classic.schema import Document
from langchain_classic.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from retriever import build_retriever



PROMPT_TEMPLATE = """You are a helpful AI assistant with access to \
document context and conversation history.

Conversation so far:
{chat_history}

Relevant document context:
{context}

Rules:
1. If the answer is in the document context, use it to answer.
2. If the question is about the conversation itself \
   (e.g. "what did I just ask?", "what did you say earlier?", \
   "summarise our chat"), answer using the conversation history above.
3. Only say you don't know if the answer is genuinely in neither source.

Question: {question}
Answer:"""

QA_PROMPT = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)


def _format_docs(docs: List[Document]) -> str:
    """Join retrieved chunks into a single context string."""
    if not docs:
        return "No relevant documents found."
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def build_chain(
    chunks: List[Document],
    embeddings,
    llm,
    **kwargs,         
) -> Tuple[object, object, object]:
    """
    Builds a simple LCEL RAG chain.

    Input expected by chain.invoke():
        {
            "question":     str   — the user's question,
            "chat_history": str   — formatted conversation so far
        }

    Returns:
        (chain, retriever, vectorstore)
        retriever is returned separately so app.py can fetch source docs.
    """
    retriever, vectorstore = build_retriever(chunks, embeddings, llm)

    chain = (
        {
            
            "context":      lambda x: _format_docs(retriever.invoke(x["question"])),
            "question":     lambda x: x["question"],
            "chat_history": lambda x: x.get("chat_history", "No previous conversation."),
        }
        | QA_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain, retriever, vectorstore