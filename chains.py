# chain.py

from typing import List, Tuple

from langchain_classic.schema import Document
from langchain_classic.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from retriever import build_retriever, build_retriever_from_saved
from tools import get_base_tools, get_all_tools


SYSTEM_PROMPT = """You are a helpful AI assistant with access to the following tools:

1. search_documents — searches documents, PDFs, and files the user has uploaded (only available when documents are loaded)
2. get_weather      — fetches real-time weather and 3-day forecasts for any location
3. get_current_datetime — returns the current real date and time
4. web_search       — searches the web for current news and real-time information

Tool selection rules:
- Questions about uploaded files or documents           → use search_documents
- Weather, temperature, forecast, climate questions     → use get_weather
- Questions about today's date or current time          → use get_current_datetime
- Current events, news, live data, general web queries  → use web_search
- If no documents are loaded, answer from your own knowledge or use web_search
- Combine multiple tools when a question needs both document context and live data
- Always use conversation history to understand follow-up questions

Always give a clear, well-structured answer after using the tools.
When reporting news or search results, summarise the actual content in your own words.
Never just list source names or URLs — extract and explain what each result says.
Cite which source (document name or URL) your information came from when relevant.
If you cannot find the answer using tools, say so honestly."""


AGENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])


def _make_executor(llm, tools: list) -> AgentExecutor:
    agent = create_openai_tools_agent(llm, tools, AGENT_PROMPT)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        max_iterations=6,
    )


def build_agent_only(llm) -> AgentExecutor:
    """Weather + web search only. Used before any docs are loaded."""
    return _make_executor(llm, get_base_tools())


def build_chain(
    chunks: List[Document],
    embeddings,
    llm,
    **kwargs,
) -> Tuple[AgentExecutor, object, object]:
    """
    Builds full agent from scratch — creates new FAISS index.
    Called when documents are first uploaded.
    """
    retriever, vectorstore = build_retriever(chunks, embeddings, llm)
    agent = _make_executor(llm, get_all_tools(retriever))
    return agent, retriever, vectorstore


def build_chain_from_saved(
    chunks: List[Document],
    vectorstore,
    llm,
) -> Tuple[AgentExecutor, object, object]:
    """
    Rebuilds full agent from a saved FAISS index.
    Called on page reload — no re-embedding, no API calls for chunks.

    Args:
        chunks:     Restored Document list from JSON.
        vectorstore: FAISS instance loaded from disk.
        llm:        ChatOpenAI instance.

    Returns:
        (agent_executor, retriever, vectorstore)
    """
    retriever, vectorstore = build_retriever_from_saved(chunks, vectorstore, llm)
    agent = _make_executor(llm, get_all_tools(retriever))
    return agent, retriever, vectorstore