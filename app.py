

import os
import sys
import logging
import streamlit as st
from pydantic import SecretStr
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    OPENAI_API_KEY, CHAT_MODEL, EMBEDDING_MODEL,
    TEMPERATURE, MAX_SOURCES_SHOW, WEATHER_API_KEY,
    MAX_TOKENS, MAX_REQUESTS_PER_SESSION, MAX_SOURCES_PER_SESSION,
)
from doc_loader import (
    load_plain_text, load_txt_file,
    load_pdf_file, load_url, load_json_file,
)
from chunker import chunk_documents
from chains import build_chain, build_agent_only

logging.getLogger("langchain").setLevel(logging.ERROR)




def empty_session() -> dict:
    return dict(
        messages=[],        
        chunks=[],          
        sources=[],         
        agent=None,        
        retriever=None,     
        vectorstore=None,
    )


def get_session() -> dict:
    return st.session_state.sessions[st.session_state.active_session]


def format_chat_history(messages: list) -> list:
    """
    Converts session messages to LangChain HumanMessage / AIMessage objects.
    The agent's MessagesPlaceholder requires actual message objects.
    Excludes the latest user message (that becomes the 'input').
    """
    history = []
    for msg in messages:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))
    return history


def init_models():
    """Initialise LLM + embeddings once. Keys come from .env via config.py."""
    if not st.session_state.get("llm"):
        st.session_state.llm = ChatOpenAI(
            model=CHAT_MODEL,
            temperature=TEMPERATURE,
            api_key=SecretStr(OPENAI_API_KEY) if OPENAI_API_KEY else None,
            max_completion_tokens=MAX_TOKENS,
        )
    if not st.session_state.get("embeddings"):
        st.session_state.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=SecretStr(OPENAI_API_KEY) if OPENAI_API_KEY else None,
        )


def init_base_agent():
    """
    Initialises weather + web search agent on every page load.
    Runs before any documents are uploaded so the user can chat immediately.
    Skipped if an agent already exists for this session.
    """
    init_models()
    sess = get_session()
    if sess["agent"] is None:
        sess["agent"] = build_agent_only(st.session_state.llm)


def process_and_add(docs, label: str):
    sess = get_session()

    if len(sess["sources"]) >= MAX_SOURCES_PER_SESSION:          
        st.warning(f"⚠️ Max {MAX_SOURCES_PER_SESSION} sources per session reached.") 
        return                                                   

    with st.spinner(f"⚙️ Processing **{label}** …"):
        try:
            init_models()
            new_chunks = chunk_documents(docs, st.session_state.embeddings)

            if not new_chunks:
                st.warning("⚠️ No text content could be extracted.")
                return

            sess["chunks"].extend(new_chunks)
            sess["sources"].append(label)

            agent, retriever, vs = build_chain(
                sess["chunks"],
                st.session_state.embeddings,
                st.session_state.llm,
            )
            sess["agent"]       = agent
            sess["retriever"]   = retriever
            sess["vectorstore"] = vs

            st.success(
                f"✅ **{label}** added — "
                f"{len(new_chunks)} new chunks | {len(sess['chunks'])} total"
            )
        except Exception as e:
            st.error(f"❌ Error: {e}")




st.set_page_config(page_title="Multi-Source RAG", page_icon="🧠", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebar"] { min-width: 340px; max-width: 340px; }
[data-testid="stMetric"]  { background: #f0f2f6; border-radius: 8px; padding: 10px 14px; }
#MainMenu, footer         { visibility: hidden; }
</style>
""", unsafe_allow_html=True)




if "sessions" not in st.session_state:
    st.session_state.sessions = {"Session 1": empty_session()}
if "active_session" not in st.session_state:
    st.session_state.active_session = "Session 1"
if "llm" not in st.session_state:
    st.session_state.llm = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None



with st.sidebar:

    # ── API status ───────────────────────────────────────────
    st.markdown("## 🔐 API Status")

    if OPENAI_API_KEY:
        st.success("✅ OpenAI key loaded")
    else:
        st.error("❌ OPENAI_API_KEY missing in .env")
        st.stop()

    if WEATHER_API_KEY:
        st.success("✅ WeatherAPI key loaded")
    else:
        st.warning("⚠️ WEATHER_API_KEY missing — weather tool disabled")

    st.success("✅ DuckDuckGo search ready (no key needed)")

    st.divider()

    # ── Active tools ─────────────────────────────────────────
    st.markdown("## 🔧 Active Tools")
    sess = get_session()
    st.markdown(f"{'✅' if sess['retriever'] else '⬜'} `search_documents` "
                f"{'— ' + str(len(sess['chunks'])) + ' chunks' if sess['retriever'] else '— no docs loaded'}")
    st.markdown(f"{'✅' if WEATHER_API_KEY else '❌'} `get_weather`")
    st.markdown("✅ `web_search` (DuckDuckGo)")

    st.divider()

    
    st.markdown("## 💬 Chat Sessions")

    col_name, col_btn = st.columns([3, 1])
    with col_name:
        new_sname = st.text_input(
            "ns", placeholder="New session name…",
            label_visibility="collapsed", key="new_sname",
        )
    with col_btn:
        if st.button("➕", use_container_width=True, key="add_sess"):
            name = new_sname.strip()
            if name and name not in st.session_state.sessions:
                st.session_state.sessions[name] = empty_session()
                st.session_state.active_session = name
                st.rerun()

    for sname in list(st.session_state.sessions.keys()):
        is_active = sname == st.session_state.active_session
        c1, c2 = st.columns([5, 1])
        with c1:
            if st.button(
                f"{'▶ ' if is_active else ''}{sname}",
                key=f"sess_{sname}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state.active_session = sname
                st.rerun()
        with c2:
            if st.button("🗑", key=f"del_{sname}", use_container_width=True):
                if len(st.session_state.sessions) > 1:
                    del st.session_state.sessions[sname]
                    if st.session_state.active_session == sname:
                        st.session_state.active_session = list(
                            st.session_state.sessions.keys()
                        )[0]
                    st.rerun()

    st.divider()

   
    st.markdown("## 📁 Add Documents *(optional)*")
    st.caption("Upload files to enable document Q&A alongside weather and web search.")

    src_type = st.selectbox(
        "Source type",
        ["📝 Plain Text", "📄 Text File (.txt)",
         "📕 PDF File", "🌐 Web URL", "📊 JSON File"],
    )

    if src_type == "📝 Plain Text":
        txt = st.text_area("Paste your text:", height=150, key="plain_txt")
        if st.button("Add Text ➕", use_container_width=True, key="add_plain"):
            if txt.strip():
                process_and_add(load_plain_text(txt.strip()), "Plain Text")
            else:
                st.warning("Text is empty.")

    elif src_type == "📄 Text File (.txt)":
        f = st.file_uploader("Upload .txt file", type=["txt"], key="up_txt")
        if st.button("Add File ➕", use_container_width=True, key="add_txt"):
            if f:   process_and_add(load_txt_file(f), f.name)
            else:   st.warning("Upload a file first.")

    elif src_type == "📕 PDF File":
        f = st.file_uploader("Upload PDF", type=["pdf"], key="up_pdf")
        if st.button("Add PDF ➕", use_container_width=True, key="add_pdf"):
            if f:   process_and_add(load_pdf_file(f), f.name)
            else:   st.warning("Upload a PDF first.")

    elif src_type == "🌐 Web URL":
        url = st.text_input("Enter URL:", placeholder="https://…", key="url_inp")
        if st.button("Fetch & Add ➕", use_container_width=True, key="add_url"):
            if url.strip():
                try:    process_and_add(load_url(url.strip()), url.strip())
                except Exception as e: st.error(f"Failed to load URL: {e}")
            else:   st.warning("Enter a URL.")

    elif src_type == "📊 JSON File":
        f = st.file_uploader("Upload JSON", type=["json"], key="up_json")
        if st.button("Add JSON ➕", use_container_width=True, key="add_json"):
            if f:   process_and_add(load_json_file(f), f.name)
            else:   st.warning("Upload a JSON file first.")

    
    sess = get_session()
    if sess["sources"]:
        st.divider()
        st.markdown("### 📊 Knowledge Base")
        st.metric("Total Chunks", len(sess["chunks"]))
        with st.expander(f"📂 {len(sess['sources'])} source(s) loaded"):
            for i, s in enumerate(sess["sources"], 1):
                st.write(f"{i}. {s}")
        if st.button("🗑 Clear Knowledge Base", use_container_width=True, key="clear_kb"):
            get_session().update(empty_session())
            st.rerun()



init_base_agent()

sess  = get_session()
aname = st.session_state.active_session

st.markdown(f"# 🧠 {aname}")

# Status bar
c1, c2, c3 = st.columns(3)
if sess["retriever"]:
    c1.metric("Status", "🟢 Docs + Tools")
elif sess["agent"]:
    c1.metric("Status", "🟡 Tools only")
else:
    c1.metric("Status", "🔴 Not initialised")
c2.metric("Chunks",  len(sess["chunks"]))
c3.metric("Sources", len(sess["sources"]))

st.divider()


if not sess["messages"]:
    st.markdown(
        "<div style='text-align:center; padding:50px 20px; color:#9ca3af;'>"
        "<h3>👋 Start chatting — no documents needed!</h3>"
        "<p>🌤️ <em>\"What's the weather in Mumbai?\"</em></p>"
        "<p>🔍 <em>\"Search for the latest AI news\"</em></p>"
        "<p>💬 <em>\"Explain how RAG works\"</em></p>"
        "<p style='margin-top:20px;'>📄 Upload documents in the sidebar to also "
        "ask questions about your files.</p>"
        "</div>",
        unsafe_allow_html=True,
    )


for msg in sess["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

        if msg.get("tools_used"):
            with st.expander("🔧 Tools used"):
                for step in msg["tools_used"]:
                    st.markdown(f"**Tool:** `{step['tool']}`")
                    st.markdown(f"**Input:** {step['input']}")
                    with st.expander("Output"):
                        st.text(step["output"][:800])
                    st.divider()

        if msg.get("sources"):
            with st.expander("📚 Source passages"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**{i}. `{src['source']}`**")
                    st.markdown(f"> {src['content'][:350]}…")
                    st.divider()


prompt = st.chat_input("Ask anything — weather, news, or questions about your documents…")

if prompt:
    if not sess["agent"]:
        st.error("Agent not initialised — please refresh the page.")

    elif len([m for m in sess["messages"] if m["role"] == "user"]) >= MAX_REQUESTS_PER_SESSION:
        st.warning(f"⚠️ Session limit of {MAX_REQUESTS_PER_SESSION} messages reached. "
                   "Start a new session from the sidebar.")

    else:
        sess["messages"].append({"role": "user", "content": prompt})
        sess["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking…"):
                try:
                    history = format_chat_history(sess["messages"][:-1])

                    result = sess["agent"].invoke({
                        "input":        prompt,
                        "chat_history": history,
                    })

                    answer = result.get("output", "No response generated.")
                    steps  = result.get("intermediate_steps", [])

                    st.write(answer)

                    # ── Parse tool calls ──────────────────────
                    tools_used   = []
                    sources_data = []
                    seen_snips   = set()

                    if steps:
                        with st.expander("🔧 Tools used"):
                            for action, observation in steps:
                                tool_name  = action.tool
                                tool_input = str(action.tool_input)
                                tool_output = str(observation)

                                tools_used.append({
                                    "tool":   tool_name,
                                    "input":  tool_input,
                                    "output": tool_output,
                                })

                                st.markdown(f"**Tool:** `{tool_name}`")
                                st.markdown(f"**Input:** {tool_input}")
                                with st.expander("Output"):
                                    st.text(tool_output[:800])
                                st.divider()

                                # Extract sources from KB search tool calls
                                if tool_name == "search_documents" and sess["retriever"]:
                                    try:
                                        query = (
                                            action.tool_input
                                            if isinstance(action.tool_input, str)
                                            else list(action.tool_input.values())[0]
                                        )
                                        src_docs = sess["retriever"].invoke(query)
                                        for doc in src_docs[:MAX_SOURCES_SHOW]:
                                            snip = doc.page_content[:200]
                                            if snip not in seen_snips:
                                                seen_snips.add(snip)
                                                sources_data.append({
                                                    "source":  doc.metadata.get("source", "Unknown"),
                                                    "content": doc.page_content,
                                                })
                                    except Exception:
                                        pass

                    if sources_data:
                        with st.expander("📚 Source passages"):
                            for i, src in enumerate(sources_data, 1):
                                st.markdown(f"**{i}. `{src['source']}`**")
                                st.markdown(f"> {src['content'][:350]}…")
                                st.divider()

                    sess["messages"].append({
                        "role":       "assistant",
                        "content":    answer,
                        "tools_used": tools_used,
                        "sources":    sources_data,
                    })

                except Exception as e:
                    err = f"Error: {e}"
                    st.error(err)
                    sess["messages"].append({"role": "assistant", "content": err})

        st.rerun()