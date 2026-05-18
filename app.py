

import os
import logging
import streamlit as st
from pydantic import SecretStr
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Project modules
from config import OPENAI_API_KEY, CHAT_MODEL, EMBEDDING_MODEL, TEMPERATURE, MAX_SOURCES_SHOW
from doc_loader import load_plain_text, load_txt_file, load_pdf_file, load_url, load_json_file
from chunker import chunk_documents
from chains import build_chain

logging.getLogger("langchain").setLevel(logging.ERROR)



def empty_session() -> dict:
    return dict(
        messages=[],        
        chunks=[],         
        sources=[],        
        chain=None,         
        retriever=None,     
        vectorstore=None,
    )


def get_session() -> dict:
    return st.session_state.sessions[st.session_state.active_session]

def format_chat_history(messages: list) -> str:
    """
    Converts session messages into a plain string for the prompt.
    Excludes the current (latest) user message — that's the 'question'.
    """
    if not messages:
        return "No previous conversation."
    lines = []
    for msg in messages:
        role = "Human" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def init_models():
    """Initialise LLM + embeddings once. Reads API key from .env via config.py."""
    if not st.session_state.get("llm"):
        st.session_state.llm = ChatOpenAI(
            model=CHAT_MODEL,
            temperature=TEMPERATURE,
            api_key=SecretStr(OPENAI_API_KEY) if OPENAI_API_KEY else None,
        )
    if not st.session_state.get("embeddings"):
        st.session_state.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=SecretStr(OPENAI_API_KEY) if OPENAI_API_KEY else None,
        )


def process_and_add(docs, label: str):
    """Chunk docs → extend KB → rebuild retriever + chain."""
    sess = get_session()
    with st.spinner(f"⚙️ Processing **{label}** …"):
        try:
            init_models()
            new_chunks = chunk_documents(docs, st.session_state.embeddings)

            if not new_chunks:
                st.warning("⚠️ No text content could be extracted.")
                return

            sess["chunks"].extend(new_chunks)
            sess["sources"].append(label)

            chain, retriever, vs = build_chain(
                sess["chunks"],
                st.session_state.embeddings,
                st.session_state.llm,
                existing_memory=sess.get("memory"),   # preserve chat history
            )
            sess["chain"]      = chain
            sess["retriever"]  = retriever
            sess["vectorstore"] = vs

            st.success(
                f"✅ **{label}** added — "
                f"{len(new_chunks)} new chunks | {len(sess['chunks'])} total"
            )

        except Exception as e:
            st.error(f"❌ Error: {e}")




st.set_page_config(
    page_title="Multi-Source RAG",
    page_icon="🧠",
    layout="wide",
)

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

   
    st.markdown("## 🔐 API Status")
    if OPENAI_API_KEY:
        st.success("✅ API key loaded from .env")
    else:
        st.error("❌ No API key found. Add OPENAI_API_KEY to your .env file.")
        st.stop()

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
                st.session_state.active_session  = name
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

    
    st.markdown("## 📁 Add Knowledge Source")

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
            if f:
                process_and_add(load_txt_file(f), f.name)
            else:
                st.warning("Upload a file first.")

    
    elif src_type == "📕 PDF File":
        f = st.file_uploader("Upload PDF", type=["pdf"], key="up_pdf")
        if st.button("Add PDF ➕", use_container_width=True, key="add_pdf"):
            if f:
                process_and_add(load_pdf_file(f), f.name)
            else:
                st.warning("Upload a PDF first.")

    
    elif src_type == "🌐 Web URL":
        url = st.text_input("Enter URL:", placeholder="https://…", key="url_inp")
        if st.button("Fetch & Add ➕", use_container_width=True, key="add_url"):
            if url.strip():
                try:
                    process_and_add(load_url(url.strip()), url.strip())
                except Exception as e:
                    st.error(f"Failed to load URL: {e}")
            else:
                st.warning("Enter a URL.")

    
    elif src_type == "📊 JSON File":
        f = st.file_uploader("Upload JSON", type=["json"], key="up_json")
        if st.button("Add JSON ➕", use_container_width=True, key="add_json"):
            if f:
                process_and_add(load_json_file(f), f.name)
            else:
                st.warning("Upload a JSON file first.")

    
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




sess  = get_session()
aname = st.session_state.active_session

st.markdown(f"# 🧠 {aname}")

# Status bar
c1, c2, c3 = st.columns(3)
c1.metric("Status",  "🟢 Ready" if sess["chain"] else "🔴 No docs loaded")
c2.metric("Chunks",  len(sess["chunks"]))
c3.metric("Sources", len(sess["sources"]))

st.divider()


if not sess["messages"]:
    st.markdown(
        "<div style='text-align:center; padding:60px 20px; color:#9ca3af;'>"
        "<h3>👋 Add a document in the sidebar, then start chatting.</h3>"
        "<p>You can add multiple files — questions work across all of them.</p>"
        "</div>",
        unsafe_allow_html=True,
    )


for msg in sess["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 View source passages"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**{i}. `{src['source']}`**")
                    st.markdown(f"> {src['content'][:350]}…")
                    st.divider()


prompt = st.chat_input(
    "Ask a question about your documents…",
    disabled=not sess["chain"],
)

if prompt:
    if not sess["chain"]:
        st.error("Add at least one document first using the sidebar.")
    else:
        # Add and display user message
        sess["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Searching knowledge base…"):
                try:
                    history_str = format_chat_history(sess["messages"][:-1])
 
                    # Invoke chain — returns plain string answer
                    answer = sess["chain"].invoke({
                        "question":     prompt,
                        "chat_history": history_str,
                    })
 
                    # Fetch source docs separately via the stored retriever
                    src_docs = sess["retriever"].invoke(prompt)

                    st.write(answer)

                    # Collect and deduplicate sources
                    sources_data, seen = [], set()
                    if src_docs:
                        with st.expander("📚 Source passages used"):
                            shown = 0
                            for doc in src_docs:
                                snip = doc.page_content[:200]
                                if snip in seen:
                                    continue
                                seen.add(snip)
                                src_name = doc.metadata.get("source", "Unknown")
                                sources_data.append({
                                    "source": src_name,
                                    "content": doc.page_content,
                                })
                                shown += 1
                                st.markdown(f"**Source {shown}:** `{src_name}`")
                                st.markdown(f"> {snip}…")
                                if shown >= MAX_SOURCES_SHOW:
                                    break
                                st.divider()

                    sess["messages"].append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources_data,
                    })

                except Exception as e:
                    err = f"Error: {e}"
                    st.error(err)
                    sess["messages"].append({"role": "assistant", "content": err})

        st.rerun()