# 🧠 Multi-Source RAG Chatbot

A production-ready, multi-user **Retrieval-Augmented Generation (RAG)** chatbot built with **LangChain**, **Streamlit**, and **OpenAI**. Upload documents from multiple sources, chat with them, search the web in real time, get live weather forecasts, and ask about the current date — all in one interface. Every user has their own private account with isolated chat history, documents, and knowledge base that fully persists across sessions.

---

## 📌 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [How Each File Works](#-how-each-file-works)
- [Tech Stack](#-tech-stack)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Adding Users](#-adding-users)
- [Running the App](#-running-the-app)
- [How to Use](#-how-to-use)
- [Retriever Pipeline Deep Dive](#-retriever-pipeline-deep-dive)
- [Chunking Strategy](#-chunking-strategy)
- [Memory and Chat History](#-memory-and-chat-history)
- [Tools Deep Dive](#-tools-deep-dive)
- [Multi-Document Querying](#-multi-document-querying)
- [Authentication and User Isolation](#-authentication-and-user-isolation)
- [Persistence — What Gets Saved](#-persistence--what-gets-saved)
- [Rate Limiting and API Protection](#-rate-limiting-and-api-protection)
- [Troubleshooting](#-troubleshooting)
- [Customisation](#-customisation)
- [Known Limitations](#-known-limitations)
- [License](#-license)

---

## ✨ Features

### Core RAG
- **Multi-source document ingestion** — plain text, `.txt` files, PDF files, web URLs, `.json` files
- **Semantic chunking** — documents split at natural meaning boundaries using OpenAI embeddings, not arbitrary character counts
- **Hybrid retrieval** — combines dense semantic search (FAISS) with sparse keyword search (BM25) using weighted ensemble fusion
- **MultiQuery expansion** — automatically generates multiple query variants to maximise recall
- **Contextual compression** — LLM strips irrelevant content from retrieved passages before answering
- **Cross-document querying** — ask questions that require information from multiple uploaded files simultaneously
- **Source attribution** — every answer shows the exact document passages it was drawn from

### Tools and Real-Time Data
- **Weather tool** — real-time weather conditions + 3-day forecast via WeatherAPI (temperature, humidity, wind, UV, air quality, alerts)
- **Web search tool** — live web search via Tavily with actual article content and summaries
- **Current datetime tool** — always returns the real current date and time, never uses training cutoff
- **Intelligent tool routing** — agent automatically decides which tool(s) to use based on the question

### Memory
- **Conversation memory** — full chat history passed on every query so the chatbot remembers context across the entire session
- **Windowed history** — configurable window prevents context window overflow on long conversations

### Multi-User Authentication
- **Login system** — username/password authentication with bcrypt hashing via `streamlit-authenticator`
- **Complete user isolation** — every user's chat history, documents, and FAISS indexes are stored in separate private folders
- **Persistent login** — cookie-based sessions so users stay logged in across browser refreshes
- **Per-user data** — no user can ever see another user's data

### Persistence
- **Full state persistence** — chat history, session names, document chunks, and FAISS vector indexes all saved to disk
- **Zero re-embedding on reload** — FAISS index loaded from disk, not rebuilt, so no OpenAI API calls on refresh
- **Multi-session support** — create, switch, and delete multiple independent chat sessions, each with their own knowledge base

### Rate Limiting
- **Per-session message limit** — prevents API exhaustion from a single session
- **Per-session source limit** — caps the number of documents per session
- **Token limit** — caps OpenAI response length per call

---

## 🏗 Architecture

```
User Question
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│                    Authentication                        │
│            streamlit-authenticator (JWT cookie)          │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              Format Chat History (manual)                │
│    sess["messages"] → [HumanMessage, AIMessage, ...]     │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              OpenAI Tools Agent (AgentExecutor)          │
│                                                          │
│   Decides which tool(s) to call based on the question   │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │               Available Tools                    │   │
│  │                                                  │   │
│  │  search_documents  → RAG retriever pipeline      │   │
│  │  get_weather       → WeatherAPI                  │   │
│  │  web_search        → Tavily Search               │   │
│  │  get_current_datetime → system clock             │   │
│  └──────────────────────────────────────────────────┘   │
└──────────────────────┬──────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
  search_documents            get_weather /
  tool called                 web_search /
          │                   datetime called
          ▼                         │
┌─────────────────────┐            │
│  Retriever Pipeline │            │
│                     │            │
│  FAISS (dense)  ─┐  │            │
│                  ├──► Ensemble   │
│  BM25 (sparse) ──┘  │     │      │
│                     │     ▼      │
│               MultiQuery        │
│               Retriever         │
│                     │           │
│                     ▼           │
│            Contextual           │
│            Compression          │
└──────────────┬──────────────────┘
               │
               ▼
        Retrieved Chunks
               │
               └──────────────┐
                              │
                              ▼
                    ┌─────────────────────┐
                    │   Final LLM Call    │
                    │                     │
                    │  system prompt      │
                    │  + chat_history     │
                    │  + tool results     │
                    │  + question         │
                    │         │           │
                    │         ▼           │
                    │   gpt-4o-mini       │
                    │         │           │
                    │         ▼           │
                    │  StrOutputParser    │
                    └─────────┬───────────┘
                              │
                              ▼
                    Answer + Sources → Streamlit UI
                              │
                              ▼
                    save_sessions(username)
                    → sessions.json + FAISS to disk
```

---

## 📁 Project Structure

```
rag_project/
│
├── .env                    # API keys (never commit this)
├── .gitignore              # excludes venv, .env, user_data
├── requirements.txt        # all Python dependencies
├── users.yaml              # user credentials (hashed passwords)
├── hash_passwords.py       # run once to generate password hashes
│
├── config.py               # all settings — models, limits, weights
├── document_loaders.py     # loaders for all 5 input types
├── chunker.py              # semantic chunking with fallback
├── retriever.py            # full retriever pipeline (FAISS+BM25+MQ+CC)
├── tools.py                # all 4 agent tools
├── chain.py                # agent builder (fresh + from saved)
└── app.py                  # Streamlit UI + auth + persistence
│
└── user_data/              # auto-created at runtime
    ├── arjun/
    │   ├── sessions.json   # arjun's chat history + chunks
    │   └── faiss_store/    # arjun's FAISS indexes
    │       ├── Session_1/
    │       │   ├── index.faiss
    │       │   └── index.pkl
    │       └── Session_2/
    └── john/
        ├── sessions.json
        └── faiss_store/
```

---

## 🔍 How Each File Works

### `config.py`
Single source of truth for all configuration. Loads environment variables from `.env` via `python-dotenv`. All tunable parameters — model names, chunk sizes, retriever weights, API keys, rate limits — live here. Change anything here without touching other files.

### `document_loaders.py`
One function per input type. Each accepts a Streamlit uploaded file object (or string/URL) and returns a list of LangChain `Document` objects with `source` metadata attached. PDF loader writes to a temp file, uses PyPDFLoader, then cleans up. JSON loader handles both list and dict structures.

### `chunker.py`
Primary splitter is `SemanticChunker` from `langchain_experimental`. It embeds sentences, measures cosine similarity between adjacent sentences, and splits where similarity drops below the 90th percentile threshold — creating chunks that contain complete thoughts. Automatically falls back to `RecursiveCharacterTextSplitter` (1000 chars, 200 overlap) if semantic chunking returns empty results.

### `retriever.py`
Builds the 5-stage retriever pipeline. Contains two entry points: `build_retriever()` (creates a new FAISS index) and `build_retriever_from_saved()` (loads an existing FAISS index from disk, skipping re-embedding). Both share the same `_build_pipeline()` function that adds BM25, Ensemble, MultiQuery, and Contextual Compression on top.

### `tools.py`
Defines all 4 agent tools using `@tool` decorator. `get_weather` and `web_search` are module-level tools. `create_kb_tool()` is a factory that binds the current session's retriever into a `search_documents` tool — called each time documents are added so the tool always points to the latest retriever. `get_base_tools()` and `get_all_tools()` are factory functions returning the right tool set.

### `chain.py`
Builds the `AgentExecutor`. Contains three entry points: `build_agent_only()` (base tools, no KB), `build_chain()` (full tools, builds new FAISS), and `build_chain_from_saved()` (full tools, loads existing FAISS). System prompt instructs the LLM on tool selection, response style, and summarisation behaviour.

### `app.py`
Main Streamlit application. Handles authentication, per-user state management, persistence (save/restore), document ingestion, and the chat UI. Key functions: `save_sessions()` serialises everything to disk, `restore_sessions()` deserialises on page load, `init_base_agent()` ensures tools work before any docs are uploaded, `process_and_add()` orchestrates the full document ingestion pipeline.

### `users.yaml`
YAML credential store consumed by `streamlit-authenticator`. Contains usernames, display names, emails, bcrypt-hashed passwords, and cookie configuration. **Never store plain text passwords here.**

### `hash_passwords.py`
One-time utility script. Pass plain text passwords, receive bcrypt hashes to paste into `users.yaml`.

---

## 🛠 Tech Stack

| Component | Library | Purpose |
|---|---|---|
| UI | Streamlit ≥ 1.35 | Web interface |
| Authentication | streamlit-authenticator | Login / user isolation |
| LLM Framework | LangChain ≥ 0.2 | Chains, agents, tools |
| LLM | langchain-openai (ChatOpenAI) | Answer generation |
| Embeddings | langchain-openai (OpenAIEmbeddings) | Semantic search + chunking |
| Agent | LangChain AgentExecutor | Tool routing |
| Vector Store | faiss-cpu ≥ 1.8 | Dense semantic search |
| Sparse Retrieval | rank-bm25 ≥ 0.2 | Keyword search |
| Semantic Chunking | langchain-experimental | Meaning-based splitting |
| Text Splitting | langchain-text-splitters | Fallback splitting |
| PDF Loading | pypdf ≥ 4.0 | PDF ingestion |
| Web Scraping | beautifulsoup4 + lxml | URL ingestion |
| Web Search | tavily-python | Real-time search |
| Weather | requests + WeatherAPI | Live weather data |
| Password Hashing | bcrypt | Secure credential storage |
| Env Management | python-dotenv | API key loading |
| Token Counting | tiktoken | Context management |

---

## ✅ Prerequisites

- Python **3.10 or higher**
- An **OpenAI API key** — [platform.openai.com](https://platform.openai.com)
- A **WeatherAPI key** (free) — [weatherapi.com](https://weatherapi.com)
- A **Tavily API key** (free tier: 1000 searches/month) — [app.tavily.com](https://app.tavily.com)
- VS Code or any terminal

---

## 🚀 Installation

### Step 1 — Create project folder
```bash
mkdir rag_project
cd rag_project
```

### Step 2 — Create virtual environment
```bash
python -m venv venv
```

### Step 3 — Activate virtual environment

**Windows:**
```bash
venv\Scripts\activate
```
**macOS / Linux:**
```bash
source venv/bin/activate
```
You will see `(venv)` at the start of your terminal prompt.

### Step 4 — Place all project files
Copy all `.py` files, `users.yaml`, `requirements.txt`, and `.env` into `rag_project/`.

### Step 5 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 6 — Select interpreter in VS Code
`Ctrl+Shift+P` → **Python: Select Interpreter** → choose the `venv` Python.

> ⚠️ You must activate the venv every time you open a new terminal. If you see import errors, this is the most likely cause.

---

## ⚙️ Configuration

### `.env` — API Keys
Create a file named `.env` in the project root:
```
OPENAI_API_KEY=sk-your-openai-key
WEATHER_API_KEY=your-weatherapi-key
TAVILY_API_KEY=tvly-your-tavily-key
```
> 🔒 Never commit `.env` to Git. Add it to `.gitignore`.

### `config.py` — All Settings

```python
# ── Models ────────────────────────────────────────────────────
CHAT_MODEL      = "gpt-4o-mini"              # LLM for answering
EMBEDDING_MODEL = "text-embedding-3-small"   # for FAISS + chunking
TEMPERATURE     = 0                          # 0=deterministic, 1=creative

# ── Rate Limits ───────────────────────────────────────────────
MAX_TOKENS               = 1000   # max tokens per LLM response
MAX_REQUESTS_PER_SESSION = 50     # max messages per session
MAX_SOURCES_PER_SESSION  = 10     # max documents per session

# ── Retriever ─────────────────────────────────────────────────
FAISS_K          = 6              # candidates from FAISS
BM25_K           = 6              # candidates from BM25
ENSEMBLE_WEIGHTS = [0.4, 0.6]     # [BM25, FAISS]
MAX_SOURCES_SHOW = 5              # source passages shown in UI

# ── Chunking ──────────────────────────────────────────────────
SEMANTIC_THRESHOLD_TYPE   = "percentile"
SEMANTIC_THRESHOLD_AMOUNT = 90    # higher = fewer, larger chunks
FALLBACK_CHUNK_SIZE       = 1000
FALLBACK_CHUNK_OVERLAP    = 200
```

**Available OpenAI models:**

| Model | Speed | Cost | Recommended for |
|---|---|---|---|
| `gpt-4o-mini` | Fast | Cheapest | Default — great balance |
| `gpt-4o` | Medium | Mid | Complex reasoning |
| `gpt-4-turbo` | Slower | Higher | Very long documents |

---

## 👥 Adding Users

### Step 1 — Edit `hash_passwords.py`
```python
passwords = [
    "arjun_password",    # for user 'arjun'
    "john_password",     # for user 'john'
]
```

### Step 2 — Run it
```bash
python hash_passwords.py
```
Output:
```
User 1: $2b$12$abc...xyz
User 2: $2b$12$def...uvw
```

### Step 3 — Paste hashes into `users.yaml`
```yaml
credentials:
  usernames:
    arjun:
      email: arjun@email.com
      name: Arjun Kapil
      password: $2b$12$abc...xyz    # ← paste here
    john:
      email: john@email.com
      name: John Doe
      password: $2b$12$def...uvw    # ← paste here

cookie:
  expiry_days: 7
  key: replace_with_long_random_secret_string
  name: rag_auth_cookie
```

### Step 4 — To add more users later
Repeat Steps 1–3. Add the new entry to `users.yaml`. No code changes needed.

> ⚠️ The `cookie.key` should be a long random string (30+ characters). Never share it. It signs the login cookie.

---

## ▶️ Running the App

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. To use a different port:
```bash
streamlit run app.py --server.port 8502
```

Stop with `Ctrl+C`.

---

## 📖 How to Use

### Logging In
Enter your username and password on the login screen. The session persists for 7 days (configurable in `users.yaml`) via a secure cookie.

### Chatting Without Documents
The chatbot is fully functional before uploading any documents. You can immediately:
- Ask weather questions: *"What's the weather in Delhi?"*
- Search the web: *"Latest news on AI regulation"*
- Ask the date: *"What day is it today?"*
- Ask general knowledge questions: *"Explain transformer architecture"*

### Adding Documents
In the **left sidebar**, select a source type:

| Source | Steps |
|---|---|
| 📝 Plain Text | Paste text → click **Add Text** |
| 📄 Text File | Upload `.txt` → click **Add File** |
| 📕 PDF | Upload `.pdf` → click **Add PDF** |
| 🌐 Web URL | Paste URL → click **Fetch & Add** |
| 📊 JSON | Upload `.json` → click **Add JSON** |

Processing involves embedding all chunks — this takes a few seconds per document.

### Asking Document Questions
Once documents are added, the `search_documents` tool activates. Ask questions naturally:
- *"Summarise the main points of the document"*
- *"What does the report say about revenue?"*
- *"Compare the findings across both uploaded files"*

### Multi-Tool Questions
The agent can combine tools in a single answer:
- *"What does my uploaded report say about AI, and what's the latest news about it?"* → uses `search_documents` + `web_search`
- *"Should I carry an umbrella in Mumbai based on my travel itinerary PDF?"* → uses `search_documents` + `get_weather`

### Managing Sessions
- **Create**: type a name → click **➕**
- **Switch**: click the session name
- **Delete**: click **🗑** (requires 2+ sessions)
- Each session has its own independent knowledge base and chat history

### Clearing Chat History
Sidebar → **🗑 Manage Chat**:
- **Clear This Chat History** — clears only the current session's messages
- **Clear ALL Sessions Chat** — clears messages from all sessions
- Documents and knowledge base are unaffected

### Logging Out
Click **Logout** at the top of the sidebar. Your data is saved and will be restored on next login.

---

## 🔬 Retriever Pipeline Deep Dive

The retrieval pipeline has 5 stages:

### Stage 1 — FAISS (Dense Semantic Retrieval)
FAISS stores document chunks as dense vectors using OpenAI embeddings (`text-embedding-3-small`). Finds chunks **semantically similar** to the query even without exact word matches. Returns top `FAISS_K` (default 6) results. Index is saved to disk and loaded on page refresh without re-embedding.

### Stage 2 — BM25 (Sparse Keyword Retrieval)
BM25 (Best Match 25) is a classical information retrieval algorithm. Ranks chunks by **exact keyword frequency**, normalised by document length. Catches results FAISS might miss when the query contains specific technical terms, names, or numbers. Returns top `BM25_K` (default 6) results. Rebuilt from stored chunks on page refresh.

### Stage 3 — EnsembleRetriever (Hybrid Fusion)
Merges FAISS and BM25 results using **Reciprocal Rank Fusion (RRF)**. Each result is scored based on its rank in both lists, then weighted (`ENSEMBLE_WEIGHTS = [0.4, 0.6]`). BM25 at 40%, FAISS at 60%. Consistently outperforms either retriever alone, especially for queries that are partly keyword-based and partly conceptual.

### Stage 4 — MultiQueryRetriever (Query Expansion)
Sends the user's question to the LLM and generates **3 alternative phrasings**. Each variant is run through the Ensemble retriever independently. All results are merged and deduplicated. This widens recall by catching relevant chunks that one phrasing might miss — especially important for vague or ambiguous questions.

### Stage 5 — ContextualCompressionRetriever (Passage Extraction)
Takes all retrieved chunks and sends each one to the LLM with the original question. The LLM extracts **only the relevant portion** from each chunk. Chunks that contain no relevant information are discarded entirely. This reduces noise in the context window, lowers token usage, and improves answer quality.

---

## ✂️ Chunking Strategy

### Primary — SemanticChunker
Documents are split by measuring the **cosine similarity between embeddings of adjacent sentences**. When similarity drops below the 90th percentile threshold, a chunk boundary is created. This produces chunks that contain complete, coherent thoughts rather than arbitrary text fragments.

**Advantages over character splitting:**
- Chunks align with natural topic shifts
- No mid-sentence or mid-concept splits
- Better retrieval relevance
- More natural context for the LLM

### Fallback — RecursiveCharacterTextSplitter
Automatically activated when SemanticChunker returns empty results (document too short, API issue, etc.). Splits by character count (1000 chars) with overlap (200 chars). A toast notification appears in the UI when fallback activates.

### On Reload
Chunks are serialised to JSON (plain `page_content` + `metadata` dicts) and saved in `sessions.json`. On reload they are deserialised back into `Document` objects — no re-embedding, no API calls.

---

## 🧠 Memory and Chat History

Memory is implemented **manually** rather than using a LangChain `ConversationBufferMemory` object. This avoids the internal format conflicts that caused errors in `ConversationalRetrievalChain`.

**How it works:**
1. Every message (user and assistant) is appended to `sess["messages"]`
2. Before each query, `format_chat_history()` converts the last N messages into a list of `HumanMessage` and `AIMessage` objects
3. This list is passed to `agent.invoke({"input": prompt, "chat_history": history})`
4. The agent prompt has `MessagesPlaceholder(variable_name="chat_history")` — the LLM sees the full conversation every call

**Window limit (configurable):**
```python
def format_chat_history(messages):
    messages = messages[-20:]    # last 20 messages = 10 exchanges
    ...
```
Prevents context window overflow on long conversations. Increase or remove this slice as needed.

**Memory scope:**
- Memory is per-session and per-user
- Switching sessions gives a completely fresh context
- Chat history persists to disk and is restored on reload
- Clearing chat history removes messages but not documents

---

## 🔧 Tools Deep Dive

The agent has 4 tools. The LLM automatically decides which to call based on the question.

### Tool 1 — `search_documents`
**When used:** questions about uploaded files
**How it works:** calls the full retriever pipeline (FAISS + BM25 + MultiQuery + Compression) with the query, returns formatted passage excerpts with source names
**Available:** only when at least one document has been added to the session
**Special:** rebuilt every time new documents are added so it always reflects the latest knowledge base

### Tool 2 — `get_weather`
**When used:** any weather-related question
**API:** WeatherAPI.com `/v1/forecast.json`
**Returns:** current conditions (temp, feels like, humidity, wind, visibility, UV, AQI) + 3-day forecast (high/low, condition, rain chance) + active weather alerts
**Input:** any city name or location string

### Tool 3 — `web_search`
**When used:** current events, news, live data, anything not in documents
**API:** Tavily Search (`search_depth="advanced"`, `include_answer=True`, `include_published_date=True`)
**Special:** today's date is appended to the query to force fresh results — prevents returning cached/old articles
**Returns:** direct answer + list of results with title, URL, date, and content summary

### Tool 4 — `get_current_datetime`
**When used:** any question about today's date, time, or day of week
**How it works:** calls `datetime.now()` at query time — always returns the real current date
**Why a tool:** the LLM's training cutoff means it doesn't know the real date; making it a tool forces a live lookup every time

---

## 📂 Multi-Document Querying

When multiple files are added to the same session, all chunks are stored in a single shared FAISS index and BM25 index. Each chunk retains its `metadata["source"]` (original filename or URL). The retriever searches across all chunks from all sources simultaneously.

**Example — two PDFs loaded:**
```
Session KB:
  annual_report_2023.pdf  →  142 chunks  [source: annual_report_2023.pdf]
  annual_report_2024.pdf  →  156 chunks  [source: annual_report_2024.pdf]
  Total: 298 chunks in one shared index

Question: "How did operating expenses change between 2023 and 2024?"

Retrieved:
  Source 1: annual_report_2023.pdf (page 12) — "Operating expenses were ₹4.2Cr..."
  Source 2: annual_report_2024.pdf (page 9)  — "Operating expenses rose to ₹5.1Cr..."

Answer: synthesises both passages with comparison
```

The UI's **Source passages** expander shows which file each retrieved passage came from.

---

## 🔐 Authentication and User Isolation

### How Login Works
1. User enters username + password on the Streamlit login screen
2. `streamlit-authenticator` verifies the password against the bcrypt hash in `users.yaml`
3. On success, a signed JWT cookie (`rag_auth_cookie`) is set in the browser
4. The cookie persists for 7 days — users stay logged in across browser refreshes
5. On logout, the cookie is cleared and session state is reset

### How User Isolation Works
Every piece of data is stored under `user_data/<username>/`:
```
user_data/
├── arjun/
│   ├── sessions.json       # only arjun's messages + chunks
│   └── faiss_store/        # only arjun's FAISS indexes
└── john/
    ├── sessions.json       # only john's messages + chunks
    └── faiss_store/        # only john's FAISS indexes
```

When a user logs in, `restore_sessions(username)` loads only their own data. The path functions `get_sessions_file(username)` and `get_faiss_dir(username)` ensure writes always go to the correct user folder.

If a different user logs in on the same browser, `st.session_state` is fully reset:
```python
if st.session_state.get("logged_in_user") != username:
    st.session_state.logged_in_user = username
    st.session_state.sessions       = None   # wipe previous user's state
```

---

## 💾 Persistence — What Gets Saved

| Data | Saved where | Format | Restored without API calls |
|---|---|---|---|
| Chat messages | `sessions.json` | JSON | ✅ Yes |
| Session names | `sessions.json` | JSON | ✅ Yes |
| Source labels | `sessions.json` | JSON | ✅ Yes |
| Document chunks (text) | `sessions.json` | JSON | ✅ Yes |
| FAISS vector index | `faiss_store/<session>/` | Binary | ✅ Yes |
| BM25 index | Rebuilt from chunks | In memory | ✅ Yes (no API) |
| Agent / Retriever | Rebuilt from FAISS | In memory | ✅ Yes (no API) |

**When does save happen:**
- After every assistant message
- After every document is added
- After a new session is created
- After a session is deleted
- After chat history is cleared

**When a session is deleted:** both its entry in `sessions.json` and its FAISS folder under `faiss_store/` are deleted.

---

## 🚦 Rate Limiting and API Protection

Three levels of protection configured in `config.py`:

### 1. Token Limit per Response (`MAX_TOKENS = 1000`)
Applied directly to `ChatOpenAI(max_tokens=MAX_TOKENS)`. Every LLM call is capped at this many output tokens. Prevents runaway long responses.
```python
CHAT_MODEL = "gpt-4o-mini"   # input tokens not limited here
MAX_TOKENS = 1000             # output tokens capped per call
```

### 2. Message Limit per Session (`MAX_REQUESTS_PER_SESSION = 50`)
Checked before each query in `app.py`. When the user hits the limit, they see a warning and must start a new session.
```python
elif len([m for m in sess["messages"] if m["role"] == "user"]) >= MAX_REQUESTS_PER_SESSION:
    st.warning("Session limit reached. Start a new session.")
```

### 3. Source Limit per Session (`MAX_SOURCES_PER_SESSION = 10`)
Checked inside `process_and_add()`. Prevents users from uploading excessive documents.

**Cost-saving tips:**
- Lower `MAX_TOKENS` to `500` for a tight budget
- Lower `FAISS_K` and `BM25_K` to `4` to retrieve fewer chunks (fewer compression calls)
- Disable Contextual Compression in `retriever.py` by returning `multiquery_retriever` directly — biggest cost saving
- Use `gpt-4o-mini` (default) rather than `gpt-4o` — ~20x cheaper per token

---

## 🔧 Troubleshooting

### `ModuleNotFoundError` on any import
Virtual environment not activated. Run:
```bash
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

---

### `No API key found` / key-related errors
- Check `.env` is in the **same folder** as `app.py`
- File must be named exactly `.env` (not `env.txt`)
- Each line must be: `KEY=value` with no spaces around `=`
- After editing `.env`, restart the app

---

### `streamlit-authenticator` login screen not appearing
```bash
pip install streamlit-authenticator bcrypt
```
Also verify `users.yaml` is in the same folder as `app.py` and has valid YAML syntax.

---

### Password hash not working
Re-run `hash_passwords.py` and paste the fresh hash. Hashes are one-way — you cannot recover the original password. If forgotten, generate a new hash for a new password.

---

### Weather shows wrong city
WeatherAPI uses the `q` parameter for location lookup. Be specific:
- ✅ `"Gurugram, India"` or `"Gurgaon"`
- ✅ `"Mumbai, Maharashtra"`
- ❌ `"my city"` (too vague)

---

### Web search returning old results
Tavily appends today's date to the query. If results are still old:
- Try a more specific query with the date: *"India news May 2026"*
- Check your Tavily API key is valid and quota not exhausted

---

### App is slow when adding large PDFs
Each chunk is embedded via the OpenAI API. A 100-page PDF may create 200+ chunks, each requiring an API call. Mitigations:
- Increase `SEMANTIC_THRESHOLD_AMOUNT` to 95 (fewer, larger chunks)
- Split very large PDFs before uploading

---

### `localhost:8501` not opening
- Wait a few seconds — Streamlit takes 3–5s to start
- Try `http://127.0.0.1:8501` instead of `localhost`
- Check for port conflict: `streamlit run app.py --server.port 8502`

---

### All chat history disappeared after restart
- Check `user_data/<username>/sessions.json` exists
- Check `user_data/<username>/faiss_store/` exists with index files
- Permissions issue? Run: `ls -la user_data/` to check
- If file is corrupt, delete `sessions.json` — app creates a fresh one

---

## 🎛 Customisation

### Change the LLM model
`config.py`:
```python
CHAT_MODEL = "gpt-4o"   # upgrade for harder questions
```

### Adjust retrieval depth
`config.py`:
```python
FAISS_K = 8    # more candidates from FAISS
BM25_K  = 8    # more candidates from BM25
```

### Favour keyword search
`config.py`:
```python
ENSEMBLE_WEIGHTS = [0.6, 0.4]   # [BM25, FAISS] — flip balance
```

### Disable contextual compression (much faster)
`retriever.py` — comment out Stage 5:
```python
# compressor = LLMChainExtractor.from_llm(llm)
# final_retriever = ContextualCompressionRetriever(...)
return multiquery_retriever, vectorstore   # skip compression
```

### Change chunk size
`config.py`:
```python
SEMANTIC_THRESHOLD_AMOUNT = 80   # lower = more, smaller chunks
FALLBACK_CHUNK_SIZE       = 500  # smaller fallback
```

### Change memory window size
`app.py` — inside `format_chat_history()`:
```python
messages = messages[-40:]   # keep last 40 messages (20 exchanges)
# or remove the slice entirely for unlimited history
```

### Change login session duration
`users.yaml`:
```yaml
cookie:
  expiry_days: 30   # stay logged in for 30 days
```

### Modify the agent's behaviour
`chain.py` — edit `SYSTEM_PROMPT`. Change persona, response language, tool preference rules, or formatting instructions.

### Add a new user
1. Add plain password to `hash_passwords.py` → run it → copy hash
2. Add new entry to `users.yaml` → restart app

---

## ⚠️ Known Limitations

- **No streaming** — full answer generated before displaying. Long answers feel slow.
- **Re-upload after clearing KB** — clearing the knowledge base deletes chunks and FAISS index; documents must be re-uploaded.
- **Login cookie security** — for production deployment, use HTTPS and a strong random `cookie.key`.
- **Single-server only** — `user_data/` is stored on local disk. Does not work across multiple server instances. For multi-server deployment, replace with a database or cloud storage.
- **Web scraping limits** — some websites block scraping. Pages behind login or paywalls will not load via URL ingestion.
- **JSON deep nesting** — deeply nested JSON structures may not chunk meaningfully. Flat or list-based JSON works best.
- **Tavily quota** — free tier is 1000 searches/month. Monitor usage at [app.tavily.com](https://app.tavily.com).
- **WeatherAPI free tier** — limited to current + 3-day forecast. Historical weather requires a paid plan.
- **No admin panel** — user management (add/remove/reset password) is done by editing `users.yaml` manually.

---

## 📄 License

MIT License. Free to use, modify, and distribute with attribution.

---

## 🙌 Acknowledgements

Built with:
- [LangChain](https://github.com/langchain-ai/langchain) — LLM application framework
- [Streamlit](https://streamlit.io) — Python web UI framework
- [OpenAI](https://openai.com) — LLM and embedding models
- [FAISS](https://github.com/facebookresearch/faiss) — vector similarity search by Meta AI
- [rank-bm25](https://github.com/dorianbrown/rank_bm25) — BM25 in Python
- [Tavily](https://tavily.com) — AI-optimised search API
- [WeatherAPI](https://weatherapi.com) — real-time weather data
- [streamlit-authenticator](https://github.com/mkhorasani/Streamlit-Authenticator) — Streamlit login system