# 🧠 Multi-Source RAG Chatbot

A production-ready Retrieval-Augmented Generation (RAG) chatbot built with **LangChain**, **Streamlit**, and **OpenAI**. Upload documents from multiple sources — PDFs, text files, web URLs, JSON files, or plain text — and chat with all of them simultaneously. The chatbot remembers your conversation history and can answer questions that span across multiple documents.

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
- [Running the App](#-running-the-app)
- [How to Use](#-how-to-use)
- [Retriever Pipeline Deep Dive](#-retriever-pipeline-deep-dive)
- [Chunking Strategy](#-chunking-strategy)
- [Memory and Chat History](#-memory-and-chat-history)
- [Multi-Document Querying](#-multi-document-querying)
- [Troubleshooting](#-troubleshooting)
- [Customisation](#-customisation)
- [Known Limitations](#-known-limitations)
- [License](#-license)

---

## ✨ Features

- **Multi-source ingestion** — supports plain text, `.txt` files, PDF files, web URLs, and `.json` files
- **Multi-session chat** — create separate chat sessions in the sidebar, each with its own isolated knowledge base and history
- **Cross-document querying** — ask questions that require information from two or more uploaded documents simultaneously
- **Semantic chunking** — documents are split at natural meaning boundaries, not arbitrary character counts
- **Hybrid retrieval** — combines dense (FAISS) and sparse (BM25) search for maximum recall
- **MultiQuery expansion** — automatically generates multiple query variants to retrieve a broader set of relevant passages
- **Contextual compression** — strips irrelevant content from retrieved passages before sending to the LLM
- **Conversation memory** — full chat history is passed with every query; the chatbot remembers what was said earlier in the session
- **Source attribution** — every answer shows the exact document passages it was drawn from
- **API key via `.env`** — your OpenAI key never appears in code

---

## 🏗 Architecture

```
User Question
      │
      ▼
 Format Chat History (plain string from session messages)
      │
      ▼
┌─────────────────────────────────┐
│         Retriever Pipeline      │
│                                 │
│  ┌──────────┐  ┌─────────────┐  │
│  │  FAISS   │  │    BM25     │  │
│  │ (dense)  │  │  (sparse)   │  │
│  └────┬─────┘  └──────┬──────┘  │
│       └──────┬─────────┘        │
│              ▼                  │
│     EnsembleRetriever           │
│     (weighted fusion)           │
│              │                  │
│              ▼                  │
│     MultiQueryRetriever         │
│     (3 query variants)          │
│              │                  │
│              ▼                  │
│  ContextualCompressionRetriever │
│  (strips irrelevant passages)   │
└──────────────┬──────────────────┘
               │
               ▼
        Retrieved Chunks
               │
               ▼
┌──────────────────────────────┐
│         LCEL Chain           │
│                              │
│  context + question          │
│  + chat_history              │
│         │                    │
│         ▼                    │
│    ChatPromptTemplate        │
│         │                    │
│         ▼                    │
│    ChatOpenAI (gpt-4o-mini)  │
│         │                    │
│         ▼                    │
│     StrOutputParser          │
└──────────┬───────────────────┘
           │
           ▼
        Answer + Sources → Streamlit UI
```

---

## 📁 Project Structure

```
rag_project/
│
├── .env                   # API keys (never commit this)
├── requirements.txt       # All Python dependencies
│
├── config.py              # Central settings — models, chunk sizes, weights
├── document_loaders.py    # Loaders for all input types
├── chunker.py             # Semantic chunking with fallback
├── retriever.py           # Full retriever pipeline (FAISS + BM25 + MultiQuery + Compression)
├── chain.py               # LCEL chain builder with memory-aware prompt
└── app.py                 # Streamlit UI + session management
```

---

## 🔍 How Each File Works

### `config.py`
Central configuration file. All tunable parameters live here — model names, chunk sizes, retriever weights, etc. Change settings here instead of hunting through the codebase.

### `document_loaders.py`
Contains one loader function per input type. Each function takes a Streamlit uploaded file object (or a string/URL) and returns a list of LangChain `Document` objects with appropriate metadata (source name, page number, index).

| Function | Input | Notes |
|---|---|---|
| `load_plain_text()` | string | Wraps raw text in a Document |
| `load_txt_file()` | Streamlit file | Decodes UTF-8 |
| `load_pdf_file()` | Streamlit file | Writes to temp file, uses PyPDFLoader, one Doc per page |
| `load_url()` | URL string | Uses WebBaseLoader, scrapes visible text |
| `load_json_file()` | Streamlit file | List JSON → one Doc per item; Dict JSON → one Doc |

### `chunker.py`
Uses `SemanticChunker` from `langchain_experimental` as the primary splitter. It embeds sentences and finds breakpoints where meaning changes significantly (using percentile thresholding). Falls back to `RecursiveCharacterTextSplitter` if the semantic chunker fails or returns empty results.

### `retriever.py`
Builds the 5-stage retriever pipeline. Returns both the final retriever and the raw FAISS vectorstore. See [Retriever Pipeline Deep Dive](#-retriever-pipeline-deep-dive) for full details.

### `chain.py`
Builds an LCEL (LangChain Expression Language) chain. Takes `question` and `chat_history` (as a plain formatted string) as inputs. Uses a custom `ChatPromptTemplate` that instructs the LLM to answer from documents OR from conversation history depending on the question type.

### `app.py`
Main Streamlit application. Manages:
- Multiple isolated chat sessions (sidebar)
- Document ingestion and processing
- Chat UI with message history display
- Manual chat history formatting (passed to chain on every call)
- Source document display with deduplication

---

## 🛠 Tech Stack

| Component | Library | Version |
|---|---|---|
| UI framework | Streamlit | ≥ 1.35 |
| LLM framework | LangChain | ≥ 0.2 |
| LLM | langchain-openai (ChatOpenAI) | ≥ 0.1 |
| Embeddings | langchain-openai (OpenAIEmbeddings) | ≥ 0.1 |
| Vector store | FAISS (faiss-cpu) | ≥ 1.8 |
| Sparse retrieval | rank-bm25 | ≥ 0.2 |
| Semantic chunking | langchain-experimental | ≥ 0.0.59 |
| Text splitting | langchain-text-splitters | ≥ 0.2 |
| PDF loading | pypdf | ≥ 4.0 |
| Web scraping | beautifulsoup4 + lxml | ≥ 4.12 |
| Token counting | tiktoken | ≥ 0.7 |
| Env management | python-dotenv | ≥ 1.0 |
| OpenAI client | openai | ≥ 1.30 |

---

## ✅ Prerequisites

- Python **3.10 or higher**
- An **OpenAI API key** (get one at [platform.openai.com](https://platform.openai.com))
- VS Code (recommended) or any terminal
- Git (optional, for version control)

---

## 🚀 Installation

### Step 1 — Clone or download the project

```bash
# If using git
git clone <your-repo-url>
cd rag_project

# Or just create the folder manually
mkdir rag_project
cd rag_project
```

### Step 2 — Create a virtual environment

```bash
python -m venv venv
```

### Step 3 — Activate the virtual environment

**Windows:**
```bash
venv\Scripts\activate
```

**macOS / Linux:**
```bash
source venv/bin/activate
```

You should see `(venv)` at the start of your terminal prompt confirming activation.

> ⚠️ You must activate the venv every time you open a new terminal. If you see import errors, this is usually the culprit.

### Step 4 — Install dependencies

```bash
pip install -r requirements.txt
```

This installs all required packages into your virtual environment. It may take 2–3 minutes on first install.

### Step 5 — Select the venv in VS Code (recommended)

1. Press `Ctrl + Shift + P` (Windows/Linux) or `Cmd + Shift + P` (Mac)
2. Type **Python: Select Interpreter**
3. Choose the interpreter that shows `venv` in its path

---

## ⚙️ Configuration

### API Key — `.env` file

Create a file named `.env` in the root of your project folder:

```
OPENAI_API_KEY=sk-your-actual-key-here
```

> 🔒 **Security note:** Never commit `.env` to Git. Add it to `.gitignore`:
> ```
> echo ".env" >> .gitignore
> ```

### Model and parameter settings — `config.py`

Open `config.py` to adjust any of these settings:

```python
CHAT_MODEL       = "gpt-4o-mini"        # LLM for answering questions
EMBEDDING_MODEL  = "text-embedding-3-small"  # Embedding model for FAISS + chunking
TEMPERATURE      = 0                     # 0 = deterministic, 1 = creative

FAISS_K          = 6                     # Docs returned by FAISS per query
BM25_K           = 6                     # Docs returned by BM25 per query
ENSEMBLE_WEIGHTS = [0.4, 0.6]            # [BM25 weight, FAISS weight]
MAX_SOURCES_SHOW = 5                     # Max source passages shown in UI

SEMANTIC_THRESHOLD_TYPE   = "percentile"
SEMANTIC_THRESHOLD_AMOUNT = 90           # Higher = fewer, larger chunks
FALLBACK_CHUNK_SIZE       = 1000         # Characters (fallback splitter only)
FALLBACK_CHUNK_OVERLAP    = 200
```

**Available OpenAI chat models:**

| Model | Speed | Cost | Best for |
|---|---|---|---|
| `gpt-4o-mini` | Fast | Cheapest | Default — good balance |
| `gpt-4o` | Medium | Mid | More complex reasoning |
| `gpt-4-turbo` | Slower | Higher | Long documents |

---

## ▶️ Running the App

Make sure your virtual environment is activated, then run:

```bash
streamlit run app.py
```

The app will open automatically at:
```
http://localhost:8501
```

To stop the app press `Ctrl + C` in the terminal.

To run on a different port:
```bash
streamlit run app.py --server.port 8502
```

---

## 📖 How to Use

### 1. Adding documents

In the **left sidebar**, select a source type from the dropdown:

| Source type | How to add |
|---|---|
| 📝 Plain Text | Paste text into the text area → click **Add Text** |
| 📄 Text File | Upload a `.txt` file → click **Add File** |
| 📕 PDF File | Upload a `.pdf` file → click **Add PDF** |
| 🌐 Web URL | Paste a URL → click **Fetch & Add** |
| 📊 JSON File | Upload a `.json` file → click **Add JSON** |

After adding, the sidebar shows a **Knowledge Base** section with total chunk count and source list.

### 2. Chatting

Type your question in the chat input at the bottom of the main area and press Enter. The chatbot will:
1. Search the knowledge base using the full retriever pipeline
2. Generate an answer using the retrieved passages and your chat history
3. Show an expandable **Source passages** section below the answer

### 3. Managing sessions

- Click **➕** in the Sessions section to create a new chat session
- Click a session name to switch to it
- Click **🗑** next to a session to delete it
- Each session has its own isolated knowledge base and conversation history

### 4. Cross-document questions

Simply upload multiple files to the same session and ask questions that require both. For example:
- Upload `report_q1.pdf` and `report_q2.pdf`
- Ask: *"Compare Q1 and Q2 revenue figures"*
- The retriever will pull relevant passages from both files simultaneously

### 5. Clearing the knowledge base

Scroll to the bottom of the sidebar and click **🗑 Clear Knowledge Base** to remove all documents from the current session without deleting the session itself.

---

## 🔬 Retriever Pipeline Deep Dive

The retrieval pipeline has 5 stages, each building on the last:

### Stage 1 — FAISS (Dense Retrieval)
FAISS (Facebook AI Similarity Search) stores document chunks as dense vectors using OpenAI embeddings. It finds chunks that are **semantically similar** to the query even if they don't share exact words. Returns the top `FAISS_K` results.

### Stage 2 — BM25 (Sparse Retrieval)
BM25 is a classic keyword-based ranking algorithm. It finds chunks that contain the **exact words** from the query and ranks them by term frequency and document length normalisation. Returns the top `BM25_K` results.

### Stage 3 — EnsembleRetriever (Hybrid Fusion)
Combines FAISS and BM25 results using **Reciprocal Rank Fusion (RRF)**, weighted by `ENSEMBLE_WEIGHTS = [0.4, 0.6]`. This means FAISS semantic results count for 60% and BM25 keyword results count for 40%. Hybrid search consistently outperforms either method alone.

### Stage 4 — MultiQueryRetriever (Query Expansion)
Sends the user's question to the LLM and asks it to generate **3 alternative phrasings** of the same question. Each variant is run through the Ensemble retriever independently. All results are merged and deduplicated. This widens recall by catching relevant chunks that one phrasing might miss.

### Stage 5 — ContextualCompressionRetriever (Passage Extraction)
Takes all retrieved chunks and sends each one to the LLM with the original question, asking it to **extract only the relevant portion**. Chunks that are entirely irrelevant are discarded. This reduces noise in the context window and improves answer quality.

---

## ✂️ Chunking Strategy

**Primary: SemanticChunker**

Documents are embedded sentence-by-sentence and split at points where the **cosine similarity between adjacent sentences drops below a threshold** (90th percentile by default). This creates chunks that contain complete thoughts rather than arbitrary character counts.

**Fallback: RecursiveCharacterTextSplitter**

If the semantic chunker returns empty results (e.g. document is too short, API call fails), the system falls back to splitting by character count (1000 chars, 200 overlap) automatically. A toast notification appears in the UI when this happens.

---

## 🧠 Memory and Chat History

Chat history is managed manually rather than through a LangChain memory object. This avoids format conflicts that occur inside `ConversationalRetrievalChain`.

**How it works:**
1. Every message (user and assistant) is stored in `st.session_state.sessions[name]["messages"]`
2. Before each query, `format_chat_history()` in `app.py` converts the message list into a plain formatted string:
   ```
   Human: what is this document about?
   Assistant: This document discusses...
   Human: can you give more details on X?
   Assistant: ...
   ```
3. This string is passed directly to `chain.invoke({"question": ..., "chat_history": ...})`
4. The prompt instructs the LLM to use the history for conversational questions (e.g. *"what did I just ask?"*)

**Memory is session-scoped.** Each chat session has its own independent message history. Switching sessions does not share or mix history.

---

## 📂 Multi-Document Querying

When multiple files are added to the same session:
- All chunks from all files are stored together in a single FAISS vectorstore and BM25 index
- Each chunk retains its `metadata["source"]` (the original file name or URL)
- The retriever searches across all chunks regardless of source
- Source attribution in the UI shows which file each passage came from

**Example — asking across two PDFs:**
```
Uploaded: annual_report_2023.pdf  →  142 chunks
Uploaded: annual_report_2024.pdf  →  156 chunks
Total in KB: 298 chunks

Question: "How did operating expenses change between 2023 and 2024?"

Retrieved passages:
  Source 1: annual_report_2023.pdf (page 12)
  Source 2: annual_report_2024.pdf (page 9)
```

---

## 🔧 Troubleshooting

### `ModuleNotFoundError` on any import
Your virtual environment is not activated. Run:
```bash
# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```
Then re-run `streamlit run app.py`.

---

### `No API key found` error in the sidebar
Your `.env` file is missing or in the wrong folder. Make sure:
1. The file is named exactly `.env` (not `env.txt` or `.env.txt`)
2. It is in the **same folder** as `app.py`
3. It contains exactly: `OPENAI_API_KEY=sk-...`

---

### `SemanticChunker fell back to recursive splitter`
This is a warning, not an error. It means the document was too short for semantic chunking. The app continues normally using the fallback splitter.

---

### App is slow when adding documents
This is expected. Adding a document triggers:
1. OpenAI API calls for embedding every chunk (semantic chunker)
2. FAISS index rebuild
3. BM25 index rebuild
4. MultiQueryRetriever and ContextualCompression both make additional LLM calls during retrieval

For faster processing: lower `SEMANTIC_THRESHOLD_AMOUNT` in `config.py` (creates fewer, larger chunks) or switch to `RecursiveCharacterTextSplitter` by default.

---

### Answer quality is poor
Try these in order:
1. Increase `FAISS_K` and `BM25_K` in `config.py` (retrieve more candidates)
2. Increase `SEMANTIC_THRESHOLD_AMOUNT` (smaller, more precise chunks)
3. Switch `CHAT_MODEL` to `gpt-4o` for harder questions
4. Check the Source passages expander — if the right content isn't being retrieved, the retriever needs tuning, not the LLM

---

### `localhost:8501` not opening
- Make sure the terminal shows `You can now view your Streamlit app in your browser`
- Try manually opening `http://127.0.0.1:8501` instead of `localhost`
- Check if another process is using port 8501: `streamlit run app.py --server.port 8502`

---

## 🎛 Customisation

### Change the LLM model
In `config.py`:
```python
CHAT_MODEL = "gpt-4o"   # upgrade for harder questions
```

### Adjust retrieval aggressiveness
In `config.py`:
```python
FAISS_K = 8            # retrieve more candidates
BM25_K  = 8
```

### Favour keyword search over semantic search
In `config.py`:
```python
ENSEMBLE_WEIGHTS = [0.6, 0.4]   # [BM25, FAISS] — flip the balance
```

### Disable contextual compression (faster responses)
In `retriever.py`, comment out Stage 5 and return `multiquery_retriever` directly:
```python
# compressor = LLMChainExtractor.from_llm(llm)
# final_ret = ContextualCompressionRetriever(...)
return multiquery_retriever, vectorstore   # skip compression
```

### Change chunk size
In `config.py`:
```python
SEMANTIC_THRESHOLD_AMOUNT = 80   # lower = more, smaller chunks
FALLBACK_CHUNK_SIZE       = 500  # smaller fallback chunks
```

### Modify the system prompt
In `chain.py`, edit `PROMPT_TEMPLATE` to change how the LLM answers, its persona, response format, or language.

---

## ⚠️ Known Limitations

- **No persistence** — knowledge bases and chat history are lost when the Streamlit app is restarted. All data lives in `st.session_state` (in-memory only).
- **No streaming** — the full answer is generated before displaying. Long answers may feel slow.
- **Cost** — every document addition makes OpenAI embedding API calls. Every query makes multiple LLM calls (MultiQuery × 3 variants + ContextualCompression per chunk). Monitor your usage at [platform.openai.com/usage](https://platform.openai.com/usage).
- **Web scraping limits** — some websites block scraping. URLs behind logins or paywalls will not load.
- **JSON structure** — deeply nested JSON may not chunk meaningfully. Flat or list-structured JSON works best.
- **Very large PDFs** — PDFs with hundreds of pages will be slow to process and may hit token limits. Consider splitting large files before uploading.

---

## 📄 License

MIT License. Free to use, modify, and distribute with attribution.

---

## 🙌 Acknowledgements

Built with:
- [LangChain](https://github.com/langchain-ai/langchain) — LLM application framework
- [Streamlit](https://streamlit.io) — Python web UI framework
- [OpenAI](https://openai.com) — LLM and embedding models
- [FAISS](https://github.com/facebookresearch/faiss) — Vector similarity search by Meta AI
- [rank-bm25](https://github.com/dorianbrown/rank_bm25) — BM25 implementation in PythonV