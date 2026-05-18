

import os
import json
import tempfile
from typing import List

from langchain_classic.schema import Document
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader


def load_plain_text(text: str, name: str = "text_input") -> List[Document]:
    """Plain text string → Document."""
    return [Document(page_content=text, metadata={"source": name})]


def load_txt_file(uploaded_file) -> List[Document]:
    """Streamlit uploaded .txt file → Document."""
    content = uploaded_file.read().decode("utf-8")
    return [Document(page_content=content, metadata={"source": uploaded_file.name})]


def load_pdf_file(uploaded_file) -> List[Document]:
    """Streamlit uploaded PDF → list of Documents (one per page)."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = uploaded_file.name
        return docs
    finally:
        os.unlink(tmp_path)


def load_url(url: str) -> List[Document]:
    """Web URL → Documents (scrapes visible text)."""
    loader = WebBaseLoader(url)
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = url
    return docs


def load_json_file(uploaded_file) -> List[Document]:
    """
    Streamlit uploaded JSON → Documents.
    - List JSON  → one Document per item
    - Dict JSON  → one Document for the whole object
    """
    data = json.loads(uploaded_file.read().decode("utf-8"))
    docs = []

    if isinstance(data, list):
        for i, item in enumerate(data):
            text = (
                json.dumps(item, indent=2)
                if isinstance(item, (dict, list))
                else str(item)
            )
            docs.append(Document(
                page_content=text,
                metadata={"source": uploaded_file.name, "index": i},
            ))
    elif isinstance(data, dict):
        docs.append(Document(
            page_content=json.dumps(data, indent=2),
            metadata={"source": uploaded_file.name},
        ))
    else:
        docs.append(Document(
            page_content=str(data),
            metadata={"source": uploaded_file.name},
        ))

    return docs