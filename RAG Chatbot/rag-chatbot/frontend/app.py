"""
app.py
------
Streamlit frontend for the Capstone RAG Chatbot.

Talks to the FastAPI backend over HTTP:
  - POST /upload        to ingest documents
  - POST /chat/stream    (SSE) to stream answers token-by-token
  - POST /reset          to clear conversational memory

Run with:  streamlit run app.py
"""

import os
import json
import uuid

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Capstone RAG Chatbot", page_icon="🤖", layout="wide")

# ----------------------------------------------------------------------
# Session state initialization
# ----------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    # Each message: {"role": "user"/"assistant", "content": str, "sources": [...] (optional)}
    st.session_state.messages = []

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# ----------------------------------------------------------------------
# Sidebar: document ingestion
# ----------------------------------------------------------------------
with st.sidebar:
    st.title("📚 Knowledge Base")
    st.caption("Upload PDF or TXT files to build the chatbot's knowledge base.")

    uploaded = st.file_uploader(
        "Upload documents",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if uploaded:
        for f in uploaded:
            if f.name in st.session_state.uploaded_files:
                continue  # already ingested this session
            with st.spinner(f"Ingesting {f.name}..."):
                try:
                    files = {"file": (f.name, f.getvalue(), f.type)}
                    resp = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=180)
                    resp.raise_for_status()
                    data = resp.json()
                    st.session_state.uploaded_files.append(f.name)
                    st.success(f"✅ {f.name}: {data['chunks_created']} chunks indexed")
                except Exception as e:
                    st.error(f"❌ Failed to ingest {f.name}: {e}")

    if st.session_state.uploaded_files:
        st.markdown("**Indexed files:**")
        for name in st.session_state.uploaded_files:
            st.markdown(f"- {name}")

    st.divider()
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        try:
            requests.post(f"{BACKEND_URL}/reset", json={"session_id": st.session_state.session_id})
        except Exception:
            pass
        st.rerun()

    st.divider()
    try:
        health = requests.get(f"{BACKEND_URL}/health", timeout=5).json()
        st.caption(f"🟢 Backend online — {health.get('chunk_count', '?')} chunks in store")
    except Exception:
        st.caption("🔴 Backend unreachable")

# ----------------------------------------------------------------------
# Main chat interface
# ----------------------------------------------------------------------
st.title("🤖 Advanced RAG Chatbot")
st.caption("Ask questions about the documents you've uploaded on the left.")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Sources"):
                for i, src in enumerate(msg["sources"], 1):
                    page_info = f" (page {src['page']})" if src.get("page") is not None else ""
                    st.markdown(f"**{i}. {src['source']}{page_info}**")
                    st.caption(src["snippet"] + "...")


def stream_sse(resp):
    """Parses a `text/event-stream` response line-by-line into JSON events."""
    for raw_line in resp.iter_lines(decode_unicode=True):
        if not raw_line or not raw_line.startswith("data: "):
            continue
        payload = raw_line[len("data: "):]
        try:
            yield json.loads(payload)
        except json.JSONDecodeError:
            continue


query = st.chat_input("Ask a question about your documents...")

if query:
    if not st.session_state.uploaded_files:
        st.warning("Please upload at least one document before asking a question.")
    else:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            sources_placeholder = st.container()
            full_text = ""
            sources = []

            try:
                with requests.post(
                    f"{BACKEND_URL}/chat/stream",
                    json={"session_id": st.session_state.session_id, "query": query},
                    stream=True,
                    timeout=180,
                ) as resp:
                    resp.raise_for_status()
                    for event in stream_sse(resp):
                        if event["type"] == "sources":
                            sources = event["data"]
                        elif event["type"] == "token":
                            full_text += event["data"]
                            placeholder.markdown(full_text + "▌")
                        elif event["type"] == "error":
                            full_text = f"⚠️ Error: {event['data']}"
                            break
                        elif event["type"] == "done":
                            break

                placeholder.markdown(full_text)

                if sources:
                    with sources_placeholder.expander("📄 Sources"):
                        for i, src in enumerate(sources, 1):
                            page_info = f" (page {src['page']})" if src.get("page") is not None else ""
                            st.markdown(f"**{i}. {src['source']}{page_info}**")
                            st.caption(src["snippet"] + "...")

            except Exception as e:
                full_text = f"⚠️ Could not reach backend: {e}"
                placeholder.markdown(full_text)

        st.session_state.messages.append(
            {"role": "assistant", "content": full_text, "sources": sources}
        )
