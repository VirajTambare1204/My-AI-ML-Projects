"""
main.py
-------
FastAPI application exposing the RAG chatbot as an HTTP/SSE API.

Endpoints:
  POST /upload         -> ingest a PDF/TXT file into the vector store
  POST /chat/stream     -> stream a token-by-token answer (Server-Sent Events)
  POST /reset           -> clear conversational memory for a session
  GET  /health          -> liveness/readiness probe + collection stats

Run directly with:  uvicorn main:app --host 0.0.0.0 --port 8000
"""

import os
import json
import shutil
import tempfile
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv

from rag_engine import engine
from schemas import ChatRequest, IngestResponse, ResetRequest

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

app = FastAPI(
    title="Capstone RAG Chatbot API",
    description="Backend for an advanced Retrieval-Augmented Generation chatbot.",
    version="1.0.0",
)

# Allow the Streamlit frontend (running on a different port/host/container)
# to call this API directly from the browser if needed.
# NOTE: tighten allow_origins to your actual frontend URL in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {"pdf", "txt"}


@app.get("/health")
def health():
    """Liveness/readiness probe. Also useful for the Streamlit sidebar status indicator."""
    return {"status": "ok", **engine.collection_stats()}


@app.post("/upload", response_model=IngestResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Accepts a PDF or TXT upload, writes it to a temp file (LangChain's
    document loaders need a filesystem path), runs it through the ingestion
    pipeline (load -> split -> embed -> upsert into Chroma), then cleans up.
    """
    ext = file.filename.lower().split(".")[-1]
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '.{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}")

    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, file.filename)

    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        chunks_created = engine.ingest_file(tmp_path, file.filename)
        return IngestResponse(filename=file.filename, chunks_created=chunks_created)

    except Exception as e:
        logger.exception("Ingestion failed")
        raise HTTPException(500, f"Failed to ingest file: {e}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    Streams the answer as Server-Sent Events (SSE). Each event is a JSON
    payload with a 'type' field: 'sources' | 'token' | 'done' | 'error'.
    The Streamlit frontend parses these to render sources + streamed tokens
    incrementally.
    """

    def event_generator():
        try:
            for event in engine.stream_answer(req.session_id, req.query):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            logger.exception("Streaming generation failed")
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/reset")
def reset(req: ResetRequest):
    """Clears the conversational memory (ConversationBufferMemory equivalent) for a session."""
    engine.reset_memory(req.session_id)
    return {"status": "cleared", "session_id": req.session_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
