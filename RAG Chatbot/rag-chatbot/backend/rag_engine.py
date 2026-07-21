"""
rag_engine.py
--------------
Core Retrieval-Augmented Generation (RAG) logic.

This module encapsulates:
  1. Document ingestion (PDF/TXT loading + chunking)
  2. Embedding + persistence into a local ChromaDB vector store
  3. Advanced retrieval: multi-query expansion + embedding-based re-ranking
  4. Conversational memory (per-session buffer)
  5. Streaming generation via an OpenAI chat model

Design notes:
  - LangChain is used as a set of composable primitives (loaders, splitters,
    vector store wrapper) rather than a monolithic pre-built chain, so the
    retrieval logic (multi-query + re-rank) stays fully transparent and easy
    to explain/defend in a viva or demo.
  - ChromaDB runs with local, on-disk persistence so the whole system works
    without any external managed vector DB - ideal for a capstone deployment
    on a single EC2 instance or Cloud Run container.
"""

import os
import uuid
import logging
from typing import List, Dict, Generator

import numpy as np
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document, HumanMessage, AIMessage, SystemMessage

logger = logging.getLogger("rag_engine")
logging.basicConfig(level=logging.INFO)

# --------------------------------------------------------------------------
# Configuration (all overridable via environment variables / .env)
# --------------------------------------------------------------------------
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "capstone_rag_collection")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-3.5-turbo")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
TOP_K = int(os.getenv("TOP_K", 5))
MULTI_QUERY_N = int(os.getenv("MULTI_QUERY_N", 3))  # number of LLM-generated query rewrites


class SessionMemory:
    """
    Minimal conversational memory store (equivalent in spirit to LangChain's
    ConversationBufferMemory, implemented explicitly here so the token
    streaming code can decide exactly what goes into the prompt).

    For a capstone/demo deployment, an in-process dict keyed by session_id
    is sufficient and keeps the stack dependency-free. In a real production
    system this would be swapped for Redis / a database-backed store so
    memory survives process restarts and scales across multiple workers.
    """

    def __init__(self, max_turns: int = 10):
        self._store: Dict[str, List[Dict[str, str]]] = {}
        self.max_turns = max_turns  # user+assistant pairs to retain

    def get(self, session_id: str) -> List[Dict[str, str]]:
        return self._store.get(session_id, [])

    def append(self, session_id: str, role: str, content: str):
        history = self._store.setdefault(session_id, [])
        history.append({"role": role, "content": content})
        if len(history) > self.max_turns * 2:
            self._store[session_id] = history[-self.max_turns * 2:]

    def clear(self, session_id: str):
        self._store.pop(session_id, None)


class RAGEngine:
    """Encapsulates the full RAG pipeline: ingestion, retrieval, and generation."""

    def __init__(self):
        if not os.getenv("OPENAI_API_KEY"):
            logger.warning(
                "OPENAI_API_KEY is not set. Set it in your environment or .env file "
                "before making embedding/chat calls."
            )

        self.embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        self.llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.2, streaming=True)
        # A separate LLM handle used for auxiliary tasks (query rewriting).
        # Same model here, but kept distinct so you can swap in a cheaper
        # model for this step without touching the main generation call.
        self.aux_llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.3)

        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        self.vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        self.memory = SessionMemory()

    # ----------------------------------------------------------------
    # 1. INGESTION
    # ----------------------------------------------------------------
    def ingest_file(self, file_path: str, original_filename: str) -> int:
        """
        Loads a PDF or TXT file, splits it into overlapping chunks, embeds
        each chunk, and upserts it into the Chroma collection.
        Returns the number of chunks created.
        """
        ext = original_filename.lower().split(".")[-1]

        if ext == "pdf":
            loader = PyPDFLoader(file_path)
        elif ext == "txt":
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file type: .{ext}. Only PDF and TXT are supported.")

        raw_docs: List[Document] = loader.load()

        # Tag every page/doc with its source filename BEFORE splitting, so
        # provenance survives chunking (used later to render "Sources").
        for d in raw_docs:
            d.metadata["source"] = original_filename

        chunks = self.splitter.split_documents(raw_docs)

        # Stable-ish unique ids so repeated uploads don't silently collide.
        ids = [f"{original_filename}-{i}-{uuid.uuid4().hex[:8]}" for i in range(len(chunks))]

        self.vectorstore.add_documents(documents=chunks, ids=ids)
        self.vectorstore.persist()

        logger.info(f"Ingested '{original_filename}' -> {len(chunks)} chunks.")
        return len(chunks)

    # ----------------------------------------------------------------
    # 2. ADVANCED RETRIEVAL: multi-query expansion + embedding re-rank
    # ----------------------------------------------------------------
    def _expand_query(self, query: str) -> List[str]:
        """
        Uses the LLM to generate paraphrased/alternative versions of the
        user's question. Retrieving for each variant and merging results
        mitigates the "vocabulary mismatch" problem of pure dense retrieval
        (e.g. user says "cost" but the document says "pricing").
        """
        prompt = (
            f"Generate {MULTI_QUERY_N} different rephrasings of the following "
            f"question that preserve its original meaning but vary the wording "
            f"and perspective. Return ONLY the rephrasings, one per line, with "
            f"no numbering or extra commentary.\n\nQuestion: {query}"
        )
        try:
            resp = self.aux_llm.invoke([HumanMessage(content=prompt)])
            variants = [line.strip("-• \t") for line in resp.content.split("\n") if line.strip()]
            return [query] + variants[:MULTI_QUERY_N]
        except Exception as e:
            logger.warning(f"Multi-query expansion failed, falling back to original query only: {e}")
            return [query]

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom else 0.0

    def _rerank(self, query: str, docs: List[Document]) -> List[Document]:
        """
        Dependency-light re-ranking step: re-embeds the original query and
        each retrieved chunk, then sorts chunks by cosine similarity to the
        *original* query (not the expanded ones). This counteracts drift
        introduced by the multi-query step and surfaces the chunks most
        faithful to what the user actually asked.

        (A cross-encoder, e.g. ms-marco-MiniLM, would give higher-fidelity
        re-ranking, but is intentionally omitted to keep the container
        lightweight and CPU-only friendly for a capstone deployment.)
        """
        if not docs:
            return docs

        query_vec = np.array(self.embeddings.embed_query(query))
        doc_vecs = self.embeddings.embed_documents([d.page_content for d in docs])

        scored = [
            (self._cosine_sim(query_vec, np.array(vec)), doc)
            for vec, doc in zip(doc_vecs, docs)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored]

    def retrieve(self, query: str, k: int = TOP_K) -> List[Document]:
        """
        Full advanced retrieval pipeline:
          1. Expand the query into several paraphrases (multi-query).
          2. Run similarity search in Chroma for each variant.
          3. Deduplicate the merged candidate pool.
          4. Re-rank the pool against the original query.
          5. Return the top-k re-ranked chunks.
        """
        queries = self._expand_query(query)

        candidate_pool: Dict[str, Document] = {}
        for q in queries:
            results = self.vectorstore.similarity_search(q, k=k)
            for doc in results:
                key = doc.page_content[:200]  # cheap dedupe key
                candidate_pool[key] = doc

        candidates = list(candidate_pool.values())
        reranked = self._rerank(query, candidates)
        return reranked[:k]

    # ----------------------------------------------------------------
    # 3. GENERATION (streaming) WITH CONVERSATIONAL MEMORY
    # ----------------------------------------------------------------
    def _build_messages(self, session_id: str, query: str, context_docs: List[Document]):
        context_text = "\n\n---\n\n".join(
            f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
            for d in context_docs
        )

        system_prompt = (
            "You are a helpful assistant answering questions using ONLY the "
            "provided context from the user's uploaded documents. "
            "If the answer is not contained in the context, say you don't know "
            "rather than making something up. Cite the source filename when relevant.\n\n"
            f"CONTEXT:\n{context_text}"
        )

        messages = [SystemMessage(content=system_prompt)]

        # Fold in prior turns so follow-up questions ("what about the second
        # point?") resolve correctly - this is the ConversationBufferMemory
        # equivalent.
        for turn in self.memory.get(session_id):
            if turn["role"] == "user":
                messages.append(HumanMessage(content=turn["content"]))
            else:
                messages.append(AIMessage(content=turn["content"]))

        messages.append(HumanMessage(content=query))
        return messages

    def stream_answer(self, session_id: str, query: str) -> Generator[Dict, None, None]:
        """
        Generator that first yields a 'sources' event, then yields 'token'
        events as the LLM streams its answer, and finally a 'done' event.
        This shape is trivially wrapped into Server-Sent Events (SSE) by the
        FastAPI layer.
        """
        docs = self.retrieve(query)

        sources = [
            {
                "source": d.metadata.get("source", "unknown"),
                "page": d.metadata.get("page", None),
                "snippet": d.page_content[:300],
            }
            for d in docs
        ]
        yield {"type": "sources", "data": sources}

        messages = self._build_messages(session_id, query, docs)

        full_response = ""
        for chunk in self.llm.stream(messages):
            token = chunk.content or ""
            if token:
                full_response += token
                yield {"type": "token", "data": token}

        # Persist this turn into memory for future follow-up questions
        self.memory.append(session_id, "user", query)
        self.memory.append(session_id, "assistant", full_response)

        yield {"type": "done", "data": None}

    def reset_memory(self, session_id: str):
        self.memory.clear(session_id)

    def collection_stats(self) -> Dict:
        try:
            count = self.vectorstore._collection.count()
        except Exception:
            count = -1
        return {"collection": COLLECTION_NAME, "chunk_count": count}


# Singleton engine instance shared across all requests in this process
engine = RAGEngine()
