# Capstone RAG Chatbot

An advanced Retrieval-Augmented Generation (RAG) chatbot with:

- **FastAPI** backend orchestrating a **LangChain**-based RAG pipeline
- **ChromaDB** for local, persisted vector storage
- **Multi-query expansion + embedding re-ranking** for higher-quality retrieval
- **Conversational memory** so users can ask follow-up questions
- **Streamlit** frontend with token-by-token streaming and an expandable "Sources" panel
- Full **Docker** / **docker-compose** setup for one-command deployment

---

## 1. Project Structure

```
rag-chatbot/
├── backend/
│   ├── main.py          # FastAPI app: /upload, /chat/stream, /reset, /health
│   ├── rag_engine.py     # Ingestion, multi-query retrieval, re-ranking, streaming generation
│   └── schemas.py        # Pydantic request/response models
├── frontend/
│   └── app.py             # Streamlit chat UI
├── requirements.txt
├── Dockerfile.backend
├── Dockerfile.frontend
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## 2. How It Works

1. **Ingestion** — Uploaded PDF/TXT files are loaded (`PyPDFLoader` / `TextLoader`), split with
   `RecursiveCharacterTextSplitter` (chunk_size=1000, overlap=200), embedded with OpenAI embeddings,
   and stored in a local, persisted ChromaDB collection.
2. **Retrieval** — For each user question:
   - The LLM generates 3 alternative phrasings of the question (**multi-query expansion**).
   - Chroma similarity search runs for the original + all variants; results are merged and deduplicated.
   - The merged pool is **re-ranked** by cosine similarity against the *original* question's embedding,
     and the top `k=5` chunks are kept.
3. **Generation** — The top chunks + prior conversation turns (**conversational memory**) are assembled
   into a prompt and streamed token-by-token from the chat model back to the UI via **Server-Sent Events**.
4. **UI** — Streamlit renders the streamed answer live and shows the retrieved chunks in an expandable
   "Sources" section underneath each answer.

---

## 3. Environment Variables

Copy `.env.example` to `.env` and fill in your key:

| Variable            | Required | Default                     | Description                                   |
|---------------------|----------|------------------------------|------------------------------------------------|
| `OPENAI_API_KEY`    | ✅ Yes   | —                             | Your OpenAI API key                            |
| `CHAT_MODEL`        | No       | `gpt-3.5-turbo`               | Chat/generation model                          |
| `EMBEDDING_MODEL`   | No       | `text-embedding-3-small`      | Embedding model                                |
| `CHUNK_SIZE`        | No       | `1000`                        | Text splitter chunk size                       |
| `CHUNK_OVERLAP`     | No       | `200`                         | Text splitter chunk overlap                    |
| `TOP_K`             | No       | `5`                           | Number of chunks retrieved per query           |
| `MULTI_QUERY_N`     | No       | `3`                           | Number of LLM-generated query rewrites         |
| `BACKEND_URL`       | No       | `http://localhost:8000`       | Frontend → backend URL (set to `http://backend:8000` in Docker) |

> **Using an open-source model instead?** Swap `ChatOpenAI` / `OpenAIEmbeddings` in `rag_engine.py`
> for `langchain_community.chat_models.ChatHuggingFace` and `HuggingFaceEmbeddings` (e.g. Mistral-7B-Instruct
> + `sentence-transformers/all-MiniLM-L6-v2`). The rest of the pipeline (splitting, retrieval, re-ranking,
> streaming interface) does not need to change.

---

## 4. Local Setup (without Docker)

**Prerequisites:** Python 3.10+

```bash
# 1. Clone/copy the project, then from the rag-chatbot/ root:
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# edit .env and set OPENAI_API_KEY

# 2. Start the backend (Terminal 1)
cd backend
export $(grep -v '^#' ../.env | xargs)   # loads .env vars into the shell (Linux/macOS)
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 3. Start the frontend (Terminal 2)
cd frontend
export BACKEND_URL=http://localhost:8000
streamlit run app.py
```

Open **http://localhost:8501**, upload a PDF/TXT in the sidebar, then start chatting.

---

## 5. Running with Docker Compose (recommended)

```bash
cp .env.example .env
# edit .env and set OPENAI_API_KEY

docker compose up --build
```

- Backend: `http://localhost:8000` (docs at `/docs`)
- Frontend: `http://localhost:8501`
- The vector store persists in a named Docker volume (`chroma_data`), so it survives restarts.

Stop with `docker compose down` (add `-v` to also wipe the vector store volume).

---

## 6. Deploying to AWS EC2

**Prerequisites:** an EC2 instance (Ubuntu 22.04, t3.medium or larger recommended), a security group
allowing inbound TCP on ports `22`, `8000`, and `8501`.

```bash
# 1. SSH into the instance
ssh -i your-key.pem ubuntu@<EC2_PUBLIC_IP>

# 2. Install Docker + Compose plugin
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
sudo usermod -aG docker $USER && newgrp docker

# 3. Copy the project onto the instance (from your local machine)
scp -i your-key.pem -r rag-chatbot ubuntu@<EC2_PUBLIC_IP>:~/rag-chatbot

# 4. On the instance: configure env and launch
cd ~/rag-chatbot
cp .env.example .env
nano .env   # set OPENAI_API_KEY

docker compose up -d --build

# 5. Verify
curl http://localhost:8000/health
```

Visit `http://<EC2_PUBLIC_IP>:8501` in your browser.

**Production hardening tips:**
- Put an Nginx reverse proxy (or an Application Load Balancer) in front of both services with TLS
  termination, and only expose port 443 publicly.
- Restrict the FastAPI CORS `allow_origins` in `main.py` to your actual frontend domain.
- Use a managed secret store (AWS Secrets Manager / SSM Parameter Store) for `OPENAI_API_KEY`
  instead of a plain `.env` file.
- Attach an EBS volume (or keep the Docker named volume) so `chroma_store` persists across
  instance replacement.

---

## 7. Deploying to Google Cloud Run

Cloud Run runs one container per service, so deploy the backend and frontend as **two separate
services** built from `Dockerfile.backend` and `Dockerfile.frontend`.

```bash
# 0. One-time setup
gcloud auth login
gcloud config set project <YOUR_GCP_PROJECT_ID>
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com

# 1. Build & push the backend image
gcloud builds submit --tag gcr.io/<YOUR_GCP_PROJECT_ID>/rag-backend -f Dockerfile.backend .

# 2. Deploy the backend
gcloud run deploy rag-backend \
  --image gcr.io/<YOUR_GCP_PROJECT_ID>/rag-backend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8000 \
  --set-env-vars OPENAI_API_KEY=<YOUR_KEY>,CHAT_MODEL=gpt-3.5-turbo

# Note the backend service URL printed after deploy, e.g.:
#   https://rag-backend-xxxxx-uc.a.run.app

# 3. Build & push the frontend image
gcloud builds submit --tag gcr.io/<YOUR_GCP_PROJECT_ID>/rag-frontend -f Dockerfile.frontend .

# 4. Deploy the frontend, pointing it at the backend URL from step 2
gcloud run deploy rag-frontend \
  --image gcr.io/<YOUR_GCP_PROJECT_ID>/rag-frontend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8501 \
  --set-env-vars BACKEND_URL=https://rag-backend-xxxxx-uc.a.run.app
```

Cloud Run will print a public HTTPS URL for `rag-frontend` — that's your chatbot.

**Important Cloud Run caveat:** Cloud Run containers are stateless/ephemeral by default, so the local
ChromaDB `chroma_store` directory **will not persist** across revisions or scale-to-zero events. For a
capstone demo this is usually fine (re-upload your PDFs after a cold start), but for anything longer-lived:
- Mount a [Cloud Run volume backed by Cloud Storage FUSE](https://cloud.google.com/run/docs/configuring/services/cloud-storage-volume-mounts), or
- Swap ChromaDB for a managed vector DB (e.g. a small Postgres+pgvector instance, or Chroma's hosted offering).

---

## 8. API Reference (backend)

Once running, interactive docs are available at `http://<backend-host>:8000/docs`.

| Method | Path            | Description                                      |
|--------|-----------------|---------------------------------------------------|
| GET    | `/health`        | Health check + current chunk count               |
| POST   | `/upload`        | Multipart file upload (`file`) → ingests a PDF/TXT |
| POST   | `/chat/stream`    | `{session_id, query}` → SSE stream of `sources`/`token`/`done` events |
| POST   | `/reset`          | `{session_id}` → clears that session's conversational memory |

---

## 9. Notes for Your Capstone Report

- **Chunking strategy:** `RecursiveCharacterTextSplitter`, size 1000 / overlap 200, chosen to balance
  retrieval precision against context coherence.
- **Retrieval upgrade over naive RAG:** multi-query expansion (query rewriting) + embedding-based
  re-ranking against the original query, which reduces sensitivity to how the user phrases a question.
- **Memory:** implemented as an explicit per-session buffer (equivalent to LangChain's
  `ConversationBufferMemory`) folded into the prompt on each turn, capped at the last 10 turns to bound
  token usage.
- **Streaming:** implemented with Server-Sent Events end-to-end (FastAPI `StreamingResponse` →
  `requests` streaming client in Streamlit), rather than polling, for real token-by-token UX.
