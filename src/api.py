"""REST API and web app for the tax form agent.

Run locally:  uvicorn src.api:app --reload
Web app:      http://localhost:8000
API docs:     http://localhost:8000/docs
"""
import os
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src import config
from src.agent import get_agent, run
from src.ingest import ingest_file
from src.rag import get_bm25, get_store, hybrid_search, page_label

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
MAX_UPLOAD_MB = 25
_ingest_lock = threading.Lock()  # one upload at a time


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the embedding model, vector store and agent once at startup,
    # so the first request isn't slow.
    os.makedirs(config.PDF_DIR, exist_ok=True)
    get_store()
    get_agent()
    yield


app = FastAPI(
    title="State Tax Form Assistant Agent",
    description="LangGraph agent with RAG over state tax form instructions.",
    version="1.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------- models
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000,
                          examples=["When is form TC-40 due for calendar year filers?"])
    thread_id: str | None = Field(
        None, description="Reuse the thread_id from a previous answer to continue that conversation.")
    document: str | None = Field(
        None, description="Optional source_file (e.g. UT_TC-40-instructions.pdf) to focus the answer on.")


class AskResponse(BaseModel):
    answer: str
    tools_used: list[str]
    thread_id: str
    seconds: float


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, examples=["extension deadline"])
    state: str | None = Field(None, min_length=2, max_length=2, examples=["UT"])
    k: int = Field(config.TOP_K, ge=1, le=20)


class Passage(BaseModel):
    state: str
    form: str
    pages: str
    text: str


class DocumentInfo(BaseModel):
    source_file: str
    state: str
    form: str
    pages: int
    chunks: int


# ---------------------------------------------------------------- web app
@app.get("/", include_in_schema=False)
def web_app():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# ---------------------------------------------------------------- info
@app.get("/health")
def health():
    return {"status": "ok", "chunks": get_store()._collection.count(), "model": config.LLM_MODEL}


@app.get("/forms")
def forms():
    metas = get_store().get(include=["metadatas"])["metadatas"]
    return {"forms": sorted({f"{m['state']} {m['form']}" for m in metas})}


# ---------------------------------------------------------------- documents
def list_docs() -> list[DocumentInfo]:
    docs: dict[str, DocumentInfo] = {}
    for m in get_store().get(include=["metadatas"])["metadatas"]:
        d = docs.setdefault(m["source_file"], DocumentInfo(
            source_file=m["source_file"], state=m["state"], form=m["form"], pages=0, chunks=0))
        d.chunks += 1
        d.pages = max(d.pages, m.get("page_end", m["page"]))
    return sorted(docs.values(), key=lambda d: (d.state, d.form))


@app.get("/documents", response_model=list[DocumentInfo])
def documents():
    return list_docs()


@app.post("/documents", response_model=DocumentInfo, status_code=201)
async def add_document(
    file: UploadFile = File(..., description="A text-based PDF"),
    state: str = Form(..., description="Two-letter state code, e.g. UT"),
    form: str = Form(..., description="Form name, e.g. TC-40-instructions"),
):
    """Upload a PDF. It is read, split and indexed so the agent can answer from it.
    Uploading the same state and form again replaces the old version."""
    state = state.strip().upper()
    form_name = re.sub(r"[^A-Za-z0-9-]+", "-", form.strip()).strip("-")
    if not re.fullmatch(r"[A-Z]{2}", state):
        raise HTTPException(422, "State must be a two-letter code, like UT.")
    if not form_name:
        raise HTTPException(422, "Enter a form name, like TC-40-instructions.")

    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File is larger than {MAX_UPLOAD_MB} MB.")
    if not data.startswith(b"%PDF"):
        raise HTTPException(415, "This file isn't a PDF.")

    source_file = f"{state}_{form_name}.pdf"
    path = os.path.join(config.PDF_DIR, source_file)
    with _ingest_lock:
        existed = os.path.exists(path)
        with open(path, "wb") as f:
            f.write(data)
        try:
            pages, chunks = ingest_file(path, get_store())
        except Exception as e:
            if not existed:
                os.remove(path)
            raise HTTPException(422, f"Couldn't read this PDF: {str(e)[:200]}")
        get_bm25.cache_clear()  # keyword index must include the new chunks

    if chunks == 0:
        os.remove(path)
        raise HTTPException(
            422, "No text found in this PDF. It may be a scanned image; "
                 "use a PDF with selectable text.")
    return DocumentInfo(source_file=source_file, state=state, form=form_name,
                        pages=pages, chunks=chunks)


@app.get("/files/{source_file}", include_in_schema=False)
def open_file(source_file: str):
    """Serve an uploaded PDF so citations can open it at the cited page."""
    path = os.path.join(config.PDF_DIR, os.path.basename(source_file))
    if not path.lower().endswith(".pdf") or not os.path.exists(path):
        raise HTTPException(404, "Document not found.")
    return FileResponse(path, media_type="application/pdf")


@app.delete("/documents/{source_file}", status_code=204)
def delete_document(source_file: str):
    store = get_store()
    ids = store.get(where={"source_file": source_file})["ids"]
    if not ids:
        raise HTTPException(404, "Document not found.")
    with _ingest_lock:
        store.delete(ids=ids)
        path = os.path.join(config.PDF_DIR, os.path.basename(source_file))
        if os.path.exists(path):
            os.remove(path)
        get_bm25.cache_clear()


# ---------------------------------------------------------------- search & ask
@app.post("/search", response_model=list[Passage])
def search(req: SearchRequest):
    """Retrieval only (no LLM call): see which passages the agent would get."""
    docs = hybrid_search(req.query, req.state, req.k)
    return [Passage(state=d.metadata["state"], form=d.metadata["form"],
                    pages=page_label(d.metadata), text=d.page_content) for d in docs]


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    thread_id = req.thread_id or str(uuid.uuid4())
    question = req.question
    if req.document:
        state, _, form_name = req.document.removesuffix(".pdf").partition("_")
        question = f"(Answer from the {state} {form_name} document.) {question}"

    start = time.time()
    try:
        result = run(question, thread_id)
    except Exception as e:
        message = str(e)
        if "429" in message or "rate limit" in message.lower():
            raise HTTPException(429, "The AI service is busy (rate limit). Wait a minute and try again.")
        raise HTTPException(502, f"Agent error: {message[:300]}")
    return AskResponse(answer=result["answer"], tools_used=result["tools_used"],
                       thread_id=thread_id, seconds=round(time.time() - start, 1))