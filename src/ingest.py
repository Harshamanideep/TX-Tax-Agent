"""Load state tax form PDFs, split them into chunks and store them in ChromaDB.

Name each PDF as STATE_FORM.pdf, for example NE_1040N-instructions.pdf,
so every chunk knows which state and form it came from.

Run:  python -m src.ingest
      python -m src.ingest --if-empty   (skip if the vector store already has data;
                                         used when the Docker container starts)
"""
import argparse
import bisect
import os
import re
import time
import unicodedata

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src import config


def parse_filename(filename: str) -> dict:
    """'NE_1040N-instructions.pdf' -> {'state': 'NE', 'form': '1040N-instructions'}"""
    stem = os.path.splitext(filename)[0]
    state, _, form = stem.partition("_")
    return {"state": state.upper(), "form": form or "unknown", "source_file": filename}


def clean_text(text: str) -> str:
    """Fix PDF extraction quirks so search can match words.
    Many government PDFs store 'fi', 'fl', 'ff' as single ligature characters,
    which come out as 'ﬁ le' instead of 'file'."""
    text = re.sub(r"([\ufb00-\ufb04]) ", r"\1", text)  # 'ﬁ le' -> 'ﬁle'
    text = unicodedata.normalize("NFKC", text)            # 'ﬁle'  -> 'file'
    return re.sub(r"[ \t]+", " ", text)


def chunk_pdf(path: str, meta: dict, splitter) -> tuple[int, list]:
    """Split a whole PDF as one text, so sentences and lists that continue
    onto the next page stay in the same chunk. Each chunk is labelled
    with the pages it covers."""
    pages = PyPDFLoader(path).load()

    text, starts = "", []
    for page in pages:
        starts.append(len(text))
        text += clean_text(page.page_content) + "\n"

    chunks = splitter.create_documents([text], metadatas=[meta])
    for chunk in chunks:
        first = chunk.metadata["start_index"]
        last = first + len(chunk.page_content) - 1
        # 1-based page numbers; a chunk can run across a page break
        chunk.metadata["page"] = bisect.bisect_right(starts, first)
        chunk.metadata["page_end"] = bisect.bisect_right(starts, last)
    return len(pages), [c for c in chunks if c.page_content.strip()]


def make_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        add_start_index=True,
    )


def ingest_file(path: str, store) -> tuple[int, int]:
    """Add (or replace) one PDF in an existing vector store.
    Used when a user uploads a document through the web app.
    Returns (pages, chunks)."""
    meta = parse_filename(os.path.basename(path))
    n_pages, chunks = chunk_pdf(path, meta, make_splitter())
    old_ids = store.get(where={"source_file": meta["source_file"]})["ids"]
    if old_ids:
        store.delete(ids=old_ids)
    if chunks:
        store.add_documents(chunks)
    return n_pages, len(chunks)


def store_has_data() -> bool:
    if not os.path.isdir(config.CHROMA_DIR):
        return False
    import chromadb
    try:
        client = chromadb.PersistentClient(path=config.CHROMA_DIR)
        return client.get_collection(config.COLLECTION).count() > 0
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--if-empty", action="store_true")
    args = parser.parse_args()
    if args.if_empty and store_has_data():
        print("Vector store already has data; skipping ingestion.")
        return

    start = time.time()
    splitter = make_splitter()

    print(f"Loading PDFs from {config.PDF_DIR}/ ...")
    total_pages, chunks, states = 0, [], set()
    for filename in sorted(os.listdir(config.PDF_DIR)):
        if not filename.lower().endswith(".pdf"):
            continue
        meta = parse_filename(filename)
        n_pages, file_chunks = chunk_pdf(os.path.join(config.PDF_DIR, filename), meta, splitter)
        total_pages += n_pages
        chunks.extend(file_chunks)
        states.add(meta["state"])
        print(f"  loaded {filename}: {n_pages} pages, {len(file_chunks)} chunks")

    if not chunks:
        raise SystemExit(f"No PDFs found in {config.PDF_DIR}/. Download some first.")

    print(f"Embedding {len(chunks)} chunks (first run downloads the model) ...")
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBED_MODEL)

    # Rebuild from scratch so re-running doesn't store duplicate chunks.
    Chroma(
        collection_name=config.COLLECTION,
        embedding_function=embeddings,
        persist_directory=config.CHROMA_DIR,
    ).delete_collection()

    Chroma.from_documents(
        chunks,
        embeddings,
        collection_name=config.COLLECTION,
        persist_directory=config.CHROMA_DIR,
    )

    # These are your resume numbers: write them down.
    print(
        f"\nDone in {time.time() - start:.1f}s: {total_pages} pages, "
        f"{len(chunks)} chunks, states: {', '.join(sorted(states))}"
    )


if __name__ == "__main__":
    main()