"""Retrieve relevant form instructions and answer with cited sources."""
import re
from functools import lru_cache

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src import config

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You answer questions about US state tax forms using ONLY the context below. "
     "Cite every fact as [STATE form, p.N]. If the context does not contain the "
     "answer, say \"I couldn't find that in the loaded forms\" instead of guessing.\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])


@lru_cache(maxsize=1)
def get_store() -> Chroma:
    """Load the vector store once and reuse it for every question."""
    return Chroma(
        collection_name=config.COLLECTION,
        embedding_function=HuggingFaceEmbeddings(model_name=config.EMBED_MODEL),
        persist_directory=config.CHROMA_DIR,
    )


@lru_cache(maxsize=1)
def get_llm():
    return init_chat_model(config.LLM_MODEL, temperature=0)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@lru_cache(maxsize=1)
def get_bm25():
    """Keyword index over the same chunks stored in ChromaDB."""
    data = get_store().get()
    docs = [Document(page_content=t, metadata=m) for t, m in zip(data["documents"], data["metadatas"])]
    if not docs:
        return None, []  # BM25 can't be built over zero documents
    return BM25Okapi([tokenize(d.page_content) for d in docs]), docs


def hybrid_search(question: str, state: str | None, k: int) -> list[Document]:
    """Combine meaning-based (vector) and keyword (BM25) search with
    Reciprocal Rank Fusion: chunks ranked high by either method win."""
    pool = k * 4
    filter_ = {"state": state.upper()} if state else None
    vector_hits = get_store().similarity_search(question, k=pool, filter=filter_)

    bm25, all_docs = get_bm25()
    if bm25 is None:
        return vector_hits[:k]
    scores = bm25.get_scores(tokenize(question))
    ranked = sorted(range(len(all_docs)), key=lambda i: scores[i], reverse=True)
    keyword_hits = [all_docs[i] for i in ranked
                    if not state or all_docs[i].metadata["state"] == state.upper()][:pool]

    fused, by_text = {}, {}
    for hits in (vector_hits, keyword_hits):
        for rank, doc in enumerate(hits):
            fused[doc.page_content] = fused.get(doc.page_content, 0) + 1 / (60 + rank)
            by_text[doc.page_content] = doc
    best = sorted(fused, key=fused.get, reverse=True)[:k]
    return [by_text[text] for text in best]


def page_label(meta: dict) -> str:
    """'p.4', or 'p.4-5' when a chunk runs across a page break."""
    start, end = meta["page"], meta.get("page_end", meta["page"])
    return f"p.{start}" if start == end else f"p.{start}-{end}"


def format_context(docs) -> str:
    return "\n\n".join(
        f"[{d.metadata['state']} {d.metadata['form']}, {page_label(d.metadata)}]\n{d.page_content}"
        for d in docs
    )


def to_text(content) -> str:
    """Gemini can return a list of content blocks; keep only the text."""
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict))


def answer(question: str, state: str | None = None) -> dict:
    """Answer a question. Pass state='NE' to search only that state's forms."""
    docs = hybrid_search(question, state, config.TOP_K)

    reply = (PROMPT | get_llm()).invoke({"context": format_context(docs), "question": question})

    sources = sorted({f"{d.metadata['state']} {d.metadata['form']} {page_label(d.metadata)}" for d in docs})
    return {"answer": to_text(reply.content), "sources": sources}