"""
Task 2.1 — RAG pipeline over the food-delivery support docs in documents/.

- Chunks + embeds documents/*.md into a FAISS vector store.
- Retrieves the top-k chunks for a question.
- Explicitly refuses to answer (no LLM call at all) when nothing relevant
  is found, instead of letting the model guess.
- Answers with citations back to the source file + chunk index.
"""

from __future__ import annotations
import glob
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from embeddings import get_embeddings

DOCS_DIR = os.path.join(os.path.dirname(__file__), "documents")
CHUNK_SIZE = 400
CHUNK_OVERLAP = 60
TOP_K = 3

# Similarity below this (on a 0-1, higher-is-better scale derived from FAISS
# L2 distance) means "nothing relevant" -> refuse instead of calling the LLM.
# Tuned for the default TF-IDF backend; re-tune if you swap to neural
# embeddings, since their distance distribution is different.
RELEVANCE_THRESHOLD = 0.58


def load_and_chunk() -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs: list[Document] = []
    for path in sorted(glob.glob(os.path.join(DOCS_DIR, "*.md"))):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        source = os.path.basename(path)
        for i, chunk in enumerate(splitter.split_text(text)):
            docs.append(Document(page_content=chunk, metadata={"source": source, "chunk": i}))
    return docs


def build_index() -> FAISS:
    docs = load_and_chunk()
    embeddings = get_embeddings()
    return FAISS.from_documents(docs, embeddings)


def retrieve(store: FAISS, question: str, k: int = TOP_K):
    """Returns [(Document, similarity), ...], similarity in ~[0, 1], higher
    is better. Computed manually from raw L2 distance so the threshold
    logic works the same regardless of which embedding backend is plugged
    in (FAISS's built-in score normalization assumes a specific metric)."""
    results = store.similarity_search_with_score(question, k=k)
    return [(doc, 1.0 / (1.0 + dist)) for doc, dist in results]


def format_context(scored_docs: list[tuple[Document, float]]) -> str:
    blocks = []
    for idx, (doc, _score) in enumerate(scored_docs, start=1):
        blocks.append(
            f"[{idx}] (source: {doc.metadata['source']}, chunk {doc.metadata['chunk']})\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(blocks)


def answer_question(store: FAISS, question: str, llm, k: int = TOP_K,
                     threshold: float = RELEVANCE_THRESHOLD) -> dict:
    scored = retrieve(store, question, k=k)

    if not scored or scored[0][1] < threshold:
        return {
            "answer": "I don't have information about that in the provided documents.",
            "citations": [],
            "context_found": False,
            "top_score": scored[0][1] if scored else None,
        }

    context = format_context(scored)
    answer_text = llm.generate(question, context)
    citations = [f"{doc.metadata['source']} (chunk {doc.metadata['chunk']})" for doc, _ in scored]
    return {
        "answer": answer_text,
        "citations": citations,
        "context_found": True,
        "top_score": scored[0][1],
    }
