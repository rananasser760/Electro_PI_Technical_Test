"""
Pluggable embedding backends for the RAG pipeline. `rag_pipeline.py` only
ever depends on the standard LangChain Embeddings interface
(`embed_documents(texts)` / `embed_query(text)`), so swapping the backend
never touches retrieval or generation code -- same decoupling idea as the
STT/TTS swap in Section 1.
"""

from __future__ import annotations
import os

from langchain_core.embeddings import Embeddings


class LocalTfidfEmbeddings(Embeddings):
    """No-download, no-API-key embedding backend (TF-IDF + SVD), so this
    pipeline runs end to end in a fully offline/sandboxed environment.
    Swap in `HuggingFaceEmbeddings` (real neural embeddings, see
    `build_huggingface_embeddings` below) once you have network access --
    nothing else in the pipeline changes.
    """

    def __init__(self, n_components: int = 128):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._n_components = n_components
        self._svd = None
        self._fitted = False

    def _fit(self, texts: list[str]) -> None:
        from sklearn.decomposition import TruncatedSVD
        tfidf = self._vectorizer.fit_transform(texts)
        n_components = max(1, min(self._n_components, tfidf.shape[1] - 1, tfidf.shape[0] - 1))
        self._svd = TruncatedSVD(n_components=n_components, random_state=0)
        self._svd.fit(tfidf)
        self._fitted = True

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not self._fitted:
            self._fit(texts)
        tfidf = self._vectorizer.transform(texts)
        return self._svd.transform(tfidf).tolist()

    def embed_query(self, text: str) -> list[float]:
        tfidf = self._vectorizer.transform([text])
        return self._svd.transform(tfidf)[0].tolist()


def build_huggingface_embeddings():
    """Real neural embeddings -- needs `langchain-huggingface` +
    `sentence-transformers` and network access to pull the model weights
    the first time it runs."""
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def get_embeddings():
    if os.environ.get("USE_HF_EMBEDDINGS") == "1":
        return build_huggingface_embeddings()
    return LocalTfidfEmbeddings()
