"""
LLM backends for the answer-generation step. Both expose the same
`generate(question, context) -> str` interface, so `build_llm()` is the
only place that changes when swapping providers.
"""

from __future__ import annotations
import os

SYSTEM_PROMPT = (
    "Answer the user's question using ONLY the numbered context below. "
    "Cite the sources you used inline like [1], [2], matching the context "
    "numbers. If the context does not actually contain the answer, say so "
    "plainly instead of guessing."
)


class GroqLLM:
    def __init__(self) -> None:
        from openai import OpenAI
        self.client = OpenAI(
            api_key=os.environ["GROQ_API_KEY"],
            base_url="https://api.groq.com/openai/v1",
        )
        self.model = "llama-3.3-70b-versatile"

    def generate(self, question: str, context: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]
        resp = self.client.chat.completions.create(model=self.model, messages=messages, temperature=0)
        return resp.choices[0].message.content


class ExtractiveMockLLM:
    """No API key needed: pulls the most relevant sentence out of the
    top-ranked chunk and returns it with a citation marker, so the
    retrieval + citation + no-context logic is demonstrable fully offline.
    """

    def generate(self, question: str, context: str) -> str:
        import re
        q_words = {w.lower().strip("?.,'\"") for w in question.split() if len(w) > 3}
        best_sentence, best_marker, best_overlap = None, "[1]", -1

        # Split on citation markers, not blank lines -- a chunk's own text
        # can contain blank lines (e.g. after a markdown heading), which
        # would otherwise split one block into several and desync the
        # marker from its body.
        blocks = re.split(r"\n(?=\[\d+\] \(source:)", context)
        for block in blocks:
            header, _, body = block.partition("\n")
            marker = header.split("]")[0] + "]"  # e.g. "[1]"
            body = "\n".join(line for line in body.splitlines() if not line.strip().startswith("#"))
            for sentence in body.replace("\n", " ").split("."):
                sentence = sentence.strip()
                if not sentence:
                    continue
                overlap = len(q_words & {w.lower().strip("?.,'\"") for w in sentence.split()})
                if overlap > best_overlap:
                    best_sentence, best_marker, best_overlap = sentence, marker, overlap

        if best_sentence is None:
            return "I don't have information about that in the provided documents."
        return f"{best_sentence}. {best_marker}"


def build_llm():
    if os.environ.get("GROQ_API_KEY"):
        return GroqLLM()
    return ExtractiveMockLLM()
