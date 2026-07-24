"""
Runs the pipeline against 3 in-scope example questions plus one
deliberately out-of-scope question (to show the "no relevant context"
path), and writes sample_answers.txt.

    export GROQ_API_KEY=...   # optional, for real generated answers
    python demo.py
"""

from rag_pipeline import build_index, answer_question
from llm import build_llm

QUESTIONS = [
    "How long do I have to request a refund after delivery?",
    "What should I do if I think someone accessed my account without permission?",
    "Can I still change a driver's tip after the delivery is done?",
    "What's the best way to bake a chocolate lava cake?",  # deliberately out of scope
]


def main() -> None:
    store = build_index()
    llm = build_llm()
    output_lines: list[str] = []

    for question in QUESTIONS:
        result = answer_question(store, question, llm)
        block = [f"Q: {question}", f"A: {result['answer']}"]
        if result["context_found"]:
            block.append("Sources: " + ", ".join(result["citations"]))
        else:
            block.append("Sources: (none -- top match scored below the relevance threshold)")
        block.append("")
        print("\n".join(block))
        output_lines.extend(block)

    with open("sample_answers.txt", "w") as f:
        f.write("\n".join(output_lines))
    print("[answers written to sample_answers.txt]")


if __name__ == "__main__":
    main()
