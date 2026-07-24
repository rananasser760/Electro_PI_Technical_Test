"""
Optional extra data point for the write-up: same 5 prompts, same model
family, but a GGUF build running through llama.cpp instead of
transformers+bitsandbytes. Not required for the core Task 3.1 deliverable
(fp16 vs bitsandbytes int4 already satisfies "at least one real
technique"), but useful if you want a third row in the trade-off table to
back up the GGUF discussion in the write-up.

This one is CPU-friendly too (llama.cpp doesn't need a GPU), so it's the
one variant you could actually also try locally without Colab.

Setup:
    pip install llama-cpp-python huggingface_hub
    python -c "from huggingface_hub import hf_hub_download; \
        hf_hub_download('Qwen/Qwen2.5-1.5B-Instruct-GGUF', \
        'qwen2.5-1.5b-instruct-q4_k_m.gguf', local_dir='.')"

Usage:
    python gguf_benchmark.py qwen2.5-1.5b-instruct-q4_k_m.gguf
"""

from __future__ import annotations
import json
import sys
import time
from pathlib import Path

from llama_cpp import Llama

from prompts import PROMPTS, MAX_NEW_TOKENS  # reuse the exact same 5 prompts

OUTPUT_DIR = Path("results")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python gguf_benchmark.py <path-to-gguf-file>")
    model_path = sys.argv[1]

    llm = Llama(model_path=model_path, n_ctx=2048, verbose=False)

    results = []
    for i, prompt in enumerate(PROMPTS, start=1):
        start = time.perf_counter()
        out = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_NEW_TOKENS,
            temperature=0.0,
        )
        elapsed = time.perf_counter() - start

        text = out["choices"][0]["message"]["content"]
        new_tokens = out["usage"]["completion_tokens"]
        tps = round(new_tokens / elapsed, 2) if elapsed > 0 else None

        print(f"\n[{i}/{len(PROMPTS)}] {tps} tok/s\nQ: {prompt}\nA: {text}")
        results.append({
            "prompt": prompt, "output": text, "new_tokens": new_tokens,
            "elapsed_sec": round(elapsed, 3), "tokens_per_sec": tps,
        })

    avg_tps = sum(r["tokens_per_sec"] for r in results) / len(results)
    summary = {
        "mode": "gguf_q4_k_m",
        "model": model_path,
        # RSS at this point approximates resident memory for the quantized
        # weights + KV cache; report the .gguf file size too since that's
        # the more reproducible number across machines.
        "gguf_file_size_gb": round(Path(model_path).stat().st_size / 1e9, 3),
        "avg_tokens_per_sec": round(avg_tps, 2),
        "results": results,
    }

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "gguf_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAvg throughput: {avg_tps:.2f} tok/s")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
