"""
Task 3.1 -- quantization benchmark: fp16/bf16 baseline vs bitsandbytes 4-bit
(NF4) on the same model and the same 5 fixed prompts.

Needs a CUDA GPU (bitsandbytes 4-bit is GPU-only). Built and tested for a
free-tier Colab T4 -- see README.md for the exact Colab setup.

Usage:
    python benchmark.py --mode fp16
    python benchmark.py --mode int4

Each run writes results/<mode>_results.json with peak memory, per-prompt
generation time/throughput, and the actual generated text, so the two runs
can be compared afterward with compare.py.
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Qwen2.5-1.5B-Instruct: small enough to run fp16 AND int4 comfortably on a
# free T4 (16GB VRAM), ungated (no HF access request needed, unlike Llama),
# and instruction-tuned so chat-style prompts work out of the box.
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = Path("results")
MAX_NEW_TOKENS = 200

# Same 5 prompts for both runs -- required by the assignment so quality is
# actually comparable. Mix of reasoning, creative, domain (ties to Sections
# 1/2), summarization, and translation, to surface quality differences a
# single prompt type might hide.
PROMPTS = [
    "Explain the difference between a list and a tuple in Python in two sentences.",
    "Write a short haiku about autumn rain.",
    "A customer says their food delivery is 40 minutes late. Write a brief, empathetic reply.",
    "Summarize in one paragraph why quantization reduces a neural network's memory footprint.",
    "Translate this sentence to French: 'The restaurant is closed on Mondays.'",
]


def load_model(mode: str):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    if mode == "fp16":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16, device_map="cuda",
        )
    elif mode == "int4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, quantization_config=bnb_config, device_map="cuda",
        )
    else:
        raise ValueError(f"unknown mode: {mode}")

    model.eval()
    return tokenizer, model


def run_prompt(tokenizer, model, prompt: str) -> dict:
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,        # <-- forces a dict with input_ids + attention_mask
    ).to(model.device)

    input_ids = inputs["input_ids"]

    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,                       # <-- unpack input_ids + attention_mask
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    new_tokens = output_ids.shape[1] - input_ids.shape[1]
    text = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

    return {
        "prompt": prompt,
        "output": text,
        "new_tokens": new_tokens,
        "elapsed_sec": round(elapsed, 3),
        "tokens_per_sec": round(new_tokens / elapsed, 2) if elapsed > 0 else None,
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fp16", "int4"], required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA GPU visible. This benchmark needs a GPU runtime "
            "(bitsandbytes 4-bit is GPU-only) -- see README.md for the Colab setup."
        )

    OUTPUT_DIR.mkdir(exist_ok=True)
    torch.cuda.reset_peak_memory_stats()

    print(f"Loading {MODEL_ID} in {args.mode} mode...")
    tokenizer, model = load_model(args.mode)
    load_memory_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"Loaded. Memory after load: {load_memory_gb:.2f} GB")

    results = []
    for i, prompt in enumerate(PROMPTS, start=1):
        r = run_prompt(tokenizer, model, prompt)
        print(f"\n[{i}/{len(PROMPTS)}] {r['tokens_per_sec']} tok/s")
        print(f"Q: {prompt}")
        print(f"A: {r['output']}")
        results.append(r)

    peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9
    avg_tps = sum(r["tokens_per_sec"] for r in results) / len(results)

    summary = {
        "mode": args.mode,
        "model": MODEL_ID,
        "load_memory_gb": round(load_memory_gb, 3),
        "peak_memory_gb": round(peak_memory_gb, 3),
        "avg_tokens_per_sec": round(avg_tps, 2),
        "results": results,
    }

    out_path = OUTPUT_DIR / f"{args.mode}_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== {args.mode} summary ===")
    print(f"Peak memory: {peak_memory_gb:.2f} GB")
    print(f"Avg throughput: {avg_tps:.2f} tok/s")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
