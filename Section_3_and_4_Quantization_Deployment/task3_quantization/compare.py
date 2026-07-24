"""
Reads results/fp16_results.json and results/int4_results.json (produced by
benchmark.py) and generates the trade-off table + side-by-side qualitative
comparison required by the assignment.

Usage (after running benchmark.py for both modes):
    python compare.py
"""

from __future__ import annotations
import json
from pathlib import Path

RESULTS_DIR = Path("results")


def load(mode: str) -> dict:
    path = RESULTS_DIR / f"{mode}_results.json"
    if not path.exists():
        raise SystemExit(f"Missing {path} -- run `python benchmark.py --mode {mode}` first.")
    with open(path) as f:
        return json.load(f)


def main() -> None:
    fp16 = load("fp16")
    int4 = load("int4")

    mem_ratio = fp16["peak_memory_gb"] / int4["peak_memory_gb"]
    speed_ratio = int4["avg_tokens_per_sec"] / fp16["avg_tokens_per_sec"]

    lines = [
        f"# Precision vs size vs speed vs quality -- {fp16['model']}",
        "",
        "| Metric | fp16 | int4 (NF4, bitsandbytes) |",
        "|---|---|---|",
        f"| Peak GPU memory | {fp16['peak_memory_gb']} GB | {int4['peak_memory_gb']} GB "
        f"({mem_ratio:.2f}x smaller) |",
        f"| Avg throughput | {fp16['avg_tokens_per_sec']} tok/s | {int4['avg_tokens_per_sec']} tok/s "
        f"({speed_ratio:.2f}x) |",
        "",
        "## Side-by-side outputs (same 5 prompts, greedy decoding)",
        "",
    ]

    for p16, p4 in zip(fp16["results"], int4["results"]):
        lines.append(f"**Prompt:** {p16['prompt']}")
        lines.append("")
        lines.append(f"- **fp16** ({p16['tokens_per_sec']} tok/s): {p16['output']}")
        lines.append(f"- **int4** ({p4['tokens_per_sec']} tok/s): {p4['output']}")
        lines.append("")

    report = "\n".join(lines)
    print(report)

    out_path = Path("comparison_report.md")
    out_path.write_text(report)
    print(f"\n[written to {out_path}]")


if __name__ == "__main__":
    main()
