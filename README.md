# Electro Pi — AI Engineer Technical Test

**Candidate:** Rana Nasser
**Role:** AI Engineer — Mid Level (3+ yrs)
**Location:** Maadi, Cairo
**Submission type:** Practical / hands-on assessment (4 sections + 1 bonus task)

This repository contains my complete solution to the Electro Pi AI Engineer take-home assessment. Each section maps to a core skill from the job description — real-time voice agents, retrieval-augmented generation, model quantization, and production deployment — and is implemented as a small, working, self-contained project rather than a theoretical exercise.

---

## Repository Structure

```
.
├── README.md                      # This file — overview, setup, and results summary
├── NOTES.md                       # Half-page write-up per section (design trade-offs, reasoning)
│
├── section1_livekit_agent/        # Real-time voice AI agent with tool calling
│   ├── agent.py
│   ├── requirements.txt
│   ├── transcript_example.log
│   └── README.md
│
├── section2_langchain_rag/        # Retrieval-augmented generation pipeline
│   ├── docs/                      # Source documents used for retrieval
│   ├── rag.py
│   ├── requirements.txt
│   ├── example_qa.md              # 3 example questions + actual pipeline answers
│   └── README.md
│
├── section3_quantization/         # Full-precision vs. quantized model comparison
│   ├── quantize_and_benchmark.py
│   ├── requirements.txt
│   ├── results/
│   │   └── benchmark_table.md
│   └── README.md
│
└── section4_deployment/           # Model served behind a REST API, containerized
    ├── app/
    │   └── main.py
    ├── Dockerfile
    ├── requirements.txt
    ├── load_test.py
    ├── results/
    │   └── latency_report.md
    └── README.md
```

Each subfolder has its own README with setup and run instructions specific to that section. This top-level README covers what's needed to get oriented and run everything end-to-end.

---

## Quick Start

```bash
git clone <repo-url>
cd <repo-name>

# Each section is isolated — set up and run independently
cd section1_livekit_agent && pip install -r requirements.txt && python agent.py
cd ../section2_langchain_rag && pip install -r requirements.txt && python rag.py
cd ../section3_quantization && pip install -r requirements.txt && python quantize_and_benchmark.py
cd ../section4_deployment && docker build -t ai-engineer-api . && docker run -p 8000:8000 ai-engineer-api
```

Every section is designed to run within ~10 minutes of cloning, per the submission checklist. Where a step depends on external API keys or GPU access, that's called out explicitly below and in the relevant section README, with a fallback/mock path so the reviewer can still see the logic run end-to-end.

---

## Section Overview

### 1 — LiveKit Agents (Real-time Voice AI)
A minimal `AgentSession` pipeline (STT → LLM → TTS) built on `livekit-agents`, with an `Agent` subclass carrying a support-assistant persona and a `@function_tool`-decorated `get_order_status(order_id)` method the LLM can invoke mid-conversation. Includes a transcript demonstrating the tool call firing during a simulated session, plus a write-up on barge-in handling and adding a second tool safely.

**Providers used:** [fill in — e.g. "OpenAI Realtime for LLM, Deepgram STT, ElevenLabs TTS" or "mocked text I/O, real tool-calling logic"]

### 2 — LangChain / RAG
A retrieval pipeline over [fill in — e.g. "5 short markdown docs on X"], chunked and embedded into a [FAISS / Chroma] vector store, wired into a LangChain chain that answers questions with citations back to source chunks and explicitly declines to answer when no relevant context is found.

**Example questions and actual answers:** see `section2_langchain_rag/example_qa.md`

### 3 — Quantization
[Model name, e.g. Qwen2.5-1.5B] run once at fp16/bf16 and once quantized via [bitsandbytes 4-bit NF4 / GGUF], with memory footprint, tokens/sec, and qualitative output compared across the same 5 fixed prompts.

**Full results table:** see `section3_quantization/results/benchmark_table.md`

### 4 — Model Deployment
The model from Section 3 served behind a [FastAPI / vLLM] endpoint, containerized with a working `Dockerfile`, supporting streaming responses, with a time-to-first-token and total-latency report for 10 concurrent requests.

**Full latency report:** see `section4_deployment/results/latency_report.md`

---

## Write-ups

The half-page reasoning write-up required for each section (extending barge-in handling, chunking/retrieval trade-offs, GPTQ/AWQ/GGUF vs. bitsandbytes, and scaling to 50 concurrent users) lives in [`NOTES.md`](./NOTES.md), one section per heading.

---

## Assumptions, Shortcuts, and Known Limitations

- [fill in — e.g. "STT/TTS in Section 1 are mocked with text I/O; tool-calling logic itself is real and unmocked."]
- [fill in — e.g. "Embeddings in Section 2 use X instead of Y because Z."]
- [fill in — e.g. "Section 3 benchmarks were run on Google Colab T4 (free tier); numbers will vary on other hardware."]
- [fill in — e.g. "Section 4 load test uses N requests rather than a full production-scale run, given the take-home time budget."]

Being explicit about trade-offs here is intentional — the assessment scores honesty about limitations, not the absence of them.

---

## Tech Stack

`livekit-agents` · `LangChain` / `LangGraph` · `FAISS` / `Chroma` · `transformers` · `bitsandbytes` / `llama.cpp` · `FastAPI` / `vLLM` · `Docker`

---

## Contact

Rana — email: ```rananasser760@gmail.com```
