"""
Task 4.1 -- FastAPI service wrapping the GGUF model from Section 3.

Design choice: FastAPI + llama-cpp-python instead of vLLM/TGI.
--------------------------------------------------------------
vLLM and TGI are built around batched GPU inference (PagedAttention,
continuous batching) -- they don't support GGUF and assume a CUDA GPU is
present. This model is served as a CPU GGUF build (see Section 3's
write-up: GGUF is the right call specifically because there's no
dedicated GPU box here), so a GPU-batching inference server doesn't
apply. FastAPI + llama-cpp-python is the lightweight match for that
target: pure Python + a C++ inference lib, no CUDA dependency, easy to
containerize, and gives full control over the request-queueing behavior
that a single CPU model instance actually needs (see below).

Concurrency model
------------------
A single llama.cpp context is not safe to call from multiple threads at
once (it holds one KV cache). Rather than pretend otherwise, this app is
explicit about it: all generation calls run through a single-worker
ThreadPoolExecutor, so concurrent requests queue up and are served one at
a time -- FIFO, no request is dropped, but latency scales with queue
depth. That queueing behavior is exactly what the Task 4.1 load test
below measures, and it's exactly what the write-up's "what changes for
50 concurrent users" section addresses (batching, multiple workers,
autoscaling).
"""

from __future__ import annotations
import asyncio
import os
import queue
import time
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

MODEL_PATH = os.environ.get("MODEL_PATH", "models/qwen2.5-1.5b-instruct-q4_k_m.gguf")
N_THREADS = int(os.environ.get("N_THREADS", os.cpu_count() or 4))
N_CTX = int(os.environ.get("N_CTX", 2048))

app = FastAPI(title="Qwen2.5-1.5B GGUF Inference API")

llm = None
# Exactly one worker: serializes access to the single llama.cpp context.
# A second, larger pool handles I/O-bound waits (queue.get) so the event
# loop is never blocked even though generation itself is serialized.
generation_executor = ThreadPoolExecutor(max_workers=1)
io_executor = ThreadPoolExecutor(max_workers=8)


@app.on_event("startup")
def load_model() -> None:
    global llm
    from llama_cpp import Llama  # imported here so --help/tests don't require the model file present
    llm = Llama(model_path=MODEL_PATH, n_ctx=N_CTX, n_threads=N_THREADS, verbose=False)


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 200
    temperature: float = 0.0


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": llm is not None}


def _blocking_generate(prompt: str, max_tokens: int, temperature: float) -> dict:
    start = time.perf_counter()
    out = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    elapsed = time.perf_counter() - start
    usage = out["usage"]
    return {
        "output": out["choices"][0]["message"]["content"],
        "completion_tokens": usage["completion_tokens"],
        "elapsed_sec": round(elapsed, 3),
        "tokens_per_sec": round(usage["completion_tokens"] / elapsed, 2) if elapsed > 0 else None,
    }


@app.post("/generate")
async def generate(req: GenerateRequest) -> dict:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        generation_executor, _blocking_generate, req.prompt, req.max_tokens, req.temperature
    )


def _run_stream_producer(prompt: str, max_tokens: int, temperature: float, q: "queue.Queue") -> None:
    """Runs on generation_executor (the single serialized worker). Pushes
    each token onto the queue as llama.cpp produces it, then a sentinel."""
    SENTINEL = None
    try:
        stream = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            token = chunk["choices"][0]["delta"].get("content", "")
            if token:
                q.put(token)
    finally:
        q.put(SENTINEL)


@app.post("/generate/stream")
async def generate_stream(req: GenerateRequest) -> StreamingResponse:
    loop = asyncio.get_event_loop()
    q: "queue.Queue" = queue.Queue()

    # Fire-and-forget onto the single generation worker -- this queues
    # behind any in-flight generation, same as /generate, but doesn't
    # block the event loop while it waits its turn.
    generation_executor.submit(_run_stream_producer, req.prompt, req.max_tokens, req.temperature, q)

    async def event_stream():
        while True:
            token = await loop.run_in_executor(io_executor, q.get)
            if token is None:  # sentinel
                break
            yield token

    return StreamingResponse(event_stream(), media_type="text/plain")