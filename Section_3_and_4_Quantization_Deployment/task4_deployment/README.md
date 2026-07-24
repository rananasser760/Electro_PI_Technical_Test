# 🚀 Qwen2.5-1.5B GGUF Deployment with FastAPI

<p align="center">

**Production-style REST API deployment for a quantized LLM using llama.cpp, FastAPI, and Docker**

</p>

---

## 📌 Overview

This project deploys a quantized LLM in a production-oriented environment:

- **Model:** Qwen2.5-1.5B-Instruct
- **Quantization:** GGUF Q4_K_M
- **Inference engine:** llama.cpp

Served through a lightweight **FastAPI REST API** with:
- ✅ Text generation endpoint
- ✅ Streaming token generation
- ✅ Dockerized deployment
- ✅ Concurrency handling
- ✅ Load testing support
- ✅ Production scaling considerations

This follows the approach chosen in **Section 3 (Quantization Benchmark)**, where GGUF was selected for its low memory footprint, CPU-friendly inference, portability, and simple deployment without CUDA.

---

## 🏗️ System Architecture

```
User / Client
     |
HTTP REST Request
     |
FastAPI Server
     |
llama-cpp-python Runtime
     |
GGUF Quantized Model
     |
qwen2.5-1.5b-instruct-q4_k_m.gguf
```

**Deployment flow:** client sends a request → FastAPI validates it → passed to the inference layer → llama.cpp loads/executes the GGUF model → tokens returned either as a complete response or streamed.

---

## 🎯 Why FastAPI + llama.cpp (not vLLM / TGI)?

vLLM and TGI are optimized for GPU-based inference, large-scale serving, continuous batching, and CUDA acceleration. This project instead targets **GGUF**, designed for CPU inference, edge deployment, local apps, and lightweight servers — so **GGUF + llama.cpp + FastAPI** is the better match here.

---

## 🧠 Deployment Stack

| Component | Technology |
|---|---|
| Model | Qwen2.5-1.5B-Instruct |
| Quantization | GGUF Q4_K_M |
| Runtime | llama.cpp |
| Python Binding | llama-cpp-python |
| API Framework | FastAPI |
| Server | Uvicorn |
| Containerization | Docker |
| Testing | Pytest + Custom Load Tester |

---

## 📂 Project Structure

```
├── task4_deployment/
│   ├── app/
│   │   ├── __init__.py
│   │   └── main.py
│   ├── tests/
│   │   ├── fake_llama_cpp.py
│   │   └── test_app.py
│   ├── load_test.py
│   ├── Dockerfile
│   ├── .dockerignore
│   ├── requirements.txt
│   └── README.md
```

---

## ⚙️ Requirements

- Ubuntu 22.04
- Python 3.10+
- Docker (optional)
- CPU with multiple threads

---

## 🐍 Local Deployment

**1. Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```
Main deps: `fastapi`, `uvicorn`, `llama-cpp-python`, `pydantic`, `requests`

**3. Place the model**
```
models/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

**4. Configure environment variables**

Linux:
```bash
export MODEL_PATH=models/qwen2.5-1.5b-instruct-q4_k_m.gguf
export N_THREADS=$(nproc)
```
Windows PowerShell:
```bash
$env:MODEL_PATH="models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
```

**5. Start the server**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Runs at `http://localhost:8000`

---

## 🐳 Docker Deployment

Docker is the recommended way to run the API reproducibly. Application code and dependencies live in the image; the GGUF model is **mounted as an external volume** rather than baked in — this keeps images small, builds fast, and lets the model be updated without a rebuild.

**Build:**
```bash
docker build -t qwen-gguf-api .
```

**Run:**
```bash
docker run --rm -p 8000:8000 -v $(pwd)/models:/app/models qwen-gguf-api
```

The host's `models/qwen2.5-1.5b-instruct-q4_k_m.gguf` (~1.1 GB) maps to `/app/models/` inside the container.

---

## 🔍 Health Check

```bash
curl http://localhost:8000/health
```
```json
{ "status": "ok", "model_loaded": true }
```

---

## 🔌 API Endpoints

### 1. `POST /generate` — Standard generation
Returns the complete response after generation finishes.

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Explain quantization in one sentence.","max_tokens":100}'
```
```json
{
    "output": "Quantization reduces model memory by representing weights using fewer bits.",
    "completion_tokens": 15
}
```

### 2. `POST /generate/stream` — Streaming generation
Streams tokens incrementally as they're generated (use `-N` with curl to disable buffering).

```bash
curl -N -X POST http://localhost:8000/generate/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Write a short haiku about autumn rain.","max_tokens":100}'
```

Streaming reduces perceived latency, gives immediate feedback, and matches modern chatbot-style UX.

---

## 🧪 Testing

Automated tests cover API routing, generation, streaming, and concurrent requests — using a mock `llama_cpp.Llama` so no GPU, large model files, or llama.cpp compilation are needed.

```bash
python3 tests/test_app.py
```
Expected: `test_health`, `test_generate`, `test_generate_stream`, and `test_concurrent_requests_are_serialized_not_dropped` all `OK`.

---

## 📊 Load Testing

```bash
pip install requests
python3 load_test.py --url http://localhost:8000 --concurrency 10
```

Measures: total execution time, per-request latency, average latency, P50 latency, streaming (time-to-first-token) latency, and throughput.

---

## 📈 Scaling to 50 Concurrent Users

The current setup (FastAPI → single llama.cpp instance → CPU inference) works for demos and small workloads, but with 50 simultaneous users, requests queue up and latency grows linearly with queue depth.

**Production scaling strategy:**

1. **Horizontal scaling** — run multiple API instances behind a load balancer (Kubernetes, Docker Compose, or Nginx), each with its own llama.cpp process → independent queues, higher throughput, better availability.
2. **GPU + continuous batching** — for sustained high traffic, move to a GPU server running GPTQ/AWQ-quantized models via vLLM/TGI with continuous batching, for much higher throughput and lower latency.
3. **External queue management** — offload queuing to Redis Queue, Celery, or RabbitMQ (Client → API Gateway → Request Queue → Workers → LLM Instances) for backpressure handling and prioritization.
4. **Caching** — a similarity/embedding/semantic cache can return answers to repeated questions without re-generating.
5. **Autoscaling** — scale instance count based on queue length, latency, and traffic patterns (e.g. Kubernetes HPA, cloud autoscaling).

---

## 🚀 Final Recommendation

- **Current GGUF CPU setup** (FastAPI + llama.cpp + Docker) — lightweight, practical for local apps, edge devices, and small production workloads.
- **Large-scale production** — GPU infrastructure + AWQ/GPTQ quantization + vLLM/TGI + continuous batching.

| Requirement | Recommended Solution |
|---|---|
| Local deployment | GGUF + llama.cpp |
| CPU inference | llama.cpp |
| Lightweight API | FastAPI |
| Container deployment | Docker |
| Small concurrent users | Multiple API workers |
| 50+ concurrent users | Horizontal scaling |
| Large production traffic | GPU + vLLM/TGI |

**Key takeaway:** GGUF balances model size, memory usage, and deployment simplicity — one of the most practical formats for CPU/edge environments. For large-scale serving, the architecture should evolve toward GPU-based inference with optimized serving engines.