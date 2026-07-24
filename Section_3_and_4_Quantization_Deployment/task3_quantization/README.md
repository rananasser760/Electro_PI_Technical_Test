````md
# 🔬 Quantization Benchmark  
## FP16 vs 4-bit NF4 (bitsandbytes) vs GGUF (llama.cpp)

**Task 3.1 — Model Quantization Evaluation**

> A practical benchmark comparing three deployment strategies for the same LLM:
> full precision inference (FP16), GPU-oriented 4-bit quantization (NF4), and CPU-friendly GGUF quantization.

All results below are from **real executions**:

- **FP16 + INT4 NF4** → Google Colab NVIDIA Tesla T4 (16 GB VRAM)
- **GGUF Q4_K_M** → Local CPU inference using llama.cpp

---

# 📌 Overview

This benchmark evaluates how different quantization strategies affect:

- **Memory footprint**
- **Inference throughput**
- **Generation quality**
- **Deployment suitability**

The comparison uses the same:

- Model
- 5 fixed evaluation prompts
- Greedy decoding strategy
- Deterministic generation settings

The goal is to understand the practical trade-off between:

> Memory efficiency ↔ Inference speed ↔ Output quality

---

# 🧠 Model

## `Qwen/Qwen2.5-1.5B-Instruct`

Selected because it:

- Fits easily on free-tier GPU hardware
- Requires no gated Hugging Face access
- Is instruction-tuned for chat evaluation
- Is lightweight enough to benchmark multiple quantization approaches

---

# ⚙️ Experimental Setup

| Component | Configuration |
|---|---|
| Model | Qwen2.5-1.5B-Instruct |
| FP16 / NF4 Hardware | Google Colab NVIDIA Tesla T4 (16GB VRAM) |
| GGUF Runtime | llama.cpp |
| GGUF Quantization | Q4_K_M |
| Decoding | Greedy decoding (`temperature=0`) |
| Evaluation Prompts | 5 identical prompts |
| Runs | One clean run per configuration |

---

# 🚀 Running the Benchmark

## 1. FP16 / INT4 NF4 (Transformers + bitsandbytes)

Install dependencies:

```bash
pip install transformers accelerate bitsandbytes sentencepiece
````

Run FP16:

```bash
python benchmark.py --mode fp16
```

Restart runtime to clear GPU memory.

Run INT4:

```bash
python benchmark.py --mode int4
```

Generate comparison report:

```bash
python compare.py
```

---

## 2. GGUF (llama.cpp)

Install:

```bash
pip install llama-cpp-python huggingface_hub
```

Download model:

```python
from huggingface_hub import hf_hub_download

hf_hub_download(
    "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
    "qwen2.5-1.5b-instruct-q4_k_m.gguf",
    local_dir="models"
)
```

Run benchmark:

```bash
python gguf_benchmark.py models/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

---

# 🧪 Evaluation Prompts

The same prompts were used for all configurations.

The prompts cover:

* Programming explanation
* Creative generation
* Customer support
* Technical summarization
* Translation

### Prompts

1. Explain the difference between a list and a tuple in Python in two sentences.

2. Write a short haiku about autumn rain.

3. A customer says their food delivery is 40 minutes late. Write a brief, empathetic reply.

4. Summarize in one paragraph why quantization reduces a neural network's memory footprint.

5. Translate:

```
The restaurant is closed on Mondays.
```

---

# 📊 Benchmark Results

## Overall Comparison

| Metric             |            FP16 |                    INT4 NF4 |     GGUF Q4_K_M |
| ------------------ | --------------: | --------------------------: | --------------: |
| Runtime            |    Transformers | Transformers + bitsandbytes |       llama.cpp |
| Hardware           |          T4 GPU |                      T4 GPU |             CPU |
| Peak GPU Memory    |    **3.107 GB** |       **1.222 GB (-60.7%)** |             N/A |
| Model Size         |         ~3.1 GB |                 ~1.0–1.1 GB |    **1.117 GB** |
| Average Throughput | **22.42 tok/s** |             **11.29 tok/s** | **11.42 tok/s** |

---

# 🔍 Per Prompt Throughput

| Prompt                 | FP16 tok/s | INT4 NF4 tok/s | GGUF tok/s |
| ---------------------- | ---------: | -------------: | ---------: |
| List vs Tuple          |      14.16 |          11.20 |      11.31 |
| Autumn Rain Haiku      |      24.56 |          12.30 |       9.56 |
| Customer Support Reply |      26.03 |          10.34 |      12.55 |
| Quantization Summary   |      22.83 |          11.43 |      13.49 |
| French Translation     |      24.50 |          11.19 |      10.21 |

---

# 📈 Key Findings

## 1. Memory Efficiency

INT4 NF4 reduced GPU memory usage from:

```
3.107 GB → 1.222 GB
```

Result:

> **60.7% reduction in peak GPU memory**

This confirms the main advantage of quantization:
reducing hardware requirements while maintaining usable quality.

---

## 2. Inference Speed

Measured throughput:

```
FP16      : 22.42 tok/s
INT4 NF4  : 11.29 tok/s
GGUF      : 11.42 tok/s
```

Observations:

* FP16 achieved the highest speed.
* NF4 reduced memory but introduced runtime dequantization overhead.
* GGUF achieved similar throughput without requiring a GPU.

---

# 🧠 Qualitative Evaluation

## Programming Explanation

All three versions generated correct and fluent explanations.

No meaningful quality difference was observed.

---

## Creative Writing

All models generated coherent haikus.

Differences were only stylistic.

---

## Customer Support Reply

All outputs were polite and usable.

FP16 was slightly smoother in wording, while quantized models remained practical.

---

## Translation Example

Prompt:

```
The restaurant is closed on Mondays.
```

### FP16

```
Le restaurant est fermé le lundi.
```

### INT4 NF4

```
La maison de restauration est fermée les lundis.
```

### GGUF

```
Le restaurant est fermé les lundis.
```

FP16 and GGUF preserved the natural French meaning better.

This demonstrates that precision-sensitive tasks such as translation can expose small quantization differences.

---

# 🏭 Production Recommendations

| Method           | Best Use Case                   | Advantages                              | Limitations                     |
| ---------------- | ------------------------------- | --------------------------------------- | ------------------------------- |
| bitsandbytes NF4 | Research & rapid experiments    | Simple integration, no calibration step | Runtime dequantization overhead |
| GPTQ             | GPU production inference        | Fast inference, mature ecosystem        | Requires calibration            |
| AWQ              | High-quality low-bit deployment | Better quality preservation             | More complex workflow           |
| GGUF             | CPU / Edge deployment           | Portable model file, no CUDA dependency | Lower GPU throughput            |

---

# 🧩 Production Analysis

## bitsandbytes (NF4)

Recommended for:

* Research
* Rapid prototyping
* Fine-tuning experiments
* Model evaluation

Advantages:

* Directly works with Transformers checkpoints
* Simple integration
* No offline quantization process

Trade-off observed in this benchmark:

```
Memory ↓ 60.7%

Throughput ↓ ~50%
```

The reason is runtime dequantization during inference.

---

## GPTQ / AWQ

Better suited for production GPU deployment.

Advantages:

* Offline quantization
* Optimized inference kernels
* Lower serving cost

AWQ is preferred when:

* Very low-bit quantization is required
* Quality preservation is critical

GPTQ remains widely adopted because of:

* Mature ecosystem
* Broad tooling support

---

## GGUF (llama.cpp)

Best suited for:

* CPU inference
* Edge devices
* Local applications
* Lightweight deployments

This benchmark showed:

```
GGUF CPU:
11.42 tok/s

NF4 GPU:
11.29 tok/s
```

For small models, GGUF can provide competitive inference without requiring:

* CUDA
* GPU infrastructure
* Transformers runtime

Its main advantage is deployment simplicity.

---

# ✅ Final Conclusion

| Goal                   | Recommended Approach |
| ---------------------- | -------------------- |
| Fast experimentation   | bitsandbytes NF4     |
| GPU production API     | GPTQ / AWQ           |
| CPU or edge deployment | GGUF                 |
| Maximum quality        | FP16                 |

The benchmark confirms that quantization is mainly:

> **A memory optimization and deployment strategy**

rather than a guaranteed speed improvement.

The optimal quantization method depends on:

* Target hardware
* Latency requirements
* Deployment environment
* Quality constraints

```
```
