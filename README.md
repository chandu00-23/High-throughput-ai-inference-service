# High-Throughput AI Inference Microservice with Redis Semantic Cache

> Production-grade FastAPI microservice serving LLM completion endpoints backed by a **Dual-Layer Redis Semantic Vector Cache**. Delivers sub-15ms response times on cached queries, cutting LLM inference costs and latency by up to 95%.

---

## 🚀 Key Architectural Features

- **Dual-Layer Caching Architecture**:
  - **Layer 1 (Exact Hash Match)**: SHA-256 string hash lookup in Redis for identical queries (< 2ms response time).
  - **Layer 2 (Semantic Vector Match)**: Non-blocking CPU vector embedding + RediSearch HNSW Index lookup with TAG filtering for semantically equivalent prompts ($>0.95$ similarity, $< 15$ms response time).
- **Context-Aware Composite Hashing (Prevents Semantic Drift)**:
  - Generates composite keys: `SHA256(model + system_prompt + temperature + user_prompt)`.
  - Filters vectors in RediSearch by `@model` and `@system_hash` TAGs, preventing temperature and prompt context pollution across requests.
- **Non-Blocking Thread Offload**:
  - Offloads CPU matrix operations for sentence embedding (`SentenceTransformers`/`FastEmbed`) to worker threads via `asyncio.to_thread()`, keeping Uvicorn's event loop completely unblocked.
- **Asynchronous Background Cache Writes**:
  - Sends streaming SSE `[DONE]` tokens immediately to clients, offloading response compilation and Redis writes to background `asyncio.Task` calls.
- **Redis Memory Protection**:
  - Guarded against payload explosion with 100 KB max response limits and configured `allkeys-lru` memory eviction policy.
- **OpenAI Standard Protocol Compliance**:
  - Drop-in replacement for any client codebase utilizing standard `/v1/chat/completions` API specifications (supports both streaming SSE and non-streaming responses).
- **Observability & Visual Dashboard**:
  - Built-in Prometheus `/metrics` exporter (Cache Hit Ratio, exact/semantic counters, TTFT histograms, estimated USD cost savings).
  - Included Dark-Mode Admin Dashboard UI (`/dashboard`) with real-time prompt playground and side-by-side latency benchmarks.

---

## 🏗️ Architecture Diagram

```text
Incoming Client Request (FastAPI /v1/chat/completions)
                       │
                       ▼
         ┌───────────────────────────┐
         │ Composite Key Generation  │ (Model + System Prompt + Temp + User Prompt)
         └─────────────┬─────────────┘
                       │
                       ▼
         ┌───────────────────────────┐
         │ 1. Exact Redis Hash Check │ (< 2ms)
         └─────────────┬─────────────┘
                       │
              ┌────────┴────────┐
              │                 │
           [ HIT ]           [ MISS ]
              │                 │
              ▼                 ▼
       Return Payload    ┌───────────────────────────┐
                         │ 2. Async Vector Embedding │ (Thread Pool Offload)
                         └──────────────┬────────────┘
                                        │
                                        ▼
                         ┌───────────────────────────┐
                         │ 3. RediSearch Vector      │ (TAG Filtered, distance <= 0.05)
                         │    Similarity Search      │
                         └──────────────┬────────────┘
                                        │
                               ┌────────┴────────┐
                               │                 │
                           [ HIT (>0.95) ]    [ MISS ]
                               │                 │
                               ▼                 ▼
                        Return Cached    ┌───────────────────┐
                        Stream / Response│ 4. LLM Engine     │ (vLLM / Ollama / OpenAI / Mock)
                                         └─────────┬─────────┘
                                                   │
                                                   ▼
                                         ┌───────────────────┐
                                         │ 5. Async Background│ (Non-blocking SSE tail write)
                                         │    Cache Write    │
                                         └───────────────────┘
```

---

## ⚡ Quick Start

### 1. Run via Docker Compose (Recommended)

```bash
docker-compose up --build
```

Access services:
- **FastAPI API Base**: `http://localhost:8000`
- **Interactive Dashboard**: [http://localhost:8000/dashboard](http://localhost:8000/dashboard)
- **Prometheus Metrics**: [http://localhost:8000/metrics](http://localhost:8000/metrics)
- **Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

### 2. Local Setup (Without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Start local Redis Stack container
docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest

# Launch FastAPI app
uvicorn app.main:app --reload --port 8000
```

---

## 🔌 OpenAI Drop-In Client Integration Example

You can plug this microservice directly into any Python / LangChain app by changing `base_url`:

```python
from openai import OpenAI

# Connect to your local inference gateway
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="llama-3-8b-instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in one sentence."}
    ],
    temperature=0.7
)

print(response.choices[0].message.content)
```

---

## 📊 Latency Benchmarking

Run the automated latency benchmark script to test side-by-side performance:

```bash
python tests/test_latency.py
```

### Benchmark Sample Output

| Query Type | Cache Status | Latency (ms) | Speedup |
| :--- | :--- | :--- | :--- |
| **1. Initial Query (LLM Miss)** | `CACHE MISS` | `842.10 ms` | 1.0x (Base) |
| **2. Identical Query (Exact Hit)** | `CACHE HIT (EXACT)` | `2.15 ms` | **391.6x faster** |
| **3. Rephrased Query (Semantic Hit)** | `CACHE HIT (SEMANTIC)` | `12.40 ms` | **67.9x faster** |

---

## 🧪 Running Tests

```bash
# Run unit & integration tests
pytest

# Run Locust high-concurrency load test
locust -f tests/locustfile.py --host http://localhost:8000
```
