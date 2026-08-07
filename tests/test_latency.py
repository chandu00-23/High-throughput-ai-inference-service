import time
import httpx
import asyncio

BASE_URL = "http://localhost:8000"

async def run_latency_benchmark():
    print("=" * 70)
    print("  HIGH-THROUGHPUT AI INFERENCE & SEMANTIC CACHE BENCHMARK")
    print("=" * 70)

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Check health
        try:
            res = await client.get(f"{BASE_URL}/health")
            if res.status_code != 200:
                print(f"Service at {BASE_URL} is not ready. Status: {res.status_code}")
                return
        except Exception as e:
            print(f"Error connecting to service at {BASE_URL}: {e}")
            print("Make sure the FastAPI app is running (e.g. uvicorn app.main:app --port 8000)")
            return

        print("\nSending Test Prompts...\n")

        prompt_original = "Explain the theory of general relativity in two sentences."
        prompt_rephrased = "Could you explain the theory of general relativity in a couple of sentences?"

        payload_base = {
            "model": "llama-3-8b-instruct",
            "messages": [
                {"role": "system", "content": "You are a professional physics tutor."},
                {"role": "user", "content": prompt_original}
            ],
            "temperature": 0.7,
            "stream": False
        }

        # ---------------------------------------------------------------------
        # Request 1: Initial Prompt (Cache Miss -> LLM Engine)
        # ---------------------------------------------------------------------
        t0 = time.perf_counter()
        resp1 = await client.post(f"{BASE_URL}/v1/chat/completions", json=payload_base)
        lat1 = (time.perf_counter() - t0) * 1000.0
        data1 = resp1.json()
        status1 = "CACHE MISS" if not data1.get("cached") else "CACHE HIT"

        # ---------------------------------------------------------------------
        # Request 2: Identical Prompt (Exact Cache Hit)
        # ---------------------------------------------------------------------
        t0 = time.perf_counter()
        resp2 = await client.post(f"{BASE_URL}/v1/chat/completions", json=payload_base)
        lat2 = (time.perf_counter() - t0) * 1000.0
        data2 = resp2.json()
        cache_type2 = data2.get("cache_type", "exact").upper()
        status2 = f"CACHE HIT ({cache_type2})" if data2.get("cached") else "CACHE MISS"

        # ---------------------------------------------------------------------
        # Request 3: Rephrased Prompt (Semantic Vector Cache Hit)
        # ---------------------------------------------------------------------
        payload_rephrased = dict(payload_base)
        payload_rephrased["messages"] = [
            {"role": "system", "content": "You are a professional physics tutor."},
            {"role": "user", "content": prompt_rephrased}
        ]
        t0 = time.perf_counter()
        resp3 = await client.post(f"{BASE_URL}/v1/chat/completions", json=payload_rephrased)
        lat3 = (time.perf_counter() - t0) * 1000.0
        data3 = resp3.json()
        cache_type3 = data3.get("cache_type", "semantic").upper()
        status3 = f"CACHE HIT ({cache_type3})" if data3.get("cached") else "CACHE MISS"

        # ---------------------------------------------------------------------
        # Print Performance Summary Table
        # ---------------------------------------------------------------------
        print(f"{'Query Type':<32} | {'Cache Status':<22} | {'Latency (ms)':<12} | {'Speedup':<10}")
        print("-" * 82)
        print(f"{'1. Initial Prompt (LLM Miss)':<32} | {status1:<22} | {lat1:>10.2f} ms | {'1.0x (Base)':<10}")
        speedup2 = lat1 / lat2 if lat2 > 0 else 1.0
        print(f"{'2. Identical Query (Exact Hit)':<32} | {status2:<22} | {lat2:>10.2f} ms | {f'{speedup2:.1f}x faster':<10}")
        speedup3 = lat1 / lat3 if lat3 > 0 else 1.0
        print(f"{'3. Rephrased Query (Semantic Hit)':<32} | {status3:<22} | {lat3:>10.2f} ms | {f'{speedup3:.1f}x faster':<10}")
        print("-" * 82)
        print("\nBenchmark completed successfully!\n")

if __name__ == "__main__":
    asyncio.run(run_latency_benchmark())
