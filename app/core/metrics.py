from prometheus_client import Counter, Histogram, Gauge

# Cache Hit Counters
CACHE_HITS_TOTAL = Counter(
    "llm_cache_hits_total",
    "Total number of cache hits",
    ["hit_type"]  # "exact" or "semantic"
)

CACHE_MISSES_TOTAL = Counter(
    "llm_cache_misses_total",
    "Total number of cache misses requiring LLM inference"
)

# Latency Histogram
REQUEST_LATENCY_SECONDS = Histogram(
    "llm_request_latency_seconds",
    "Latency of chat completion requests in seconds",
    ["cache_status"],  # "exact_hit", "semantic_hit", "miss"
    buckets=(0.005, 0.01, 0.015, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
)

# Token Counters & Financial Metrics
TOKENS_SAVED_TOTAL = Counter(
    "llm_tokens_saved_total",
    "Estimated tokens saved by returning cached responses"
)

ESTIMATED_COST_SAVED_USD = Counter(
    "llm_estimated_cost_saved_usd",
    "Estimated cloud LLM API cost saved in USD"
)

# System Health Gauge
REDIS_CONNECTED = Gauge(
    "llm_redis_connected",
    "Redis connection status (1 for connected, 0 for disconnected)"
)
