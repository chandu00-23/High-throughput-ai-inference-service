import time
import json
import uuid
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from prometheus_client import make_asgi_app

from app.core.config import settings
from app.core.metrics import (
    CACHE_HITS_TOTAL,
    CACHE_MISSES_TOTAL,
    REQUEST_LATENCY_SECONDS,
    TOKENS_SAVED_TOTAL,
    ESTIMATED_COST_SAVED_USD,
    REDIS_CONNECTED,
)
from app.schemas.openai import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatMessage,
    UsageInfo,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
)
from app.services.cache_service import CacheService
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import get_llm_provider

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("mlops-inference")

# Global Service Singletons
cache_service = CacheService()
embedding_service: Optional[EmbeddingService] = None  # Initialized during lifespan or on demand

def get_embedding_service() -> EmbeddingService:
    global embedding_service
    if embedding_service is None:
        embedding_service = EmbeddingService(settings.EMBEDDING_MODEL_NAME)
    return embedding_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager for setup and teardown."""
    logger.info("Initializing ML Inference Microservice services...")
    
    # 1. Load Embedding Model
    get_embedding_service()
    
    # 2. Connect to Redis Stack & Initialize Index
    await cache_service.connect()
    if cache_service.redis:
        REDIS_CONNECTED.set(1)
    else:
        REDIS_CONNECTED.set(0)

    yield  # Server runs here
    
    # Teardown
    logger.info("Shutting down microservice connections...")
    await cache_service.close()


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for local testing and web dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Prometheus /metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.get("/health")
async def health_check():
    """Health check endpoint checking Redis and service status."""
    redis_healthy = cache_service.redis is not None
    return {
        "status": "healthy" if redis_healthy else "degraded",
        "redis_connected": redis_healthy,
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "llm_provider": settings.DEFAULT_LLM_PROVIDER,
        "similarity_threshold": settings.SIMILARITY_THRESHOLD,
    }


@app.get("/v1/models")
async def list_models():
    """OpenAI standard models list endpoint."""
    return {
        "object": "list",
        "data": [
            {
                "id": settings.DEFAULT_MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "mlops-infrastructure"
            },
            {
                "id": "mistral-7b-instruct",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "mlops-infrastructure"
            }
        ]
    }


@app.post("/v1/chat/completions")
async def chat_completions(request_body: ChatCompletionRequest, background_tasks: BackgroundTasks):
    """
    OpenAI-Compatible Chat Completion Endpoint backed by Dual-Layer Redis Cache:
    1. Exact Hash Match (< 2ms)
    2. RediSearch HNSW Vector Similarity Search (< 15ms)
    3. LLM Inference Engine Cache Miss Fallback
    """
    start_time = time.perf_counter()
    user_prompt = request_body.extract_user_prompt()
    system_prompt = request_body.extract_system_prompt()

    if not user_prompt:
        raise HTTPException(status_code=400, detail="Request must contain at least one user message.")

    # Generate Composite Hash and System Hash to prevent context drift
    system_hash, composite_hash = cache_service.generate_composite_hash(
        model=request_body.model,
        system_prompt=system_prompt,
        temperature=request_body.temperature,
        user_prompt=user_prompt
    )

    # ------------------------------------------------------------------------
    # STEP 1: Exact Hash Cache Lookup (< 2ms)
    # ------------------------------------------------------------------------
    exact_hit_response = await cache_service.get_exact(composite_hash)
    if exact_hit_response:
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        CACHE_HITS_TOTAL.labels(hit_type="exact").inc()
        REQUEST_LATENCY_SECONDS.labels(cache_status="exact_hit").observe(latency_ms / 1000.0)
        
        # Track estimated metrics saved
        estimated_tokens = len(exact_hit_response.split()) * 4 / 3
        TOKENS_SAVED_TOTAL.inc(estimated_tokens)
        ESTIMATED_COST_SAVED_USD.inc(estimated_tokens * 0.000002)

        logger.info(f"CACHE HIT (Exact) in {latency_ms:.2f}ms")
        return format_response(request_body, exact_hit_response, cached=True, cache_type="exact", latency_ms=latency_ms)

    # ------------------------------------------------------------------------
    # STEP 2: Semantic Vector Cache Lookup (< 15ms)
    # ------------------------------------------------------------------------
    emb_service = get_embedding_service()
    query_vector = await emb_service.get_embedding(user_prompt)
    semantic_hit = await cache_service.query_semantic(
        model=request_body.model,
        system_hash=system_hash,
        vector=query_vector
    )

    if semantic_hit:
        cached_response, similarity = semantic_hit
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        CACHE_HITS_TOTAL.labels(hit_type="semantic").inc()
        REQUEST_LATENCY_SECONDS.labels(cache_status="semantic_hit").observe(latency_ms / 1000.0)

        estimated_tokens = len(cached_response.split()) * 4 / 3
        TOKENS_SAVED_TOTAL.inc(estimated_tokens)
        ESTIMATED_COST_SAVED_USD.inc(estimated_tokens * 0.000002)

        logger.info(f"CACHE HIT (Semantic: sim={similarity:.4f}) in {latency_ms:.2f}ms")
        return format_response(request_body, cached_response, cached=True, cache_type="semantic", latency_ms=latency_ms)

    # ------------------------------------------------------------------------
    # STEP 3: Cache MISS -> Route to LLM Engine
    # ------------------------------------------------------------------------
    CACHE_MISSES_TOTAL.inc()
    llm_provider = get_llm_provider(settings.DEFAULT_LLM_PROVIDER)
    logger.info("CACHE MISS - Routing query to LLM Inference Engine...")

    if not request_body.stream:
        # Non-streaming execution
        response_text = await llm_provider.generate(request_body)
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        REQUEST_LATENCY_SECONDS.labels(cache_status="miss").observe(latency_ms / 1000.0)

        # Asynchronous non-blocking background cache population
        asyncio.create_task(
            cache_service.set_async(
                composite_hash=composite_hash,
                system_hash=system_hash,
                model=request_body.model,
                user_prompt=user_prompt,
                response_text=response_text,
                vector=query_vector
            )
        )

        return format_response(request_body, response_text, cached=False, cache_type=None, latency_ms=latency_ms)
    
    else:
        # Streaming SSE execution with non-blocking tail cache write & cancellation protection
        return StreamingResponse(
            stream_generator(
                request_body=request_body,
                llm_provider=llm_provider,
                composite_hash=composite_hash,
                system_hash=system_hash,
                user_prompt=user_prompt,
                query_vector=query_vector,
                start_time=start_time
            ),
            media_type="text/event-stream"
        )


def format_response(
    request_body: ChatCompletionRequest,
    response_text: str,
    cached: bool,
    cache_type: str,
    latency_ms: float
):
    """Formats string response into standard OpenAI ChatCompletionResponse object."""
    if request_body.stream:
        # If client requested stream but we hit cache, simulate fast SSE stream chunks
        async def cached_sse_stream():
            chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            words = response_text.split()
            for i, word in enumerate(words):
                chunk = ChatCompletionChunk(
                    id=chunk_id,
                    model=request_body.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatCompletionChunkDelta(content=word + (" " if i < len(words) - 1 else "")),
                            finish_reason=None if i < len(words) - 1 else "stop"
                        )
                    ]
                )
                yield f"data: {chunk.model_dump_json()}\n\n"
                await asyncio.sleep(0.002)  # Ultra-fast 2ms chunk playback for cache hit
            yield "data: [DONE]\n\n"

        return StreamingResponse(cached_sse_stream(), media_type="text/event-stream")

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    prompt_tokens = sum(len(m.content.split()) for m in request_body.messages)
    completion_tokens = len(response_text.split())

    return ChatCompletionResponse(
        id=completion_id,
        model=request_body.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=response_text),
                finish_reason="stop"
            )
        ],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        ),
        cached=cached,
        cache_type=cache_type,
        latency_ms=round(latency_ms, 2)
    )


async def stream_generator(
    request_body: ChatCompletionRequest,
    llm_provider,
    composite_hash: str,
    system_hash: str,
    user_prompt: str,
    query_vector: list,
    start_time: float
) -> AsyncGenerator[str, None]:
    """
    Async SSE stream generator with try...finally safety to handle client cancellation
    and trigger non-blocking post-stream Redis cache population.
    """
    accumulated_tokens: list[str] = []
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    try:
        async for token in llm_provider.generate_stream(request_body):
            accumulated_tokens.append(token)
            chunk = ChatCompletionChunk(
                id=chunk_id,
                model=request_body.model,
                choices=[
                    ChatCompletionChunkChoice(
                        index=0,
                        delta=ChatCompletionChunkDelta(content=token),
                        finish_reason=None
                    )
                ]
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

        # Final completion chunk
        final_chunk = ChatCompletionChunk(
            id=chunk_id,
            model=request_body.model,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionChunkDelta(),
                    finish_reason="stop"
                )
            ]
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    except asyncio.CancelledError:
        logger.warning("SSE Stream client connection cancelled prematurely.")
        raise
    finally:
        # Non-blocking post-stream Redis Cache write execution
        if accumulated_tokens:
            full_response = "".join(accumulated_tokens)
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            REQUEST_LATENCY_SECONDS.labels(cache_status="miss").observe(latency_ms / 1000.0)

            asyncio.create_task(
                cache_service.set_async(
                    composite_hash=composite_hash,
                    system_hash=system_hash,
                    model=request_body.model,
                    user_prompt=user_prompt,
                    response_text=full_response,
                    vector=query_vector
                )
            )


@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard():
    """Serves dark-mode visual analytics dashboard and live testing sandbox."""
    try:
        with open("app/static/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return "<html><body><h1>Dashboard HTML file loading...</h1></body></html>"
