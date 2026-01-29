import time

from prometheus_client import Counter, Histogram, Gauge, generate_latest

llm_requests_total = Counter(
    "llm_requests_total",
    "Total number of LLM requests",
    ["endpoint", "status"],
)

llm_request_duration_seconds = Histogram(
    "llm_request_duration_seconds",
    "Time spent processing LLM requests",
    ["endpoint"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

llm_tokens_generated = Counter(
    "llm_tokens_generated_total",
    "Total number of tokens generated",
    ["endpoint"],
)

llm_request_size = Histogram(
    "llm_request_size_bytes",
    "Size of incoming requests",
    ["endpoint"],
    buckets=[100, 500, 1000, 5000, 10000, 50000, 100000],
)

llm_active_requests = Gauge(
    "llm_active_requests",
    "Number of currently active requests",
    ["endpoint"],
)


def start_request(endpoint: str, request_size: int) -> float:
    llm_active_requests.labels(endpoint=endpoint).inc()
    llm_request_size.labels(endpoint=endpoint).observe(request_size)
    return time.time()


def record_result(endpoint: str, status: str, token_count: int | None = None) -> None:
    llm_requests_total.labels(endpoint=endpoint, status=status).inc()
    if status == "success" and token_count is not None and token_count > 0:
        llm_tokens_generated.labels(endpoint=endpoint).inc(token_count)


def finish_request(endpoint: str, start_time: float) -> None:
    llm_request_duration_seconds.labels(endpoint=endpoint).observe(time.time() - start_time)
    llm_active_requests.labels(endpoint=endpoint).dec()


def get_metrics():
    return generate_latest()
