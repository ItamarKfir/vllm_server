# LLM Server Monitoring with Prometheus and Grafana

This directory contains the monitoring setup for the LLM server using Prometheus and Grafana.

## Architecture Overview

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│  LLM Server │─────▶│  Prometheus  │─────▶│   Grafana   │
│  (Port 8000)│      │  (Port 9090) │      │  (Port 3000)│
└─────────────┘      └──────────────┘      └─────────────┘
     │                      │                      │
     │                      │                      │
  /metrics            Scrapes metrics        Visualizes data
```

### How It Works

**Important Note**: Since you're using `AsyncLLMEngine` directly (not running vLLM as a separate server), vLLM's built-in metrics endpoint is not automatically available. The `/metrics` endpoint we created exposes application-level metrics.

**If you run vLLM as a separate server** (using `python -m vllm.entrypoints.api_server`), it will expose its own `/metrics` endpoint with vLLM-specific metrics like:
- GPU utilization
- Token generation rates  
- Inference latency
- Queue depth
- Model-specific metrics

**Current Setup**:
- Our FastAPI server exposes application-level metrics at `/metrics`
- Tracks: request counts, endpoint performance, custom business logic
- These complement vLLM's metrics if you run vLLM separately

**Prometheus**:
   - Scrapes metrics from LLM server every 15 seconds
   - Stores time-series data
   - Provides query language (PromQL) for data analysis
   - Runs in Docker container on port 9090

3. **Grafana**:
   - Connects to Prometheus as data source
   - Creates visual dashboards with graphs and charts
   - Runs in Docker container on port 3000
   - Default login: admin/admin

## Setup Instructions

### Step 1: Install Dependencies

```bash
pip install prometheus-client==0.21.0
```

### Step 2: Start Monitoring Stack

```bash
cd llm_monitor
docker-compose up -d
```

This will start:
- Prometheus on http://localhost:9090
- Grafana on http://localhost:3000

### Step 3: Verify Metrics Endpoint

Make sure your LLM server is running, then check:

```bash
curl http://localhost:8000/metrics
```

You should see Prometheus metrics in text format.

### Step 4: Access Dashboards

1. **Prometheus UI**: http://localhost:9090
   - Go to "Status" → "Targets" to see if LLM server is being scraped
   - Use "Graph" tab to run PromQL queries

2. **Grafana UI**: http://localhost:3000
   - Login with admin/admin
   - Go to "Dashboards" → "Import"
   - Import the dashboard from `grafana/dashboards/dashboard.json`

## Metrics Explained

### Application-Level Metrics (Our Custom Metrics)

These metrics track your FastAPI application layer:

#### `llm_requests_total`
- **Type**: Counter
- **Description**: Total number of LLM requests through FastAPI
- **Labels**: `endpoint` (stream/nonstream), `status` (success/error)
- **Use**: Track request volume and error rates at application level

#### `llm_request_duration_seconds`
- **Type**: Histogram
- **Description**: Time spent processing requests (end-to-end)
- **Labels**: `endpoint`
- **Use**: Monitor overall latency including FastAPI overhead

#### `llm_tokens_generated_total`
- **Type**: Counter
- **Description**: Total tokens generated (counted at application level)
- **Labels**: `endpoint`
- **Use**: Track token generation rate

#### `llm_request_size_bytes`
- **Type**: Histogram
- **Description**: Size of incoming requests
- **Labels**: `endpoint`
- **Use**: Monitor input sizes

#### `llm_active_requests`
- **Type**: Gauge
- **Description**: Currently active requests
- **Labels**: `endpoint`
- **Use**: Monitor concurrent load

### vLLM Engine Metrics (If Running vLLM Separately)

If you run vLLM as a separate server, it exposes metrics like:
- `vllm:num_requests_running` - Active requests in vLLM
- `vllm:num_requests_waiting` - Queued requests
- `vllm:gpu_cache_usage_perc` - GPU cache utilization
- `vllm:time_to_first_token_seconds` - Time to first token
- `vllm:time_per_output_token_seconds` - Time per output token
- And many more vLLM-specific metrics

## Useful PromQL Queries

### Application-Level Metrics

```promql
# Request rate per second
rate(llm_requests_total[5m])

# 95th percentile latency
histogram_quantile(0.95, rate(llm_request_duration_seconds_bucket[5m]))

# Error rate
rate(llm_requests_total{status="error"}[5m])

# Tokens per second
rate(llm_tokens_generated_total[5m])

# Active requests
llm_active_requests
```

### vLLM Engine Metrics (If Running Separately)

```promql
# vLLM active requests
vllm:num_requests_running

# vLLM queue depth
vllm:num_requests_waiting

# GPU cache usage
vllm:gpu_cache_usage_perc

# Time to first token
vllm:time_to_first_token_seconds

# Tokens per second (vLLM level)
rate(vllm:num_output_tokens_total[5m])
```

## Troubleshooting

### Prometheus can't scrape LLM server

If running LLM server on host and Prometheus in Docker:
- Use `host.docker.internal:8000` in `prometheus.yml`
- On Linux, you may need to add `extra_hosts` to docker-compose.yml:
  ```yaml
  extra_hosts:
    - "host.docker.internal:host-gateway"
  ```

### Grafana can't connect to Prometheus

- Check that Prometheus container is running: `docker ps`
- Verify network: Both should be on `monitoring` network
- Check Grafana logs: `docker logs llm-grafana`

### Metrics not appearing

- Verify `/metrics` endpoint: `curl http://localhost:8000/metrics`
- Check Prometheus targets: http://localhost:9090/targets
- Ensure LLM server is making requests (metrics only appear after requests)

## Stopping Monitoring

```bash
cd llm_monitor
docker-compose down
```

To remove all data:
```bash
docker-compose down -v
```
