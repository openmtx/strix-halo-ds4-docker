# ds4 ROCm Server

Multi-stage Docker setup for [antirez/ds4](https://github.com/antirez/ds4) with ROCm GPU acceleration on Strix Halo / AMD APUs.

## Prerequisites

- Docker with `nvidia-container-toolkit` not needed — uses native `/dev/kfd` and `/dev/dri`
- ROCm-compatible GPU (gfx1151 tested)
- ~124 GiB GTT (kernel param: `amdgpu.gttsize=126976`)
- 128 GB system RAM recommended

## Model Download

```sh
uv tool run --from huggingface-hub hf download \
  antirez/deepseek-v4-gguf \
  DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --local-dir ./models
```

## Build & Run

```sh
docker compose build
docker compose up -d
```

The server listens on `http://localhost:8000`.

## Configuration

Edit `docker-compose.yml` to adjust:

| Flag | Default | Description |
|------|---------|-------------|
| `--ctx` | 262144 | Context window size (tokens) |
| `--prefill-chunk` | 4096 | Prompt prefill chunk size |

## Test

```sh
uv run --with openai python3 tests/test_api.py
```

## Notes

- Server is **single-worker sequential** — no parallel request processing
- Model ID exposed as `deepseek-v4-flash`
- Supports OpenAI-compatible chat completions, tool calling, streaming
- Thinking/reasoning supported via `reasoning_effort` parameter
