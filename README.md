# Strix Halo DS4 Docker

Multi-stage Docker build for [antirez/ds4](https://github.com/antirez/ds4) with ROCm GPU acceleration, optimised for **AMD Strix Halo APUs** (Radeon 8060S / gfx1151).

This repo uses a **pinned git submodule** to track the ds4 source. Updating the submodule to a specific commit triggers a rebuild via your CI/CD workflow.

---

## Prerequisites

- Docker with `nvidia-container-toolkit` **not needed** — uses native `/dev/kfd` and `/dev/dri`
- ROCm-compatible GPU (gfx1151 tested on Strix Halo)
- ~124 GiB GTT — requires kernel cmdline tuning (see below)
- 128 GB system RAM recommended

## Kernel Tuning for Large GTT

The 81 GiB model requires a larger GART table than the kernel default. Edit
`/etc/default/grub` and add these parameters to `GRUB_CMDLINE_LINUX_DEFAULT`:

```
amdgpu.gttsize=126976 ttm.pages_limit=32505856 ttm.page_pool_size=32505856 amd_iommu=off
```

Then update grub and reboot:

```sh
sudo update-grub && sudo reboot
```

Verify the setting:

```sh
cat /sys/module/amdgpu/parameters/gttsize
# Should be: 126976
```

On systems with >=128 GiB RAM, this allocates ~124 GiB of GTT for GPU memory,
enough to cache the full model (~81 GiB) plus a large KV context.

## Model Download

```sh
uv tool run --from huggingface-hub hf download \
  antirez/deepseek-v4-gguf \
  DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --local-dir ./models
```

## Build

```sh
docker compose build
```

The build copies source from the local `ds4/` submodule rather than cloning
from GitHub at runtime. This makes the build **deterministic** — you always get
the exact commit pinned in the submodule.

### Updating the submodule

```sh
cd ds4
git checkout <commit-hash>    # or pull latest
cd ..
git add ds4
git commit -m "Update ds4 to <commit-hash>"
git push                      # triggers your CI workflow
```

## Run

```sh
docker compose up -d
```

The server listens on `http://localhost:8000`.

## Configuration

Edit `docker-compose.yml` to adjust:

| Flag | Default | Description |
|------|---------|-------------|
| `--ctx` | 131072 | Context window size (tokens) |
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

## Repository Structure

```
.
├── Dockerfile          # Multi-stage ROCm build
├── docker-compose.yml  # Service definition
├── ds4/                # Git submodule (pinned source)
├── models/             # Downloaded GGUF models (gitignored)
└── tests/              # Benchmark and quality tests
```
