# Strix Halo DS4 Docker

Multi-stage Docker build for [antirez/ds4](https://github.com/antirez/ds4) with ROCm GPU acceleration, optimised for **AMD Strix Halo APUs** (Radeon 8060S / gfx1151).

This repo uses a **pinned git submodule** to track the ds4 source. Updating the submodule to a specific commit triggers a rebuild via your CI/CD workflow.

## Why Docker?

gfx1151 (Radeon 8060S) is recent enough that **the host distribution's ROCm may lag, mis-detect the part, or not be packaged at all** for your distro — and a flaky host-side ROCm stack is a painful thing to debug on bleeding-edge silicon. This setup sidesteps the host entirely:

- **Build stage** compiles ds4 inside a pinned `rocm/dev-ubuntu-24.04:7.2.4-complete` image — you choose the ROCm version known to support gfx1151, regardless of what's installed on the host.
- **Runtime stage** is plain `ubuntu:24.04`, pulling in only the lightweight HIP execution libraries (`hip-runtime-amd`, `hipblas`, `hipblaslt`) from AMD's apt repo. No compiler, no full ROCm stack in the deployed image.

There is **no host-side ROCm userspace to install or fight with** — the entire ROCm stack lives inside the container, fully under your control. The host only needs a working kernel and `amdgpu` driver exposing `/dev/kfd` and `/dev/dri`, plus the one-time GTT tuning described below.

---

## Prerequisites

- Docker (no `nvidia-container-toolkit` required — passes native `/dev/kfd` and `/dev/dri` through to the container)
- An AMD GPU supported by the ROCm build image (gfx1151 / Radeon 8060S tested)
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

## Clone

This repo vendors [antirez/ds4](https://github.com/antirez/ds4) as a **git submodule**, so clone it recursively (or init the submodule in an existing checkout) — otherwise the build copies an empty `ds4/` directory and fails:

```sh
git clone --recursive <this-repo-url> dwarf-star
# or, in an existing checkout:
git submodule update --init --recursive
```

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

This compiles `ds4` against the full ROCm toolchain in the build stage, so the
first build can take several minutes; subsequent builds are cached. Source is
copied from the local `ds4/` submodule (rather than cloned at build time), which
makes the build **deterministic** — you always get the exact commit pinned in
the submodule.

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

Functional smoke test:

```sh
uv run --with openai python3 tests/test_api.py
```

Quantization stress test (52 probes across factual recall, hallucination resistance, math, reasoning, code, and determinism):

```sh
uv run --with openai python3 tests/test_quant_quality.py
```

Throughput benchmarks (`bench_prefill.py`, `bench_xnack.py`) and the tool-delegation experiment (`tool_delegation_experiment.py`) live alongside these.

## Evaluation

We ran a limited evaluation suite against the live server, and the **2-bit (IQ2XXS) quantized DeepSeek-V4-Flash performs surprisingly well on this 128 GB Strix Halo machine** — strong on factual recall, hallucination resistance, instruction-following, and executed code generation, with measured prefill around ~260 tok/s and decode around ~16 tok/s. A full writeup covering the evaluation methodology, results, and a tool-based "compensation" experiment is forthcoming as a blog post; a link will be added here once it's published.

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
└── tests/              # Functional tests, benchmarks, and eval probes
```
