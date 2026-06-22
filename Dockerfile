# ==========================================
# STAGE 1: Build the binary inside the Dev image
# ==========================================
FROM rocm/dev-ubuntu-24.04:7.2.4-complete AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git build-essential && rm -rf /var/lib/apt/lists/*

ENV PATH="${PATH}:/opt/rocm/bin"
ENV ROCM_PATH="/opt/rocm"

WORKDIR /build
RUN git clone https://github.com/antirez/ds4.git .
RUN make rocm -j"$(nproc)"

# ==========================================
# STAGE 2: Create a minimal production image
# ==========================================
FROM ubuntu:24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Install ONLY the base hardware runtimes needed to execute HIP binaries
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    numactl \
    && rm -rf /var/lib/apt/lists/*

# Add the AMD repo to fetch just the tiny core runtime libraries (no compiler)
RUN mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor -o /etc/apt/keyrings/rocm.gpg

RUN tee /etc/apt/sources.list.d/rocm.list << 'EOF'
deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/7.2.4 noble main
deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/graphics/7.2.4/ubuntu noble main
EOF

RUN tee /etc/apt/preferences.d/rocm-pin-600 << 'EOF'
Package: *
Pin: release o=repo.radeon.com
Pin-Priority: 600
EOF

# Install just the core lightweight HIP execution runtime
RUN apt-get update && apt-get install -y \
    hip-runtime-amd \
    hipblas \
    hipblaslt \
    && rm -rf /var/lib/apt/lists/*

# Set up paths and hardware compliance variables
ENV PATH="${PATH}:/opt/rocm/bin"
ENV LD_LIBRARY_PATH="/opt/rocm/lib"
ENV HSA_OVERRIDE_GFX_VERSION=11.5.1

# Rename the default ubuntu user (UID/GID 1000) to ds4
RUN usermod -d /app -s /bin/bash -l ds4 ubuntu && \
    groupmod -n ds4 ubuntu

WORKDIR /app

# Install compiled binaries into PATH
COPY --from=builder /build/ds4 /usr/local/bin/ds4
COPY --from=builder /build/ds4-server /usr/local/bin/ds4-server

# Create model directory and set ownership
RUN mkdir -p /app/models && \
    chown -R ds4:ds4 /app

USER ds4

