# Build wrapper for HAMi-core's libvgpu.so.
#
# Run this on the host once. The resulting libvgpu.so is copied to
# /opt/hami/libvgpu.so (see "HAMi GPU Isolation" in README.md) and bind-mounted into
# SLAM containers at runtime. HAMi is CUDA-version-coupled; pin the
# commit to a tag that matches the CUDA runtime baked into your SLAM
# images. Wrong version → HAMi silently falls through.
#
# Typical build (from repo root):
#   docker build -t sal-hami-builder -f docker/hami/Dockerfile .
#   docker create --name sal-hami-extract sal-hami-builder
#   docker cp sal-hami-extract:/out/libvgpu.so /opt/hami/libvgpu.so
#   docker rm sal-hami-extract

ARG CUDA_IMAGE=nvidia/cuda:12.9.1-cudnn-devel-ubuntu20.04
FROM ${CUDA_IMAGE} AS builder

ARG HAMI_CORE_REF=main

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        git \
        pkg-config \
        tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone https://github.com/Project-HAMi/HAMi-core.git . \
    && git checkout "${HAMI_CORE_REF}"

# SAL fix (pending upstream): the dlsym hook's libvgpu self-lookup degrades to
# RTLD_DEFAULT when libvgpu is preloaded from a nonstandard path, resolving
# "cu"-prefixed LIBRARY symbols (cudnnCreate) back to the caller's own export.
# cuDNN 9's dispatcher then recurses into itself until stack overflow: any
# LibTorch >= 2.5 cu12x app dies at its first convolution. Verified fix:
# recover the self-handle via dladdr, treat a NULL handle as a miss, and fire
# preInit only on a confirmed hook hit. See the patch header for details.
COPY docker/hami/libvgpu-dlsym-self-handle.patch /tmp/
RUN git apply /tmp/libvgpu-dlsym-self-handle.patch

# Build only the shared library target (vgpu -> libvgpu.so). The default
# `all` target also links in-tree CUDA test binaries that fail to link
# in our base image (and are not needed at runtime).
RUN mkdir -p build \
    && cd build \
    && cmake .. \
         -DDLSYM_HOOK_ENABLE=1 \
         -DMULTIPROCESS_LIMIT_ENABLE=1 \
         -DHOOK_MEMINFO_ENABLE=1 \
         -DHOOK_NVML_ENABLE=1 \
         -DCMAKE_BUILD_TYPE=Debug \
    && make -j"$(nproc)" vgpu

RUN mkdir -p /out \
    && cp build/libvgpu.so /out/libvgpu.so \
    && chmod 644 /out/libvgpu.so

# Minimal final stage: hold the artifact for `docker cp` extraction.
FROM scratch
COPY --from=builder /out/libvgpu.so /out/libvgpu.so
