# Nitro-SLAM (GPU-accelerated ORB-SLAM3 fork from SFU RSL) image: nitroslam:latest
#
# Nitro's own repo Dockerfile only sets up the base environment and never clones
# or builds the project (that is done in its .devcontainer), so this is a
# complete, self-contained build. It targets the EuRoC stereo-inertial path
# (stereo_inertial_euroc), the one driver that activates Nitro's FastTrack /
# TurboMap / FastLoop GPU modules.
#
# Build with deps/slam-algorithms/nitroslam-docker/build.sh (honors
# $CONTAINER_RUNTIME / first arg: docker|podman).

# devel base (not runtime) so nvcc + CUDA dev libs are present to compile the
# native CUDA kernels and the Kompute/Vulkan compute backend. Fully qualified
# (docker.io/) so headless `podman build` resolves it under
# short-name-mode=enforcing without a TTY prompt.
FROM docker.io/nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV NVIDIA_VISIBLE_DEVICES=all
ARG DEBIAN_FRONTEND=noninteractive

# Base tooling + Pangolin prerequisites (mirrors Nitro's Dockerfile).
RUN apt-get update && apt-get -y install \
    build-essential cmake git ninja-build g++ \
    libgl1-mesa-dev libwayland-dev libxkbcommon-dev wayland-protocols libegl1-mesa-dev \
    libc++-dev libglew-dev libeigen3-dev \
    libjpeg-dev libpng-dev \
    libavcodec-dev libavutil-dev libavformat-dev libswscale-dev libavdevice-dev

# ORB-SLAM3 + CUDA dependencies. cudss is pulled from the CUDA apt repo that the
# devel base image already has configured.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata && \
    apt-get -y install \
    libopencv-dev libopencv-core-dev libboost-serialization-dev libssl-dev \
    cudss-cuda-12 \
    python3-dev python3-setuptools \
    # Vulkan / Kompute (Graphite GPU backend) toolchain.
    libvulkan-dev vulkan-tools glslang-tools python3-click xxd \
    # Headless display for xvfb-run (parity with the ORB-SLAM3 image).
    xvfb xauth x11-utils

# Pangolin v0.6 (visualization library ORB-SLAM3/Nitro link against).
RUN git clone --branch v0.6 --recursive https://github.com/stevenlovegrove/Pangolin.git /tmp/Pangolin && \
    cd /tmp/Pangolin && \
    cmake -B build -GNinja && \
    cmake --build build && \
    cd build && ninja install && \
    cd / && rm -rf /tmp/Pangolin

# NVIDIA Vulkan + EGL ICD configuration (required for the Kompute GPU backend).
RUN mkdir -p /etc/vulkan/icd.d && \
    cat > /etc/vulkan/icd.d/nvidia_icd.json <<'EOF'
{
    "file_format_version" : "1.0.0",
    "ICD": {
        "library_path": "libGLX_nvidia.so.0",
        "api_version" : "1.3.194"
    }
}
EOF

RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    cat > /usr/share/glvnd/egl_vendor.d/10_nvidia.json <<'EOF'
{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
EOF

RUN mkdir -p /etc/vulkan/implicit_layer.d && \
    cat > /etc/vulkan/implicit_layer.d/nvidia_layers.json <<'EOF'
{
    "file_format_version" : "1.0.0",
    "layer": {
        "name": "VK_LAYER_NV_optimus",
        "type": "INSTANCE",
        "library_path": "libGLX_nvidia.so.0",
        "api_version" : "1.3.194",
        "implementation_version" : "1",
        "description" : "NVIDIA Optimus layer",
        "functions": {
            "vkGetInstanceProcAddr": "vk_optimusGetInstanceProcAddr",
            "vkGetDeviceProcAddr": "vk_optimusGetDeviceProcAddr"
        },
        "enable_environment": {
            "__NV_PRIME_RENDER_OFFLOAD": "1"
        },
        "disable_environment": {
            "DISABLE_LAYER_NV_OPTIMUS_1": ""
        }
    }
}
EOF

# Clone Nitro-SLAM, pinned for reproducibility. The Thirdparty modules
# (g2o/compute-engine, graphite, pose-graph-optimizer) are vendored directly in
# the repo tree (committed content, not live gitlinks), so a plain clone pulls
# everything; no --recursive / SSH submodule handling is needed. The ORB
# vocabulary is a regular committed file (no Git LFS). Bump NITRO_SLAM_COMMIT to
# update.
ARG NITRO_SLAM_COMMIT=8a8ea7d768d316c2ee72dc1b13e1199f35a09f0a
RUN git clone https://github.com/sfu-rsl/Nitro-SLAM.git /Nitro-SLAM && \
    cd /Nitro-SLAM && git checkout "${NITRO_SLAM_COMMIT}"

# Restore source files that Nitro's .gitignore drops. Its ignore list contains
# `time*` / `Time*` (intended for profiling logs), but those patterns also match
# git-wide and exclude g2o's stuff/timeutil.{cpp,h} and DBoW2's
# DUtils/Timestamp.{cpp,h} from the repo entirely, so a clean clone cannot build
# g2o or DBoW2. These are the stock files from ORB-SLAM3's g2o/DBoW2 (Nitro's
# lineage), vendored in this build context under time_sources_patch/.
COPY time_sources_patch/timeutil.cpp  /Nitro-SLAM/Thirdparty/g2o/g2o/stuff/timeutil.cpp
COPY time_sources_patch/timeutil.h    /Nitro-SLAM/Thirdparty/g2o/g2o/stuff/timeutil.h
COPY time_sources_patch/Timestamp.cpp /Nitro-SLAM/Thirdparty/DBoW2/DUtils/Timestamp.cpp
COPY time_sources_patch/Timestamp.h   /Nitro-SLAM/Thirdparty/DBoW2/DUtils/Timestamp.h

# Build the Thirdparty libs (DBoW2, g2o + Kompute compute-engine, Sophus),
# uncompress the ORB vocabulary, and build Nitro-SLAM. build.sh does all of it.
RUN cd /Nitro-SLAM && chmod +x build.sh && ./build.sh

# Nitro's build.sh has no `set -e` (a failed sub-build is silently skipped), so
# verify the target binary was actually produced instead of trusting the script.
RUN test -x /Nitro-SLAM/Examples/Stereo-Inertial/stereo_inertial_euroc || \
    (echo "ERROR: stereo_inertial_euroc was not built (check the build log above)" >&2 && exit 1)

# === SAL real-time deadline support ===
# Plumb the generic C++ deadline iterator into stereo_inertial_euroc so frames
# whose wall-clock deadline passed (tracking too slow) get dropped in the real
# frame loop. deadline_iterator.h is the single source of truth from
# src/runtime_stress/, staged into the build context by build.sh (use build.sh,
# not a raw build). Appended after the build so the heavy CUDA/Vulkan compile
# stays cached; recompiles ONLY the stereo_inertial_euroc target.
COPY deadline_iterator.h      /Nitro-SLAM/Examples/Stereo-Inertial/deadline_iterator.h
COPY stereo_inertial_euroc.cc /Nitro-SLAM/Examples/Stereo-Inertial/stereo_inertial_euroc.cc
RUN cd /Nitro-SLAM/build && make stereo_inertial_euroc -j"$(nproc)"

# Volume mount points used by the SAL wrapper.
RUN mkdir -p /dataset /output

# Make the built shared libraries discoverable at runtime.
ENV LD_LIBRARY_PATH=/Nitro-SLAM/lib:/Nitro-SLAM/Thirdparty/DBoW2/lib:/Nitro-SLAM/Thirdparty/g2o/lib:/Nitro-SLAM/Thirdparty/Sophus/lib:${LD_LIBRARY_PATH}

WORKDIR /Nitro-SLAM

CMD ["/bin/bash"]
