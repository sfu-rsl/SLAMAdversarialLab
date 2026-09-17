# OKVIS2-X (ethz-mrl) container image for SAL: stereo visual-inertial SLAM on
# EuRoC, with BOTH the plain VI app (okvis_app_synchronous, CPU) and the
# stereo-depth-network app (okvis2x_app_snetwork_synchronous, CUDA LibTorch).
#
# Build with ./build.sh (stages the SAL deadline iterator header into this
# context), not a raw `docker build`.
#
# CUDA devel base: same tag Nitro-SLAM uses, proven on this host's driver.
# Overridable together with the LibTorch args below to produce the cu118
# variant (okvis2x:cu118) that HAMi/libvgpu can interpose: cu118-built
# LibTorch predates the cuLibrary/cuKernel launch APIs that break HAMi with
# the cu128 stack (see docs/OKVIS2X_PODMAN.md, HAMi + NN workload).
ARG CUDA_IMAGE=docker.io/nvidia/cuda:12.8.1-devel-ubuntu22.04
FROM ${CUDA_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

# Build tools + OKVIS2-X system deps. BUILD_ROS2=OFF drops the only PCL
# dependency, so no ROS/PCL here. OpenCV from apt (22.04 ships 4.5.4, inside
# OKVIS's supported 2.4-4.x range). GeographicLib is required by the apps.
# xvfb: the apps call cv::imshow unconditionally, SAL runs them under
# xvfb-run -a.
RUN apt-get update && apt-get install -y \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        ca-certificates \
        libeigen3-dev \
        libopencv-dev \
        libboost-all-dev \
        libgoogle-glog-dev \
        libgflags-dev \
        libsuitesparse-dev \
        libatlas-base-dev \
        libgeographic-dev \
        xvfb \
        xauth \
        x11-utils \
    && rm -rf /var/lib/apt/lists/*

# LibTorch: official C++ cxx11-ABI shared-with-deps zip for CUDA 12.8. A pip
# torch wheel is NOT used: this is a pure C++ CMake superbuild and the cxx11
# ABI zip guarantees ABI compatibility with the gcc-11-built Ceres/OpenCV/
# brisk below, without dragging Python into the image.
ARG LIBTORCH_VERSION=2.7.1
ARG LIBTORCH_CUDA=cu128
RUN wget -q "https://download.pytorch.org/libtorch/${LIBTORCH_CUDA}/libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2B${LIBTORCH_CUDA}.zip" \
        -O /tmp/libtorch.zip && \
    unzip -q /tmp/libtorch.zip -d /opt && \
    rm /tmp/libtorch.zip

# Clone OKVIS2-X at a pinned commit. The supereight2 submodule is declared
# with an SSH URL (git@github.com:...), which fails in a keyless container
# build; rewrite to https first.
ARG OKVIS2X_COMMIT=38043e4afe56d9b32a98434cc74e723737dd2bce
RUN git config --global url."https://github.com/".insteadOf "git@github.com:" && \
    git clone https://github.com/ethz-mrl/OKVIS2-X.git /okvis2x && \
    cd /okvis2x && \
    git checkout ${OKVIS2X_COMMIT} && \
    git submodule update --init --recursive

# PCL: okvis_multisensor_processing/CMakeLists.txt requires PCL
# unconditionally (independent of BUILD_ROS2, which only gates the top-level
# ROS PCL usage). Installed as its own layer after the clone so the LibTorch
# download and clone layers above stay cached.
RUN apt-get update && apt-get install -y libpcl-dev && rm -rf /var/lib/apt/lists/*

# Build. USE_NN=ON builds the okvis2x_* apps (needs LibTorch); USE_GPU=ON
# runs network inference on CUDA (OKVIS_USE_CUDA compile definition).
# The vendored BRISK/DBoW2/Ceres/opengv superbuild stays at its defaults
# (USE_SYSTEM_*=OFF). CMake downloads the stereo depth model into build/ at
# configure time and renames it to depth-model.pt.
RUN cd /okvis2x && mkdir -p build && cd build && \
    cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_ROS2=OFF \
        -DUSE_NN=ON \
        -DUSE_GPU=ON \
        -DHAVE_LIBREALSENSE=OFF \
        -DBUILD_APPS=ON \
        -DBUILD_TESTS=OFF \
        -DCMAKE_PREFIX_PATH=/opt/libtorch \
        .. && \
    make -j"$(nproc)"

# Guards: both apps built, LibTorch links resolve, the depth model landed in
# build/ (downloaded+renamed at configure), and the DBoW vocabulary ships in
# the cloned tree. A broken image fails HERE, not at first SAL run.
RUN test -x /okvis2x/build/okvis_app_synchronous && \
    test -x /okvis2x/build/okvis2x_app_snetwork_synchronous && \
    ldd /okvis2x/build/okvis2x_app_snetwork_synchronous | grep -q libtorch && \
    ! (ldd /okvis2x/build/okvis_app_synchronous | grep -q "not found") && \
    ! (ldd /okvis2x/build/okvis2x_app_snetwork_synchronous | grep -q "not found") && \
    test -f /okvis2x/build/depth-model.pt && \
    test -f /okvis2x/resources/small_voc.yml.gz

# SAL-owned config copies (vendored + adapted from config/euroc/ at the
# pinned commit; see configs/*.yaml headers for the exact diffs).
COPY configs/euroc_vi.yaml /okvis2x/sal_configs/euroc_vi.yaml
COPY configs/euroc_nn.yaml /okvis2x/sal_configs/euroc_nn.yaml
COPY configs/se2_euroc.yaml /okvis2x/sal_configs/se2_euroc.yaml

# === SAL real-time deadline support ===
# Plumb the generic C++ deadline iterator into BOTH dataset readers so frames
# whose wall-clock deadline passed (because the SLAM was too slow) get dropped
# before delivery, while IMU keeps flowing: DatasetReader drives
# okvis_app_synchronous (VI mode) and XDatasetReader drives the okvis2x_* apps
# (NN mode). Appended last so the superbuild layers stay cached; only the
# okvis_multisensor_processing objects rebuild and the two apps relink.
# deadline_iterator.h is the single source of truth from src/runtime_stress/,
# staged into the build context by build.sh (use build.sh, not raw builds).
COPY deadline_iterator.h /okvis2x/okvis_multisensor_processing/src/deadline_iterator.h
COPY DatasetReader.cpp   /okvis2x/okvis_multisensor_processing/src/DatasetReader.cpp
COPY XDatasetReader.cpp  /okvis2x/okvis_multisensor_processing/src/XDatasetReader.cpp
# touch: COPY preserves source mtimes, which can be OLDER than the build
# products of the main build layer (fresh, uncached builds), making `make`
# skip the recompile and ship unpatched binaries. The strings guard below
# catches that; the touch prevents it.
RUN cd /okvis2x && \
    touch okvis_multisensor_processing/src/DatasetReader.cpp \
          okvis_multisensor_processing/src/XDatasetReader.cpp && \
    cd build && \
    make okvis_app_synchronous okvis2x_app_snetwork_synchronous -j"$(nproc)" && \
    strings okvis_app_synchronous | grep -q SAL_DEADLINE_FPS && \
    strings okvis2x_app_snetwork_synchronous | grep -q SAL_DEADLINE_FPS

ENV LD_LIBRARY_PATH=/opt/libtorch/lib:${LD_LIBRARY_PATH}

# Mount points: /dataset (EuRoC root, ro) and /output (results, rw).
RUN mkdir -p /dataset /output

WORKDIR /okvis2x/build

CMD ["/bin/bash"]
