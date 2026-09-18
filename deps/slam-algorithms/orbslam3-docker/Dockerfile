FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update

RUN apt-get install -y gnupg2 curl lsb-core vim wget python3-pip libpng16-16 libjpeg-turbo8 libtiff5

RUN apt-get install -y \
        # Base tools
        cmake \
        build-essential \
        git \
        unzip \
        pkg-config \
        python3-dev \
        # OpenCV dependencies
        python3-numpy \
        # Pangolin dependencies
        libgl1-mesa-dev \
        libglew-dev \
        libpython3-dev \
        libeigen3-dev \
        apt-transport-https \
        ca-certificates \
        software-properties-common \
        # ORB-SLAM3 additional dependencies
        libboost-all-dev \
        libssl-dev \
        # Headless display and debugging
        xvfb \
        xauth \
        x11-utils \
        gdb

# Build OpenCV (3.0 or higher should be fine)
RUN apt-get install -y python3-dev python3-numpy
RUN apt-get install -y libavcodec-dev libavformat-dev libswscale-dev
RUN apt-get install -y libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev
RUN apt-get install -y libgtk-3-dev

RUN cd /tmp && git clone https://github.com/opencv/opencv.git && \
    cd opencv && \
    git checkout 4.4.0 && \
    mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=Release -D BUILD_EXAMPLES=OFF  -D BUILD_DOCS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_TESTS=OFF -D CMAKE_INSTALL_PREFIX=/usr/local .. && \
    make -j$(nproc) && make install && \
    cd / && rm -rf /tmp/opencv

# Build Pangolin
RUN cd /tmp && git clone https://github.com/stevenlovegrove/Pangolin && \
    cd Pangolin && git checkout v0.6 && mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-std=c++11 .. && \
    make -j$(nproc) && make install && \
    cd / && rm -rf /tmp/Pangolin

# Configure NVIDIA GPU support (Vulkan and EGL)
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

# Clone and build ORB-SLAM (Community Edition - heavydragon99 fork)
RUN cd / && \
    git clone https://github.com/heavydragon99/ORB_SLAM_COMMUNITY.git && \
    mv ORB_SLAM_COMMUNITY orbslam3 && \
    cd orbslam3 && \
    git checkout 9b8baad64abc724a259585b950418df7f9e83cfa && \
    chmod +x build.sh && \
    ./build.sh

# Download ORB vocabulary
RUN cd /orbslam3/Vocabulary && \
    wget https://github.com/UZ-SLAMLab/ORB_SLAM3/raw/master/Vocabulary/ORBvoc.txt.tar.gz && \
    tar -xf ORBvoc.txt.tar.gz && \
    rm ORBvoc.txt.tar.gz

# Create mount points for volumes
RUN mkdir -p /datasets /results

# Set environment variables
ENV ORB_SLAM3_PATH=/orbslam3
ENV LD_LIBRARY_PATH=/orbslam3/lib:${LD_LIBRARY_PATH}

# NOTE (mtime trap): COPY preserves source mtimes, which can be OLDER than the
# cached build products, making `make` skip the recompile and ship an UNPATCHED
# binary that silently ignores SAL_DEADLINE_* (zero frame drops, scored as a
# stressed run). The `touch` prevents it; the `strings` guard fails the build
# loudly if it ever happens again.
# === SAL real-time deadline support ===
# Plumb the generic C++ deadline iterator into the monocular KITTI example so
# frames whose wall-clock deadline passed (because tracking was too slow) get
# dropped inside the real frame loop, the faithful analog of DROID-SLAM's
# Python demo.py wiring. Appended last so every layer above (OpenCV, Pangolin,
# the ORB-SLAM3 build) stays cached; only these layers rebuild when the patch
# changes. deadline_iterator.h is the single source of truth from
# src/runtime_stress/, staged into the build context by build.sh (so use
# build.sh, not a raw `docker build`). Recompiles ONLY the mono_kitti target;
# the ORB-SLAM3 library and all other examples are reused from cache.
COPY deadline_iterator.h /orbslam3/Examples/Monocular/deadline_iterator.h
COPY mono_kitti.cc       /orbslam3/Examples/Monocular/mono_kitti.cc
RUN touch /orbslam3/Examples/Monocular/mono_kitti.cc && \
    cd /orbslam3/build && make mono_kitti -j"$(nproc)" && \
    strings /orbslam3/Examples/Monocular/mono_kitti | grep -q SAL_DEADLINE_FPS

# Same deadline harness plumbed into the EuRoC stereo-inertial example, for the
# ORB-SLAM3-vs-Nitro real-time comparison. Recompiles ONLY the
# stereo_inertial_euroc target; the library and all other examples stay cached.
COPY deadline_iterator.h      /orbslam3/Examples/Stereo-Inertial/deadline_iterator.h
COPY stereo_inertial_euroc.cc /orbslam3/Examples/Stereo-Inertial/stereo_inertial_euroc.cc
RUN touch /orbslam3/Examples/Stereo-Inertial/stereo_inertial_euroc.cc && \
    cd /orbslam3/build && make stereo_inertial_euroc -j"$(nproc)" && \
    strings /orbslam3/Examples/Stereo-Inertial/stereo_inertial_euroc | grep -q SAL_DEADLINE_FPS

# Same deadline harness plumbed into the EuRoC STEREO (vision-only) example.
# This is the controlled counterpart to stereo_inertial_euroc: identical
# sequence and stereo geometry, IMU as the only difference, so frame-drop
# robustness can be attributed to inertial sensing or not. Recompiles ONLY the
# stereo_euroc target; the library and all other examples stay cached.
COPY deadline_iterator.h /orbslam3/Examples/Stereo/deadline_iterator.h
COPY stereo_euroc.cc     /orbslam3/Examples/Stereo/stereo_euroc.cc
RUN touch /orbslam3/Examples/Stereo/stereo_euroc.cc && \
    cd /orbslam3/build && make stereo_euroc -j"$(nproc)" && \
    strings /orbslam3/Examples/Stereo/stereo_euroc | grep -q SAL_DEADLINE_FPS

WORKDIR /orbslam3

CMD ["/bin/bash"]
