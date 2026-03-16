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

WORKDIR /orbslam3

CMD ["/bin/bash"]
