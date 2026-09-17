# cuVSLAM (NVIDIA Isaac) for the SAL runtime-stress framework.
#
# Built from the PUBLISHED WHEEL, not from source. The wheel bundles
# libcuvslam.so (171 MB) and finds it by RPATH ($ORIGIN), so no CUDA dev
# toolkit and no source build are needed -- cudart is statically linked.
#
# BASE VERSION IS LOAD-BEARING, and 12.1 does NOT work. The wheel is tagged
# cu12, so any CUDA 12 toolkit looks acceptable, but libcuvslam.so links
# `cusparseSpMV_preprocess` from libcusparse.so.12 -- a symbol introduced in
# CUDA 12.4. On 12.1.1 the import dies with:
#
#   ImportError: .../libcuvslam.so: undefined symbol:
#     cusparseSpMV_preprocess, version libcusparse.so.12
#
# so the real floor is CUDA >= 12.4, not merely "CUDA 12". The published C++
# SDK archives are built against 12.6.3, which is consistent with that.
#
# 12.8.1 is used because it satisfies the floor and was already present on this
# host. A -runtime base at 12.8 would also work and would be roughly half the
# size; -devel buys nothing here beyond convenience, since the wheel needs no
# compiler (cudart is statically linked and libcuvslam.so ships in the wheel).
#
# Ubuntu 22.04 gives Python 3.10, which is what the cp310 wheel requires.
#
# NVIDIA Community License: the library may only run on NVIDIA GPU/CPU
# systems, and any redistribution of this image must carry the license, so
# LICENSE is copied in rather than left behind in the source tree.
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /cuvslam

# Runtime deps for the driver and for cuVSLAM's own EuRoC loader
# (dataset_utils imports numpy, yaml, PIL and scipy). rerun-sdk and
# opencv are NOT installed: rerun spawns a GUI viewer and there is no
# display in the container.
RUN python3 -m pip install --no-cache-dir \
        "numpy==2.2.4" "pyyaml==6.0.2" "pillow==12.2.0" "scipy==1.14.1"

COPY cuvslam-17.0.0+cu12-cp310-cp310-manylinux_2_35_x86_64.whl /tmp/
RUN python3 -m pip install --no-cache-dir \
        /tmp/cuvslam-17.0.0+cu12-cp310-cp310-manylinux_2_35_x86_64.whl \
    && rm /tmp/cuvslam-17.0.0+cu12-cp310-cp310-manylinux_2_35_x86_64.whl

# cuVSLAM's own EuRoC helpers: rig construction from sensor.yaml, IMU
# calibration, CSV parsing and camera/IMU interleaving. Reused rather than
# reimplemented, because getting stereo-inertial calibration wrong is the
# easiest way to produce a plausible but meaningless trajectory.
COPY euroc_helpers/ /cuvslam/euroc_helpers/
COPY LICENSE /cuvslam/LICENSE

# The SAL driver. COPYed here so the image is self-contained, and
# bind-mounted over at run time by apply_entrypoint_override so host edits
# take effect without a rebuild.
COPY sal_driver.py /cuvslam/sal_driver.py

ENV PYTHONPATH=/cuvslam/euroc_helpers:/cuvslam
