# Nitro-SLAM Docker image (`nitroslam:latest`)

[Nitro-SLAM](https://github.com/sfu-rsl/Nitro-SLAM) is an ORB-SLAM3 fork from SFU
RSL that adds four GPU acceleration modules — **FastTrack** (front-end tracking),
**TurboMap** (local mapping), **FastLoop** (loop closure), and **Graphite** (graph
optimization) — built on CUDA 12.8 + Vulkan/Kompute.

It is wired into SAL as the first **C++ CUDA SLAM** in the runtime-stress
framework: a workload where HAMi GPU/VRAM caps actually bite (ORB-SLAM3 is
CPU-only, so its HAMi plumbing is inert).

## Important: GPU modules only run on the stereo-inertial path

Nitro's kernel controllers (`TrackingKernelController`, `MappingKernelController`,
`LoopClosingKernelController`) default to `is_active=false`. They are only ever
turned on (`activate()` + `setGPURunMode()`) by the **stereo-inertial** drivers
(`Examples/Stereo-Inertial/stereo_inertial_euroc.cc` and the TUM-VI equivalent).

Every other driver — KITTI mono/stereo, TUM RGB-D, the non-inertial EuRoC
binaries (including `stereo_euroc`, which sets the run-mode flags but never calls
`activate()`) — runs as **stock ORB-SLAM3 on CPU**.

So the SAL wrapper (`src/algorithms/nitroslam.py`) targets **EuRoC
stereo-inertial** only, via `stereo_inertial_euroc`.

## Build

```bash
./build.sh            # docker (or $CONTAINER_RUNTIME)
./build.sh podman     # podman
```

This is a heavy build (CUDA 12.8 toolkit, Vulkan/Kompute, g2o, Pangolin) and
takes a while on a cold cache. Notes:

- **Submodules are vendored.** The Thirdparty modules (`g2o/compute-engine`,
  `graphite`, `pose-graph-optimizer`) are committed directly in Nitro's repo
  tree, not live gitlinks, so a plain `git clone` pulls everything. No
  `--recursive`, no SSH, no private-repo access needed. (The `.gitmodules` file
  is vestigial; the `sfu-rsl/pose-graph-optimizer` GitHub repo is private but
  unused — that submodule path's URL points at the public `compute-engine`.)
- **`.gitignore` drops g2o/DBoW2 sources (worked around).** Nitro's `.gitignore`
  contains `time*` / `Time*` (meant for profiling logs) which also match
  git-wide, so `Thirdparty/g2o/g2o/stuff/timeutil.{cpp,h}` and
  `Thirdparty/DBoW2/DUtils/Timestamp.{cpp,h}` were never committed and a clean
  clone cannot build g2o or DBoW2. The Dockerfile `COPY`s them back from
  `time_sources_patch/` (stock files from ORB-SLAM3's g2o/DBoW2, Nitro's
  lineage). Because Nitro's `build.sh` has no `set -e` (failed sub-builds are
  skipped silently), a verification step asserts `stereo_inertial_euroc` was
  actually produced.
- **Vocabulary is a plain file** (`Vocabulary/ORBvoc.txt.tar.gz`, ~42 MB, no Git
  LFS); `build.sh` uncompresses it.
- **Reproducibility:** the Dockerfile pins `NITRO_SLAM_COMMIT`
  (`8a8ea7d7…`). Bump it (or pass `--build-arg NITRO_SLAM_COMMIT=…`) to update.
- `cudss-cuda-12` is installed from the CUDA apt repo the devel base image ships
  with. If the build fails there, that package name/repo is the first thing to
  check.

## Run (what the wrapper does)

The wrapper invokes the inertial binary with a trajectory file name plus the
seven trailing module/kernel arguments (all modules on; Nitro's default
bitmasks). For a single sequence `stereo_inertial_euroc` **requires** the
file-name argument (its `min_num_argc = 6 + 7` guard forces it) and writes
`f_<name>.txt` (full camera trajectory) and `kf_<name>.txt` (keyframe
trajectory, EuRoC format, TUM-compatible) to the working directory; the wrapper
copies them to `/output` as `Camera`/`KeyFrameTrajectory.txt`:

```bash
podman run --rm --device nvidia.com/gpu=all \
  -v <staged_euroc>:/dataset:ro -v <out>:/output \
  nitroslam:latest bash -c '
    xvfb-run -a ./Examples/Stereo-Inertial/stereo_inertial_euroc \
      Vocabulary/ORBvoc.txt Examples/Stereo-Inertial/EuRoC.yaml \
      /dataset /dataset/orbslam3_timestamps.txt \
      nitro  /output  1 1 1  11110 1111 11111 ;
    cp f_nitro.txt  /output/CameraTrajectory.txt ;
    cp kf_nitro.txt /output/KeyFrameTrajectory.txt'
#   file_name=nitro  <statsDir>=/output  FT TM FL  FT_bm TM_bm FL_bm
```

The staged `/dataset` must contain `mav0/cam0/data`, `mav0/cam1/data`,
**`mav0/imu0/data.csv`** (the wrapper stages the IMU stream alongside the stereo
cameras), and `orbslam3_timestamps.txt`. Camera-IMU extrinsics and IMU noise come
from the bundled `Examples/Stereo-Inertial/EuRoC.yaml`.

## Dual runtime & runtime-stress

The wrapper accepts `container_runtime` of `"docker"` (default) or `"podman"`.
The GPU is always attached (`--device nvidia.com/gpu=all` for podman, `--gpus
all` for docker). When a runtime-stress session is active with GPU controls, HAMi
env (`LD_PRELOAD`, `CUDA_DEVICE_MEMORY_LIMIT`, …), the `libvgpu.so` mount, and the
`/tmp/cudevshr.cache` mid-run mutation file are injected and **live** (CUDA
workload). See `configs/slamadversariallab/runtime_stress/` for a smoke config.

Known things to validate on first GPU run:

- HAMi `libvgpu.so` is built against CUDA 12.9; here it runs against a 12.8
  container. Expected to work via forward-compat, but confirm the VRAM cap
  actually bites (re-run HAMi setup "Step 6.5" if it silently no-ops).
- Vulkan/Kompute must initialize inside rootless podman
  (`NVIDIA_DRIVER_CAPABILITIES=all` + CDI `nvidia.com/gpu=all`).
