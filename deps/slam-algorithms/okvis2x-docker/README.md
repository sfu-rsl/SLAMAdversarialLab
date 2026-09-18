# OKVIS2-X container image for SAL

Builds `okvis2x:latest`: OKVIS2-X (ethz-mrl) stereo visual-inertial SLAM with
BOTH EuRoC-relevant apps in one image:

| App | SAL algorithm | Workload |
| --- | --- | --- |
| `okvis_app_synchronous` | `okvis2x` | plain visual-inertial, CPU |
| `okvis2x_app_snetwork_synchronous` | `okvis2xnn` | + Unimatch stereo-depth network via CUDA LibTorch, GPU |

## Build

```bash
# From anywhere (uses docker; pass 'podman' to build with podman):
deps/slam-algorithms/okvis2x-docker/build.sh podman
```

`build.sh` stages the generic SAL deadline iterator
(`src/runtime_stress/deadline_iterator.h`, single source of truth shared with
the Python `deadline_iterator.py`) into this build context and removes the
staged copy afterwards. Build with the script, not a raw `docker build`.

A first build takes roughly 30-60 minutes (vendored Ceres/BRISK/DBoW2/opengv
superbuild + supereight2 + OKVIS2-X + the LibTorch download). Rebuilds after a
deadline-patch change reuse the cache and only recompile the affected targets.

## Provenance (pinned)

- Source: https://github.com/ethz-mrl/OKVIS2-X at commit
  `38043e4afe56d9b32a98434cc74e723737dd2bce` (cloned inside the Dockerfile;
  BSD-3 license).
- Depth-network weights: downloaded by OKVIS2-X's own CMake configure step
  into `build/` and renamed to `depth-model.pt` (the repo also ships
  `resources/depth-model.pt` and `resources/fast-scnn.pt` as plain git blobs).
  No manual weight download is needed.
- DBoW2 vocabulary: ships in the cloned tree at `resources/small_voc.yml.gz`.
- The supereight2 submodule is declared with an SSH URL upstream; the
  Dockerfile rewrites it to https (`git config url.insteadOf`) for keyless
  container builds.
- LibTorch: official `libtorch-cxx11-abi-shared-with-deps` zip for CUDA 12.8
  at `/opt/libtorch` (C++ ABI matches the gcc-11-built dependencies; no
  Python/pip in the image).
- CMake: `-DBUILD_ROS2=OFF` (drops PCL/ROS), `-DUSE_NN=ON` (builds the
  `okvis2x_*` apps), `-DUSE_GPU=ON` (CUDA inference), `-DHAVE_LIBREALSENSE=OFF`.

## SAL-owned configs

`configs/euroc_vi.yaml`, `configs/euroc_nn.yaml`, and `configs/se2_euroc.yaml`
are vendored copies of OKVIS2-X's `config/euroc/{okvis2,se2}.yaml` at the
pinned commit, baked into the image at `/okvis2x/sal_configs/`. Each file's
header comment documents the exact diffs vs upstream (camera_type for VI mode,
display flags off for headless runs, submapping off for VI).

## Manual run (outside SAL)

Note: the image is built `USE_GPU=ON`, so LibTorch initializes a CUDA context
at startup in BOTH apps; `--device nvidia.com/gpu=all` is required even for
the plain VI app (without it, c10 aborts with "CUDA driver version is
insufficient" before streaming starts).

```bash
# Plain visual-inertial on EuRoC V1_01_easy:
podman run --rm --device nvidia.com/gpu=all \
  -v $PWD/datasets/euroc/V1_01_easy:/dataset:ro \
  -v /tmp/okvis_out:/output \
  okvis2x:latest \
  ./okvis_app_synchronous /okvis2x/sal_configs/euroc_vi.yaml /dataset/mav0 /output

# Stereo-depth-network mode (needs an X display: unconditional cv::imshow):
podman run --rm --device nvidia.com/gpu=all \
  -v $PWD/datasets/euroc/V1_01_easy:/dataset:ro \
  -v /tmp/okvis_out:/output \
  okvis2x:latest \
  bash -c 'Xvfb :99 -screen 0 1024x768x24 -nolisten tcp & sleep 1; DISPLAY=:99 ./okvis2x_app_snetwork_synchronous /okvis2x/sal_configs/euroc_nn.yaml /okvis2x/sal_configs/se2_euroc.yaml /dataset/mav0 /output'
```

Outputs land in the output directory as `okvis2-<slam|vio>[-calib]_trajectory.csv`
(online) and `okvis2-...-final_trajectory.csv` (loop-closed / final-BA). The
SAL wrapper converts the final CSV to a TUM-format `CameraTrajectory.txt` with
nanosecond timestamps (the EuRoC metrics convention).

Display handling, verified empirically:

- The VI app (`okvis_app_synchronous`) runs bare headless; the vendored
  config disables every display output so it never touches X.
- The NN app (`okvis2x_app_snetwork_synchronous`) calls `cv::imshow`
  unconditionally and needs an X server: start Xvfb manually with a fixed
  display as in the example above.
- Do NOT use `xvfb-run` for either app: it kills the process at startup in
  this image (with or without the GPU attached).
