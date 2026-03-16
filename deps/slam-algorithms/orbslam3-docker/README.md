# Docker Setup for ORB-SLAM3

Minimal Docker setup for running ORB-SLAM3 on your datasets.

## Quick Start

### Build the Image

```bash
cd docker
chmod +x build.sh
./build.sh
```

**Note**: Building takes 10-20 minutes as it compiles OpenCV, Pangolin, and ORB-SLAM3.

### Run with Volume Mounting

```bash
chmod +x run.sh
./run.sh
```

This automatically mounts:
- `./datasets` → `/datasets` (read-only)
- `./results` → `/results` (read-write)

## Manual Docker Commands

### Run with Custom Volume Paths

```bash
docker run --rm -it \
    -v /path/to/your/datasets:/datasets:ro \
    -v /path/to/your/results:/results \
    slamperturbationlab:latest
```

The `:ro` flag makes datasets read-only (safer).

### Run ORB-SLAM3 on TUM Dataset

Inside the container:

```bash
# Example: RGB-D TUM dataset
./Examples/RGB-D/rgbd_tum \
    Vocabulary/ORBvoc.txt \
    Examples/RGB-D/TUM1.yaml \
    /datasets/TUM/freiburg1/rgbd_dataset_freiburg1_desk \
    /datasets/TUM/freiburg1/rgbd_dataset_freiburg1_desk/associations.txt
```

Output trajectory will be saved as `CameraTrajectory.txt` in the working directory.

### Save Results Outside Container

```bash
# Inside container, save to /results which is mounted to your host
./Examples/RGB-D/rgbd_tum \
    Vocabulary/ORBvoc.txt \
    Examples/RGB-D/TUM1.yaml \
    /datasets/TUM/freiburg1/rgbd_dataset_freiburg1_desk \
    /datasets/TUM/freiburg1/rgbd_dataset_freiburg1_desk/associations.txt

# Copy trajectory to results folder
cp CameraTrajectory.txt /results/baseline_trajectory.txt

# Run on perturbed data
./Examples/RGB-D/rgbd_tum \
    Vocabulary/ORBvoc.txt \
    Examples/RGB-D/TUM1.yaml \
    /datasets/perturbed/fog_heavy \
    /datasets/perturbed/fog_heavy/associations.txt

cp CameraTrajectory.txt /results/fog_trajectory.txt
```

## Docker Image Contents

- **Ubuntu 20.04** base
- **OpenCV 4.4.0** (built from source)
- **Pangolin v0.6** (built from source)
- **ORB-SLAM3** at `/opt/ORB_SLAM3` with vocabulary file

## Directory Structure

```
/opt/ORB_SLAM3/              # ORB-SLAM3 installation
├── Examples/
│   ├── RGB-D/               # rgbd_tum executable
│   ├── Monocular/
│   └── Stereo/
├── Vocabulary/
│   └── ORBvoc.txt           # Vocabulary file
└── lib/
    └── libORB_SLAM3.so

/datasets/                   # Your datasets (mounted volume)
/results/                    # Output directory (mounted volume)
```

## Usage Examples

### Process Baseline Dataset

```bash
docker run --rm -it \
    -v $(pwd)/datasets:/datasets:ro \
    -v $(pwd)/results:/results \
    slamperturbationlab:latest \
    bash -c "cd /opt/ORB_SLAM3 && \
    ./Examples/RGB-D/rgbd_tum \
        Vocabulary/ORBvoc.txt \
        Examples/RGB-D/TUM1.yaml \
        /datasets/TUM/freiburg1/rgbd_dataset_freiburg1_desk \
        /datasets/TUM/freiburg1/rgbd_dataset_freiburg1_desk/associations.txt && \
    cp CameraTrajectory.txt /results/baseline_trajectory.txt"
```

### Process Perturbed Dataset

```bash
docker run --rm -it \
    -v $(pwd)/datasets:/datasets:ro \
    -v $(pwd)/results:/results \
    slamperturbationlab:latest \
    bash -c "cd /opt/ORB_SLAM3 && \
    ./Examples/RGB-D/rgbd_tum \
        Vocabulary/ORBvoc.txt \
        Examples/RGB-D/TUM1.yaml \
        /datasets/perturbed_fog \
        /datasets/perturbed_fog/associations.txt && \
    cp CameraTrajectory.txt /results/fog_trajectory.txt"
```

## Why Use Volumes?

**Advantages:**
- Image stays small (no dataset bloat)
- Easy to swap datasets without rebuilding
- Results automatically saved to host
- Can process multiple datasets with same image

**Without volumes (bad):**
- Datasets baked into image → huge image size
- Need to rebuild for different data
- Hard to extract results

## Troubleshooting

### Permission Issues

If you get permission denied errors:

```bash
# Run with your user ID
docker run --rm -it \
    --user $(id -u):$(id -g) \
    -v $(pwd)/datasets:/datasets:ro \
    -v $(pwd)/results:/results \
    slamperturbationlab:latest
```

### Path Issues on Windows

If using Git Bash or WSL on Windows:

```bash
# Use Windows paths
docker run --rm -it \
    -v /c/Users/YourName/project/datasets:/datasets:ro \
    -v /c/Users/YourName/project/results:/results \
    slamperturbationlab:latest
```

## Performance Notes

- **Image size**: ~4-5 GB (no datasets included)
- **Build time**: 10-20 minutes
- **Runtime overhead**: Minimal (<5%)
