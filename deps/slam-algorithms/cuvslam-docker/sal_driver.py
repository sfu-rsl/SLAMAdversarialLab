"""SAL driver for cuVSLAM on EuRoC, stereo-inertial or stereo.

cuVSLAM ships bindings, not a runnable pipeline: there is no frame loop and no
trajectory writer in the library, so the caller owns both. That is convenient
here, because it makes THIS file the deadline harness's entry point -- the
compiled `.so` never has to be touched, unlike the C++ SLAMs whose loop is baked
into the image and cannot be bind-mounted over.

Calibration and dataset parsing are NOT reimplemented. `dataset_utils` is
cuVSLAM's own EuRoC helper, copied into the image, and it already builds the
`Rig` from `sensor.yaml` (Brown/radial-tangential when only the stock yaml is
present), re-references every sensor to cam0, and fills the IMU noise terms.
Reimplementing that is the easiest way to produce a plausible but meaningless
trajectory.

THE DEADLINE DECISION, which is the subtle part.

`prepare_frame_metadata_euroc` returns ONE timestamp-sorted list mixing stereo
records and IMU records. On V1_01_easy that is 2912 camera rows against 29120
IMU rows -- a 10:1 ratio. Handing that merged list to DeadlineIterator would:

  * report total_items = 32032 instead of 2912, making every drop rate in the
    framework wrong by an order of magnitude,
  * count IMU samples as "dropped frames", and
  * tear holes in the inertial stream, which cuVSLAM itself warns about.

So the iterator wraps ONLY the camera frames. Every IMU measurement is still
replayed, including those falling in intervals whose camera frame was dropped.
That is also the physically honest model: a loaded system keeps receiving cheap
200 Hz IMU while it fails to keep up with image processing.
"""
import argparse
import os
import sys
import time

import numpy as np

import cuvslam
from dataset_utils import get_rig, load_frame, prepare_frame_metadata_euroc

# Set by the SAL pipeline when a realtime deadline is configured. Absent means
# run every frame, which is the unstressed baseline.
_SAL_DEADLINE_FPS = os.environ.get("SAL_DEADLINE_FPS")


def _load_deadline_iterator():
    """Import DeadlineIterator from the bind-mounted SAL runtime directory.

    The module has no SAL imports by design, so a bare sys.path insert is
    enough and no package install is needed inside this image.
    """
    runtime_path = os.environ.get("SAL_RUNTIME_PATH")
    if not runtime_path:
        raise RuntimeError(
            "SAL_DEADLINE_FPS is set but SAL_RUNTIME_PATH is not. The deadline "
            "harness cannot be loaded, and running without it would silently "
            "produce an unstressed trajectory labelled as deadline-paced."
        )
    if runtime_path not in sys.path:
        sys.path.insert(0, runtime_path)
    from deadline_iterator import DeadlineIterator  # noqa: E402
    return DeadlineIterator


def _mode_from_name(name):
    modes = {
        "inertial": cuvslam.Tracker.OdometryMode.Inertial,
        "multicamera": cuvslam.Tracker.OdometryMode.Multicamera,
        "mono": cuvslam.Tracker.OdometryMode.Mono,
    }
    if name not in modes:
        raise ValueError(f"Unsupported odometry mode '{name}'. Choose from {sorted(modes)}.")
    return modes[name]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="path to the EuRoC mav0 directory")
    ap.add_argument("--output", required=True, help="directory for CameraTrajectory.txt")
    ap.add_argument("--mode", default="inertial",
                    choices=["inertial", "multicamera", "mono"])
    ap.add_argument("--max-frames", type=int, default=0,
                    help="0 = all frames present in the staged dataset")
    args = ap.parse_args()

    odometry_mode = _mode_from_name(args.mode)

    # async_sba is pinned OFF. cuVSLAM documents no determinism guarantee, and
    # asynchronous bundle adjustment makes the result depend on how threads were
    # scheduled -- which is exactly the variable this framework manipulates. A
    # stressor must not be able to change the answer through scheduling alone.
    cfg = cuvslam.Tracker.OdometryConfig(
        async_sba=False,
        enable_observations_export=False,
        enable_final_landmarks_export=False,
        rectified_stereo_camera=False,   # EuRoC is raw; cuVSLAM undistorts internally
        odometry_mode=odometry_mode,
    )

    print(f"cuVSLAM: loading rig from {args.dataset}", flush=True)
    rig = get_rig(args.dataset)
    tracker = cuvslam.Tracker(rig, cfg)
    print(f"cuVSLAM: tracker initialized, odometry_mode={cfg.odometry_mode}", flush=True)

    records = prepare_frame_metadata_euroc(args.dataset, odometry_mode)
    camera_frames = [r for r in records if r["type"] != "imu"]
    imu_records = [r for r in records if r["type"] == "imu"]
    if args.max_frames:
        camera_frames = camera_frames[:args.max_frames]
    print(f"cuVSLAM: {len(camera_frames)} camera frames, "
          f"{len(imu_records)} imu records", flush=True)

    # Only the camera frames are deadline-paced. See the module docstring.
    if _SAL_DEADLINE_FPS:
        DeadlineIterator = _load_deadline_iterator()
        frame_source = DeadlineIterator(
            camera_frames,
            target_fps=float(_SAL_DEADLINE_FPS),
            warmup_frames=int(os.environ.get("SAL_DEADLINE_WARMUP_FRAMES", 0)),
            queue_size=int(os.environ.get("SAL_DEADLINE_QUEUE_SIZE", 1)),
            drop_policy=os.environ.get("SAL_DEADLINE_DROP_POLICY", "drop_oldest"),
        )
        print(f"cuVSLAM: deadline harness active at {_SAL_DEADLINE_FPS} fps "
              f"over {len(camera_frames)} camera frames", flush=True)
    else:
        frame_source = camera_frames

    trajectory = []
    imu_cursor = 0
    imu_registered = 0
    tracked = 0
    lost = 0
    delivered = 0
    started = time.time()

    for frame in frame_source:
        delivered += 1
        timestamp = frame["timestamp"]

        # Replay every IMU sample up to this frame, including samples from
        # intervals whose camera frame was dropped -- the IMU stream does not
        # stop just because the vision pipeline fell behind.
        if odometry_mode == cuvslam.Tracker.OdometryMode.Inertial:
            while imu_cursor < len(imu_records) and \
                    imu_records[imu_cursor]["timestamp"] <= timestamp:
                m = imu_records[imu_cursor]
                meas = cuvslam.ImuMeasurement()
                meas.timestamp_ns = int(m["timestamp"])
                meas.linear_accelerations = np.asarray(m["accel"])
                meas.angular_velocities = np.asarray(m["gyro"])
                tracker.register_imu_measurement(0, meas)
                imu_cursor += 1
                imu_registered += 1

        images = [load_frame(p) for p in frame["images_paths"]]
        estimate, _ = tracker.track(timestamp, images)

        if estimate.world_from_rig is None:
            # NOT silently skipped, unlike the upstream example. Pose count is a
            # health signal in this framework, so a frame that was delivered but
            # not tracked has to be visible in the log rather than inferred from
            # a short trajectory.
            lost += 1
            print(f"cuVSLAM: tracking failed at frame {delivered} ts={timestamp}",
                  flush=True)
            continue

        pose = estimate.world_from_rig.pose
        t = pose.translation
        q = pose.rotation          # (x, y, z, w), which is TUM's order
        trajectory.append((int(timestamp), t[0], t[1], t[2], q[0], q[1], q[2], q[3]))
        tracked += 1

    elapsed = time.time() - started

    # EuRoC ground truth is in NANOSECONDS and this repo's metrics layer scales
    # the evo association window when it sees ns-magnitude stamps. Written as an
    # integer because 1.4e18 ns exceeds float64's exact-integer range, so a
    # decimal round trip would quietly corrupt the low digits.
    os.makedirs(args.output, exist_ok=True)
    traj_path = os.path.join(args.output, "CameraTrajectory.txt")
    with open(traj_path, "w") as f:
        for row in trajectory:
            f.write("%d %.9f %.9f %.9f %.9f %.9f %.9f %.9f\n" % row)

    print(f"cuVSLAM: delivered={delivered} tracked={tracked} lost={lost} "
          f"imu_registered={imu_registered} elapsed={elapsed:.2f}s", flush=True)
    print(f"cuVSLAM: wrote {len(trajectory)} poses to {traj_path}", flush=True)
    print("cuVSLAM: done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
