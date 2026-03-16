"""Physics-based cracked lens using the camera-failure project."""

import sys
import os
import numpy as np
import cv2
from typing import Optional
from dataclasses import dataclass

CAMERA_FAILURE_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'deps', 'perturbations', 'camera-failure'
)
CAMERA_FAILURE_PATH = os.path.abspath(CAMERA_FAILURE_PATH)
if CAMERA_FAILURE_PATH not in sys.path:
    sys.path.insert(0, CAMERA_FAILURE_PATH)

from ..base import PerturbationModule
from ...utils import get_logger

logger = get_logger(__name__)


@dataclass
class PhysicsCrackParameters:
    """Parameters for physics-based cracked lens simulation."""
    impact_x: float = 0.5
    """Normalized horizontal impact position on the lens, from 0.0 to 1.0."""
    impact_y: float = 0.5
    """Normalized vertical impact position on the lens, from 0.0 to 1.0."""
    impact_force: float = 500.0
    """Impulse magnitude applied at the impact point."""
    break_threshold: float = 300.0
    """Stress level required before glass edges are considered fractured."""
    nn_radius: float = 65.0
    """Neighborhood radius used by the stress propagation graph."""
    num_points: int = 10000
    """Number of sampled support points used to approximate the glass surface."""
    sun_angle: float = 90.0
    """Lighting direction in degrees for rendering bright crack reflections."""
    blur_kernel_size: int = 15
    """Gaussian blur kernel size applied around fractured regions."""
    seed: Optional[int] = None
    """Optional random seed for reproducible crack patterns."""


class CrackedLensPhysicsModule(PerturbationModule):
    """Physics-based cracked lens using camera-failure project."""

    module_name = "cracked_lens"
    module_description = "Physics-based cracked lens with stress propagation"

    PARAMETERS_CLASS = PhysicsCrackParameters

    def __init__(self, config):
        super().__init__(config)
        parameters = self.parameters if self.parameters else {}

        self.params = PhysicsCrackParameters(
            impact_x=parameters.get('impact_x', 0.5),
            impact_y=parameters.get('impact_y', 0.5),
            impact_force=parameters.get('impact_force', 500.0),
            break_threshold=parameters.get('break_threshold', 300.0),
            nn_radius=parameters.get('nn_radius', 65.0),
            num_points=parameters.get('num_points', 10000),
            sun_angle=parameters.get('sun_angle', 90.0),
            blur_kernel_size=parameters.get('blur_kernel_size', 15),
            seed=parameters.get('seed', None),
        )

        self._crack_pattern = None
        self._cached_shape = None

    def _setup(self, context) -> None:
        logger.info(f"  Impact point: ({self.params.impact_x}, {self.params.impact_y})")
        logger.info(f"  Impact force: {self.params.impact_force}")
        logger.info(f"  Break threshold: {self.params.break_threshold}")

    def _generate_crack_pattern(self, H: int, W: int) -> np.ndarray:
        """Generate crack pattern using camera-failure physics simulation."""
        from utils import Node, StressPropagator
        from visualizers import generate_glass_image, direction_comparison
        from scipy.sparse.csgraph import minimum_spanning_tree
        from scipy.spatial import distance

        if self.params.seed is not None:
            np.random.seed(self.params.seed)

        N = int(self.params.num_points)
        points = np.random.uniform(low=[0, 0], high=[H, W], size=(N, 2)).astype('int')

        impact_y_px = int(self.params.impact_y * H)
        impact_x_px = int(self.params.impact_x * W)

        distances = np.sqrt((points[:, 0] - impact_y_px)**2 + (points[:, 1] - impact_x_px)**2)
        impact_pt_idx = np.argmin(distances)

        IMPACT_POINT = np.array([points[impact_pt_idx]])
        IMPACT_ANGLE = 35

        node_features = np.zeros((N, 3))
        node_features[impact_pt_idx] = np.array([
            self.params.impact_force,
            np.cos(np.deg2rad(IMPACT_ANGLE)),
            np.sin(np.deg2rad(IMPACT_ANGLE))
        ])

        all_points = {}
        for i in range(points.shape[0]):
            x, y = points[i][0], points[i][1]
            all_points[(x, y)] = Node(x, y)

        SP = StressPropagator(
            BREAK_THRESHOLD=self.params.break_threshold,
            NN_RADIUS=self.params.nn_radius,
            IMPACT_POINT=IMPACT_POINT,
            IMPACT_FORCE=self.params.impact_force,
            IMPACT_ANGLE=IMPACT_ANGLE,
            K=1,
            all_points=all_points,
            points=points,
        )

        pointsAll = []
        stress_vals = []
        for edge_list in SP.all_edges.values():
            for edge in edge_list:
                stress_vals.append(edge.edge_stress)
                pointsAll.append([edge.source_node.x, edge.source_node.y])
                pointsAll.append([edge.target_node.x, edge.target_node.y])

        if len(pointsAll) == 0:
            logger.warning("No crack edges generated, returning empty pattern")
            return np.zeros((H, W, 3), dtype=np.uint8)

        pointsAll = np.array(pointsAll)
        stress_vals = np.array(stress_vals)

        dist_matrix = distance.squareform(distance.pdist(pointsAll))
        mst = minimum_spanning_tree(dist_matrix)
        mst = mst.toarray().astype(float)

        broken_glass_img3 = np.zeros((H, W, 3), dtype=np.uint8)
        sun_angle = self.params.sun_angle
        sun_angle_rad = np.radians(sun_angle)
        sun_vector = np.array([np.cos(sun_angle_rad), np.sin(sun_angle_rad)])

        for i in range(len(pointsAll)):
            for j in range(i + 1, len(pointsAll)):
                if mst[i, j] != 0:
                    start_point = (int(pointsAll[i, 1]), int(pointsAll[i, 0]))
                    end_point = (int(pointsAll[j, 1]), int(pointsAll[j, 0]))
                    angle, _ = direction_comparison(
                        sun_direction=sun_vector,
                        point1=start_point,
                        point2=end_point
                    )
                    grey_value = max(0, 200 * (1 - abs(np.cos(np.radians(angle)))))
                    line_color = (int(grey_value), int(grey_value), int(grey_value))
                    cv2.line(broken_glass_img3, start_point, end_point, line_color, thickness=1)

        if len(SP.all_edges.keys()) >= 2:
            img = generate_glass_image(
                all_edges=SP.all_edges,
                H=H,
                W=W,
                pts_img=np.zeros((H, W, 3), dtype=np.uint8),
                sun_angle=sun_angle
            )
            return np.maximum(img, broken_glass_img3)

        return broken_glass_img3

    def _apply_crack_overlay(self, image: np.ndarray, crack_pattern: np.ndarray) -> np.ndarray:
        """Apply crack pattern overlay with blur effect (based on PBR.py logic)."""
        h, w = image.shape[:2]

        if crack_pattern.shape[:2] != (h, w):
            crack_pattern = cv2.resize(crack_pattern, (w, h))

        crack_gray = cv2.cvtColor(crack_pattern, cv2.COLOR_BGR2GRAY)

        crack_mask = crack_gray > 10

        result = image.copy()
        result[crack_mask] = crack_pattern[crack_mask]

        kernel_size = self.params.blur_kernel_size
        if kernel_size % 2 == 0:
            kernel_size += 1
        points_to_blur = np.argwhere(crack_mask)

        if len(points_to_blur) > 0:
            blurred = cv2.GaussianBlur(result, (kernel_size, kernel_size), 0)

            blur_mask = cv2.dilate(
                crack_mask.astype(np.uint8) * 255,
                np.ones((kernel_size, kernel_size), np.uint8),
                iterations=1
            )
            blur_mask = blur_mask > 0

            result[blur_mask] = blurred[blur_mask]

        return result

    def _apply(
        self,
        image: np.ndarray,
        depth: Optional[np.ndarray],
        frame_idx: int,
        camera: str = "left",
        **kwargs
    ) -> np.ndarray:
        h, w = image.shape[:2]

        if self._crack_pattern is None or self._cached_shape != (h, w):
            logger.info(f"  Generating crack pattern for {w}x{h} image...")
            self._crack_pattern = self._generate_crack_pattern(h, w)
            self._cached_shape = (h, w)

        return self._apply_crack_overlay(image, self._crack_pattern)

    def _cleanup(self) -> None:
        self._crack_pattern = None
        self._cached_shape = None
