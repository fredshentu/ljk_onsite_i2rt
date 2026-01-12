from quest_tools.quest3_reader import QuestReader

import numpy as np
import rerun as rr
from dataclasses import dataclass, field
from typing import Optional
from modern_robotics import TransInv, TransToRp, RpToTrans

from scipy.spatial.transform import Rotation


CLUTCH_THRESHOLD = 0.8

# Transformation from Quest controller frame to World frame
# Quest controller frame (natural pose): different axis orientation
# World frame: X right, Y up, Z forward
# This rotation aligns Quest axes with World frame axes
T_QUEST_TO_WORLD = np.array(
    [
        [0, 0, -1, 0],  # World X = -Quest Z
        [-1, 0, 0, 0],  # World Y = -Quest X
        [0, 1, 0, 0],  # World Z = Quest Y
        [0, 0, 0, 1],
    ]
)


@dataclass
class HandState:
    last_pose: Optional[np.ndarray] = None
    clutch_was_pressed: bool = False


@dataclass
class ClutchTracker:
    left: HandState = field(default_factory=HandState)
    right: HandState = field(default_factory=HandState)


def convert_left_to_right_handed(quest_matrix: np.ndarray) -> np.ndarray:
    y_flip_transform = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    intermediate_result = y_flip_transform @ quest_matrix
    final_result = intermediate_result @ TransInv(y_flip_transform)

    return final_result


def compute_relative_transform(
    current_pose: np.ndarray,
    last_pose: np.ndarray,
) -> np.ndarray:
    """Compute relative motion in the controller frame (local/body frame).

    This computes: T_last^{-1} @ T_current
    Which gives the delta expressed in the controller's local coordinate system.
    """
    return TransInv(last_pose) @ current_pose


def log_pose(name: str, pose: np.ndarray):
    if isinstance(pose, np.ndarray):
        if pose.shape == (4, 4):
            position = pose[:3, 3]
            rotation_matrix = pose[:3, :3]
            rr.log(name, rr.Transform3D(translation=position, mat3x3=rotation_matrix))
        elif pose.shape == (7,):
            position = pose[:3]
            quaternion = pose[3:]
            rr.log(
                name,
                rr.Transform3D(
                    translation=position, rotation=rr.Quaternion(xyzw=quaternion)
                ),
            )
        else:
            rr.log(name, rr.Points3D([pose[:3]]))

    rr.log(
        f"{name}/axes",
        rr.Arrows3D(
            origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            vectors=[[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        ),
    )


def init_rerun(app_name: str = "teleop_visualization"):
    rr.init(app_name)
    server_uri = rr.serve_grpc()
    rr.serve_web_viewer(connect_to=server_uri, open_browser=False)


def create_quest_reader(ip: str = "127.0.0.1", port: int = 12345) -> QuestReader:
    return QuestReader(ip=ip, port=port)


def get_vr_poses(reader: QuestReader) -> dict:
    data = reader.get_data()
    return {
        "right_hand": data["Right Hand"],
        "left_hand": data["Left Hand"],
        "head": data["Head"],
    }


def log_vr_poses(poses: dict):
    log_pose("right_hand", poses["right_hand"])
    log_pose("left_hand", poses["left_hand"])
    log_pose("head", poses["head"])


def log_relative_transform(name: str, rel_transform: np.ndarray):
    if rel_transform.shape != (4, 4):
        return

    R, p = TransToRp(rel_transform)

    rr.log(
        name,
        rr.Transform3D(
            translation=p,
            mat3x3=R,
        ),
    )
    rr.log(
        f"{name}/axes",
        rr.Arrows3D(
            origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            vectors=[[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        ),
    )

    r = Rotation.from_matrix(R)
    euler_angles = r.as_euler("xyz", degrees=False)
    roll, pitch, yaw = euler_angles
    x, y, z = p

    # Log translation components as scalars
    rr.log(f"{name}/translation/x", rr.Scalars(x))
    rr.log(f"{name}/translation/y", rr.Scalars(y))
    rr.log(f"{name}/translation/z", rr.Scalars(z))

    # Log rotation components as scalars (in radians)
    rr.log(f"{name}/rotation/roll", rr.Scalars(roll))
    rr.log(f"{name}/rotation/pitch", rr.Scalars(pitch))
    rr.log(f"{name}/rotation/yaw", rr.Scalars(yaw))

    print(
        f"{name}: x={x:.4f}, y={y:.4f}, z={z:.4f}, roll={roll:.4f}, pitch={pitch:.4f}, yaw={yaw:.4f}"
    )


def process_hand_clutch(
    hand_state: HandState,
    current_pose: np.ndarray,
    grip_value: float,
    hand_name: str,
) -> Optional[np.ndarray]:
    clutch_pressed = grip_value > CLUTCH_THRESHOLD
    T_rel = None

    if clutch_pressed:
        current_pose_rh = convert_left_to_right_handed(current_pose)
        if hand_state.last_pose is not None:
            T_rel = compute_relative_transform(current_pose_rh, hand_state.last_pose)
            print(
                f"{hand_name.capitalize()} Hand - Clutch PRESSED - Relative transform: {T_rel}"
            )

        hand_state.last_pose = current_pose_rh.copy()
        hand_state.clutch_was_pressed = True

    else:
        if hand_state.clutch_was_pressed:
            print(
                f"{hand_name.capitalize()} Hand - Clutch RELEASED - Clearing frame buffer"
            )
            hand_state.last_pose = None
            hand_state.clutch_was_pressed = False
            rr.log(f"{hand_name}_hand_relative", rr.Clear(recursive=True))

    return T_rel


if __name__ == "__main__":
    init_rerun()
    reader = create_quest_reader()
    tracker = ClutchTracker()

    import time

    while True:
        data = reader.get_data()
        if not data:
            time.sleep(0.01)
            continue

        poses = {
            "right_hand": data.get("Right Hand", np.eye(4)),
            "left_hand": data.get("Left Hand", np.eye(4)),
            "head": data.get("Head", np.eye(4)),
        }

        left_grip = data.get("Left Grip", 0.0)
        right_grip = data.get("Right Grip", 0.0)

        left_trigger = data.get("Left Trigger", 0.0)
        right_trigger = data.get("Right Trigger", 0.0)

        # Log grip and trigger values to rerun
        rr.log("inputs/left_grip", rr.Scalars(left_grip))
        rr.log("inputs/right_grip", rr.Scalars(right_grip))
        rr.log("inputs/left_trigger", rr.Scalars(left_trigger))
        rr.log("inputs/right_trigger", rr.Scalars(right_trigger))

        # # Visualize VR poses
        # log_vr_poses(poses)

        left_rel = process_hand_clutch(
            tracker.left,
            poses["left_hand"],
            left_grip,
            "left",
        )
        if left_rel is not None:
            # Transform relative motion to world frame
            # left_rel_world = transform_relative_to_world_frame(left_rel)
            log_relative_transform("left_hand_relative", left_rel)

        right_rel = process_hand_clutch(
            tracker.right,
            poses["right_hand"],
            right_grip,
            "right",
        )
        if right_rel is not None:
            # Transform relative motion to world frame
            # right_rel_world = transform_relative_to_world_frame(right_rel)
            log_relative_transform("right_hand_relative", right_rel)

        time.sleep(0.033)
