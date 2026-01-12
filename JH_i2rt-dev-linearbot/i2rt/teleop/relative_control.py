from typing import Optional, Tuple, Dict

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation

from i2rt.robots.get_robot import get_whole_robot
from i2rt.robots.motor_chain_robot import MotorChainRobot
from i2rt.robots.utils import ArmType, GripperType, save_assembled_robot_xml
from i2rt.robots.kinematics import Kinematics

from modern_robotics import RpToTrans, TransToRp


def add_world_frame_axes(viewer, axis_length: float = 0.15, axis_radius: float = 0.005):
    """Add world frame coordinate axes to the MuJoCo viewer scene.

    Args:
        viewer: MuJoCo passive viewer instance
        axis_length: Length of each axis in meters
        axis_radius: Radius of axis cylinders
    """
    scene = viewer.user_scn

    # Define axis colors (RGBA): Red=X, Green=Y, Blue=Z
    colors = [
        [1.0, 0.0, 0.0, 1.0],  # X-axis: Red
        [0.0, 1.0, 0.0, 1.0],  # Y-axis: Green
        [0.0, 0.0, 1.0, 1.0],  # Z-axis: Blue
    ]

    # Define axis directions
    directions = [
        [axis_length, 0.0, 0.0],  # X-axis
        [0.0, axis_length, 0.0],  # Y-axis
        [0.0, 0.0, axis_length],  # Z-axis
    ]

    origin = np.array([0.0, 0.0, 0.0])

    for i, (direction, color) in enumerate(zip(directions, colors)):
        if scene.ngeom >= scene.maxgeom:
            break

        end_point = origin + np.array(direction)

        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.array([axis_radius, axis_radius, axis_length / 2]),
            (origin + end_point) / 2,  # Position at midpoint
            np.eye(3).flatten(),  # Identity rotation (will be adjusted)
            np.array(color, dtype=np.float32),
        )

        # Set proper orientation for each axis
        geom = scene.geoms[scene.ngeom]
        if i == 0:  # X-axis: rotate 90 degrees around Y
            rot = Rotation.from_euler("y", 90, degrees=True).as_matrix()
        elif i == 1:  # Y-axis: rotate -90 degrees around X
            rot = Rotation.from_euler("x", -90, degrees=True).as_matrix()
        else:  # Z-axis: no rotation needed (default cylinder is along Z)
            rot = np.eye(3)

        geom.mat[:] = rot
        scene.ngeom += 1


HOME_JOINTS = np.array(
    [
        -0.03757534,
        0.00629435,
        0.01354238,
        -0.097467,
        -0.01239796,
        0.05130846,
        1.04008602,
    ]
)
INIT_JOINTS = np.array(
    [
        -0.03757534,
        1.31551843,
        0.22487984,
        1.24494545,
        -0.15011063,
        -0.09670405,
        1.04008602,
    ]
)


class SimulatedRobot:
    """A simulated robot that mimics MotorChainRobot interface using MuJoCo."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self._joint_pos = np.zeros(6)  # Only arm joints (gripper stored separately)
        self._gripper_pos = np.array([1.0])
        # self._dt = 0.01  # Simulation timestep for animation

        # Launch viewer automatically
        self._viewer = mujoco.viewer.launch_passive(
            model=self.model,
            data=self.data,
            show_left_ui=False,
            show_right_ui=False,
        )
        mujoco.mjv_defaultFreeCamera(self.model, self._viewer.cam)
        self._viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
        self._update_viewer()

    def __del__(self):
        """Close the viewer when the object is destroyed."""
        self.close()

    def get_observations(self) -> Dict[str, np.ndarray]:
        return {
            "joint_pos": self._joint_pos.copy(),
            "gripper_pos": self._gripper_pos.copy(),
        }

    def _update_viewer(self) -> None:
        """Update the viewer with current robot state."""
        if self._viewer is not None and self._viewer.is_running():
            self.data.qpos[: len(self._joint_pos)] = self._joint_pos
            # Update gripper position in qpos
            gripper_idx = len(self._joint_pos)
            if gripper_idx < self.model.nq:
                self.data.qpos[gripper_idx] = self._gripper_pos[0]
            mujoco.mj_kinematics(self.model, self.data)
            add_world_frame_axes(self._viewer)
            self._viewer.sync()

    def move_joints(
        self,
        joint_positions: np.ndarray,
        time_interval_s: float = 2.0,
    ) -> None:
        import time

        target_joints = joint_positions[:6].copy()
        target_gripper = (
            joint_positions[6] if len(joint_positions) > 6 else self._gripper_pos[0]
        )

        # If viewer is set, animate the motion
        if self._viewer is not None and self._viewer.is_running():
            start_joints = self._joint_pos.copy()
            start_gripper = self._gripper_pos[0]

            steps = time_interval_s / 0.01
            for i in range(steps + 1):
                if not self._viewer.is_running():
                    break
                alpha = i / steps if steps > 0 else 1.0
                # Linear interpolation
                self._joint_pos = start_joints + alpha * (target_joints - start_joints)
                self._gripper_pos = np.array(
                    [start_gripper + alpha * (target_gripper - start_gripper)]
                )
                self._update_viewer()
                time.sleep(0.01)
        else:
            # No viewer, set instantly
            self._joint_pos = target_joints
            self._gripper_pos = np.array([target_gripper])

    def close(self) -> None:
        """Close the viewer if it's running."""
        if hasattr(self, "_viewer") and self._viewer is not None:
            self._viewer.close()
            self._viewer = None


class RelativeControl:
    _instance: Optional["RelativeControl"] = None
    _initialized: bool = False

    def __new__(cls, *args, **kwargs) -> "RelativeControl":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        robot: MotorChainRobot,
        kinematics: Kinematics,
        site_name: str = "grasp_site",
    ):
        if RelativeControl._initialized:
            return

        self.robot = robot
        self.kinematics = kinematics
        self.site_name = site_name
        RelativeControl._initialized = True

    def __del__(self):
        print("Closing connection to robot ...")
        self.robot.close()
        print("Robot connection closed")

    def get_current_pose(self) -> np.ndarray:
        q = self.get_current_joints()
        return self.kinematics.fk(q[:6])  # FK expects only arm joints

    def get_current_joints(self) -> np.ndarray:
        obs = self.robot.get_observations()
        joint_pos = obs["joint_pos"]
        # Real robot returns 6 arm joints separately from gripper
        # Sim robot returns all 7 in joint_pos
        # Kinematics expects 7 joints (arm + gripper)
        if len(joint_pos) == 6 and "gripper_pos" in obs:
            joint_pos = np.append(joint_pos, obs["gripper_pos"])
        return joint_pos

    def get_current_gripper_pos(self) -> np.ndarray:
        obs = self.robot.get_observations()
        return obs["gripper_pos"][0]

    def move_gripper(
        self,
        gripper_ratio: float,
        time_interval_s: float = 2.0,
    ) -> bool:
        current_joints = self.get_current_joints()
        full_joint_positions = np.append(current_joints[:6], gripper_ratio)
        self.robot.move_joints(full_joint_positions, time_interval_s)

        return True

    def move_relative(
        self,
        delta_T: np.ndarray,
        gripper_ratio: float = None,
        time_interval_s: float = 2.0,
        verbose: bool = False,
    ) -> bool:
        if verbose:
            R, p = TransToRp(delta_T)
            r = Rotation.from_matrix(R)
            euler_angles = r.as_euler("xyz", degrees=False)
            roll, pitch, yaw = euler_angles
            print(roll, pitch, yaw)
            print(p)

        success, target_joints = self._compute_goal(delta_T, verbose)

        if success:
            gripper_pos = (
                self.get_current_gripper_pos()
                if gripper_ratio is None
                else gripper_ratio
            )
            full_joint_positions = np.append(target_joints[:6], gripper_pos)

            self.robot.move_joints(
                full_joint_positions,
                time_interval_s=time_interval_s,
            )
            return True
        else:
            return False

    def move_home(
        self,
        home_joints: Optional[np.ndarray] = None,
        time_interval_s: float = 2.0,
    ) -> bool:
        if home_joints is None:
            home_joints = np.array(
                [
                    -0.03757534,
                    0.00629435,
                    0.01354238,
                    -0.097467,
                    -0.01239796,
                    0.05130846,
                    1.04008602,
                ]
            )

        self.robot.move_joints(home_joints, time_interval_s=time_interval_s)
        return True

    def move_to_init(
        self,
        init_joints: Optional[np.ndarray] = None,
        time_interval_s: float = 2.0,
    ) -> bool:
        if init_joints is None:
            init_joints = INIT_JOINTS

        self.robot.move_joints(init_joints, time_interval_s=time_interval_s)
        return True

    def _compute_goal(
        self,
        delta_T: np.ndarray,
        verbose: bool = False,
    ) -> Tuple[bool, np.ndarray]:
        delta_T = np.asarray(delta_T)
        if delta_T.shape != (4, 4):
            raise ValueError(f"delta_T must have shape (4, 4), got {delta_T.shape}")

        current_joints = self.get_current_joints()
        current_pose = self.kinematics.fk(
            current_joints[:6]
        )  # FK expects only arm joints
        target_pose = current_pose @ delta_T

        success, target_joints = self.kinematics.ik(
            target_pose=target_pose,
            site_name=self.site_name,
            init_q=current_joints[:6],  # IK expects only arm joints
            verbose=verbose,
            pos_threshold=0.001,
            ori_threshold=0.001,
        )

        return success, target_joints


def create_relative_controller(
    channel: str = "can0",
    arm_type: ArmType = ArmType.YAM,
    gripper_type: GripperType = GripperType.CRANK_4310,
    site_name: str = "grasp_site",
) -> tuple[MotorChainRobot, RelativeControl]:
    robot = get_whole_robot(
        channel=channel,
        arm_type=arm_type,
        gripper_type=gripper_type,
    )

    model_path = save_assembled_robot_xml(gripper_type, arm_type)
    kinematics = Kinematics(xml_path=model_path, site_name=site_name)
    controller = RelativeControl(robot, kinematics, site_name)

    return robot, controller


def create_relative_sim_controller(
    arm_type: ArmType = ArmType.YAM,
    gripper_type: GripperType = GripperType.CRANK_4310,
    site_name: str = "grasp_site",
) -> tuple[SimulatedRobot, RelativeControl]:
    model_path = save_assembled_robot_xml(gripper_type, arm_type)
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    robot = SimulatedRobot(model, data)
    kinematics = Kinematics(xml_path=model_path, site_name=site_name)

    # Reset singleton state to allow new instance
    RelativeControl._instance = None
    RelativeControl._initialized = False

    controller = RelativeControl(robot, kinematics, site_name)
    return robot, controller


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(
        description="Relative control for robot arm end-effector motion."
    )
    parser.add_argument(
        "--dx",
        type=float,
        default=0.0,
        help="X displacement in meters",
    )
    parser.add_argument(
        "--dy",
        type=float,
        default=0.0,
        help="Y displacement in meters",
    )
    parser.add_argument(
        "--dz",
        type=float,
        default=0.0,
        help="Z displacement in meters",
    )
    parser.add_argument(
        "--dr",
        type=float,
        default=0.0,
        help="Roll rotation in degrees",
    )
    parser.add_argument(
        "--dp",
        type=float,
        default=0.0,
        help="Pitch rotation in degrees",
    )
    parser.add_argument(
        "--dyaw",
        type=float,
        default=0.0,
        help="Yaw rotation in degrees",
    )
    parser.add_argument(
        "--gripper",
        type=float,
        default=1.0,
        help="Gripper open offset ratio",
    )
    parser.add_argument("--channel", type=str, default="can0", help="CAN channel")
    parser.add_argument(
        "--time",
        type=float,
        default=2.0,
        help="Motion time in seconds",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--sim",
        action="store_true",
        help="Run in MuJoCo simulation mode with visualization",
    )
    args = parser.parse_args()

    delta_p = np.array([args.dx, args.dy, args.dz])
    delta_R = Rotation.from_euler(
        "xyz",
        [args.dr, args.dp, args.dyaw],
        degrees=True,
    ).as_matrix()
    delta_T = RpToTrans(delta_R, delta_p)

    print(delta_T)

    # Only difference: which factory function to call
    if args.sim:
        robot, controller = create_relative_sim_controller(
            arm_type=ArmType.YAM,
            gripper_type=GripperType.CRANK_4310,
        )
    else:
        robot, controller = create_relative_controller(
            channel=args.channel,
            arm_type=ArmType.YAM,
            gripper_type=GripperType.CRANK_4310,
        )

    print("Robot initialized")
    print("Current pose:")
    print(controller.get_current_pose())

    print("\nMoving to init position...")
    controller.move_to_init(time_interval_s=args.time)
    print("Reached init position.")

    print("Pose at init position:")
    print(controller.get_current_pose())

    print(f"\nExecuting relative motion:")
    print(
        f"  Position: dx={args.dx*1000:.2f}mm, dy={args.dy*1000:.2f}mm, dz={args.dz*1000:.2f}mm"
    )
    print(
        f"  Rotation: dr={args.dr:.2f}deg, dp={args.dp:.2f}deg, dyaw={args.dyaw:.2f}deg"
    )

    success = controller.move_relative(
        delta_T=delta_T,
        gripper_ratio=args.gripper,
        time_interval_s=args.time,
        verbose=args.verbose,
    )

    if success:
        print("Motion executed successfully!")
    else:
        print("IK failed, motion not executed.")

    print("\nMoving to home position...")
    controller.move_home(time_interval_s=args.time)
    print("Reached home position.")

    robot.close()
