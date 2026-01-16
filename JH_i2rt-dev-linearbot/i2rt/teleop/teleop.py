from i2rt.robots.utils import ArmType, GripperType

from i2rt.teleop.relative_control import (
    create_relative_controller,
    create_relative_sim_controller,
)
from i2rt.teleop.vr_get_pose import (
    ClutchTracker,
    create_quest_reader,
    process_hand_clutch,
    init_rerun,
)

import argparse
import numpy as np
import time
from modern_robotics import TransInv

# Transform from controller frame (x right, y forward, z up)
# to gripper frame (x down, y left, z forward)
T_CONTROLLER_TO_GRIPPER = np.array(
    [
        [0, 0, -1, 0],  # gripper x = -controller z
        [-1, 0, 0, 0],  # gripper y = -controller x
        [0, 1, 0, 0],  # gripper z = controller y
        [0, 0, 0, 1],
    ]
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quest3 teleoperator")
    parser.add_argument(
        "--sim",
        action="store_true",
        help="Whether to use mujoco simulator",
    )

    args = parser.parse_args()

    init_rerun(app_name="Quest3 Teleoperation")
    if args.sim:
        robot, controller = create_relative_sim_controller(
            arm_type=ArmType.YAM,
            gripper_type=GripperType.CRANK_4310,
            site_name="grasp_site",
        )
    else:
        robot, controller = create_relative_controller(
            channel="can0",
            arm_type=ArmType.YAM,
            gripper_type=GripperType.CRANK_4310,
        )

    quest_reader = create_quest_reader(ip="127.0.0.1", port=12345)
    tracker = ClutchTracker()

    last_time = time.time()
    try:
        while True:
            current_time = time.time()
            dt = current_time - last_time
            if dt > 0:
                frequency = 1.0 / dt
                print(f"Loop frequency: {frequency:.1f} Hz")
            last_time = current_time
            data = quest_reader.get_data()
            if not data:
                time.sleep(0.02)
                continue

            left_pose = data.get("Left Hand", np.eye(4))
            right_pose = data.get("Right Hand", np.eye(4))
            head_pose = data.get("Head", np.eye(4))

            left_grip = data.get("Left Grip", 0.0)
            right_grip = data.get("Right Grip", 0.0)

            left_trigger = data.get("Left Trigger", 0.0)
            right_trigger = data.get("Right Trigger", 0.0)

            right_T_rel = process_hand_clutch(
                tracker.right,
                right_pose,
                right_grip,
                "right",
            )

            if right_T_rel is not None:
                # Convert relative motion from controller frame to gripper frame
                right_T_rel_gripper = (
                    T_CONTROLLER_TO_GRIPPER
                    @ right_T_rel
                    @ TransInv(T_CONTROLLER_TO_GRIPPER)
                )
                controller.move_relative(
                    delta_T=right_T_rel_gripper,
                    gripper_ratio=right_trigger,
                    time_interval_s=0.02,
                    verbose=True,
                )

            # print(right_trigger)
            else:
                controller.move_gripper(
                    gripper_ratio=right_trigger,
                    time_interval_s=0.02,
                )

    except KeyboardInterrupt:
        print("Ctrl-C Pressed, move robot to home ...")
        controller.move_home(time_interval_s=2.0)
        print("Robot homed")

    finally:
        robot.close()
