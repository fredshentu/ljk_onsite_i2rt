from i2rt.robots.utils import ArmType, GripperType

from i2rt.teleop.relative_control import (
    create_relative_controller,
    create_simulated_controller,
)
from i2rt.teleop.vr_get_pose import (
    ClutchTracker,
    create_quest_reader,
    process_hand_clutch,
    init_rerun,
)


import numpy as np
import time

T_QUEST_WORLD = np.array([[]])

if __name__ == "__main__":
    init_rerun(app_name="Quest3 Teleoperation")
    # robot, controller = create_relative_controller(
    #     channel="can0",
    #     arm_type=ArmType.YAM,
    #     gripper_type=GripperType.CRANK_4310,
    # )
    robot, controller = create_simulated_controller(
        arm_type=ArmType.YAM,
        gripper_type=GripperType.CRANK_4310,
        site_name="grasp_site",
    )

    quest_reader = create_quest_reader(ip="127.0.0.1", port=12345)
    tracker = ClutchTracker()

    try:
        while True:
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
                controller.move_relative(
                    delta_T=right_T_rel,
                    gripper_ratio=right_trigger,
                    time_interval_s=0.03,
                    verbose=False,
                )

            # print(right_trigger)
            else:
                controller.move_gripper(
                    gripper_ratio=right_trigger,
                    time_interval_s=0.03,
                )

            time.sleep(0.033)

    except KeyboardInterrupt:
        print("Ctrl-C Pressed, move robot to home ...")
        controller.move_home(time_interval_s=2.0)
        print("Robot homed")

    finally:
        robot.close()
