import argparse
import logging
import os
import time
import xml.etree.ElementTree as ET
from functools import partial
from typing import Any, Callable, Optional, Tuple

import numpy as np

from i2rt.motor_drivers.dm_driver import (
    CanInterface,
    DMChainCanInterface,
    EncoderChain,
    PassiveEncoderReader,
    ReceiveMode,
)
from i2rt.robots.motor_chain_robot import MotorChainRobot
from i2rt.robots.utils import (
    I2RT_ROOT,
    ArmType,
    GripperType,
    save_assembled_robot_xml,
)


def _find_gripper_body(element: ET.Element) -> Optional[ET.Element]:
    """Recursively find the deepest body element (gripper) in the kinematic chain."""
    bodies = element.findall("body")
    if not bodies:
        return None
    for body in bodies:
        deeper = _find_gripper_body(body)
        if deeper is not None:
            return deeper
        # Check if this is a leaf body (no child bodies) - likely the gripper
        if not body.findall("body"):
            return body
    return bodies[-1]


def modify_gripper_inertial(
    xml_path: str,
    gripper_mass: Optional[float] = None,
    gripper_inertia: Optional[Tuple[float, float, float]] = None,
) -> str:
    """
    Modify the gripper's mass and inertia in the robot XML without changing the original file.

    Args:
        xml_path: Path to the assembled robot XML file.
        gripper_mass: Optional new mass value for the gripper.
        gripper_inertia: Optional new diagonal inertia (ixx, iyy, izz) for the gripper.

    Returns:
        Path to the modified XML file (in /tmp/).
    """
    if gripper_mass is None and gripper_inertia is None:
        return xml_path

    tree = ET.parse(xml_path)
    root = tree.getroot()

    worldbody = root.find("worldbody")
    if worldbody is None:
        logging.warning("No worldbody found in XML, returning original path")
        return xml_path

    gripper_body = _find_gripper_body(worldbody)
    if gripper_body is None:
        logging.warning("No gripper body found in XML, returning original path")
        return xml_path

    inertial = gripper_body.find("inertial")
    if inertial is None:
        logging.warning("No inertial element found in gripper body, returning original path")
        return xml_path

    if gripper_mass is not None:
        inertial.set("mass", str(gripper_mass))
        logging.info(f"Modified gripper mass to: {gripper_mass}")

    if gripper_inertia is not None:
        inertia_str = f"{gripper_inertia[0]} {gripper_inertia[1]} {gripper_inertia[2]}"
        inertial.set("diaginertia", inertia_str)
        logging.info(f"Modified gripper inertia to: {inertia_str}")

    # Save to a new file with modified suffix
    base_dir = os.path.dirname(xml_path)
    base_name = os.path.splitext(os.path.basename(xml_path))[0]
    modified_path = os.path.join(base_dir, f"{base_name}_modified.xml")

    ET.indent(root, space="  ")
    tree.write(modified_path, encoding="unicode", xml_declaration=True)

    return modified_path


def get_encoder_chain(can_interface: CanInterface) -> EncoderChain:
    passive_encoder_reader = PassiveEncoderReader(can_interface)
    return EncoderChain([0x50E], passive_encoder_reader)


def get_big_yam_robot(
    channel: str = "can0",
    gripper_type: GripperType = GripperType.LINEAR_4310,
    zero_gravity_mode: bool = True,
    joint_state_saver_factory: Optional[Callable[[str], Any]] = None,
    set_realtime_and_pin_callback: Optional[Callable[[int], None]] = None,
) -> MotorChainRobot:
    xml_path = os.path.join(I2RT_ROOT, "robot_models/big_yam/big_yam.xml")
    motor_list = [
        [0x01, "DM6248"],
        [0x02, "DM6248"],
        [0x03, "DM4340"],
        [0x04, "DM4340"],
        [0x05, "DM4310"],
        [0x06, "DM4310"],
        [0x07, "DM4310"],
    ]
    motor_offsets = [0, 0, 0, 0, 0, 0, 0]
    motor_directions = [1, -1, 1, 1, 1, 1, 1]
    kp = np.array([80, 80, 80, 40, 40, 10, 10])
    kd = np.array([5, 5, 5, 3, 3, 3, 0.5])

    motor_chain = DMChainCanInterface(
        motor_list,
        motor_offsets,
        motor_directions,
        channel,
        motor_chain_name="yam_real",
        receive_mode=ReceiveMode.p16,
    )
    motor_states = motor_chain.read_states()
    logging.info(f"YAM initial motor_states: {motor_states}")
    get_robot = partial(
        MotorChainRobot,
        motor_chain=motor_chain,
        xml_path=xml_path,
        use_gravity_comp=True,
        gravity_comp_factor=1.0,
        kp=kp,
        kd=kd,
        zero_gravity_mode=zero_gravity_mode,
        joint_state_saver_factory=joint_state_saver_factory,
        set_realtime_and_pin_callback=set_realtime_and_pin_callback,
        gripper_index=6,
        gripper_limits=gripper_type.get_gripper_limits(),
        gripper_type=gripper_type,
        limit_gripper_force=50.0,
    )
    return get_robot()


def get_yam_robot(
    channel: str = "can0",
    gripper_type: GripperType = GripperType.CRANK_4310,
    zero_gravity_mode: bool = True,
    joint_state_saver_factory: Optional[Callable[[str], Any]] = None,
    set_realtime_and_pin_callback: Optional[Callable[[int], None]] = None,
) -> MotorChainRobot:
    with_gripper = True
    with_teaching_handle = False
    if gripper_type == GripperType.YAM_TEACHING_HANDLE:
        with_gripper = False
        with_teaching_handle = True
    if gripper_type == GripperType.NO_GRIPPER:
        with_gripper = False
        with_teaching_handle = False

    model_path = gripper_type.get_xml_path()
    motor_list = [
        [0x01, "DM4340"],
        [0x02, "DM4340"],
        [0x03, "DM4340"],
        [0x04, "DM4310"],
        [0x05, "DM4310"],
        [0x06, "DM4310"],
    ]
    motor_offsets = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    motor_directions = [1, 1, 1, 1, 1, 1]
    joint_limits = np.array(
        [
            [-2.617, 3.13],
            [0, 3.65],
            [0.0, 3.13],
            [-1.57, 1.57],
            [-1.57, 1.57],
            [-2.09, 2.09],
        ]
    )
    joint_limits[:, 0] += -0.15  # add some buffer to the joint limits
    joint_limits[:, 1] += 0.15

    kp = np.array([80, 80, 80, 40, 10, 10])
    kd = np.array([5, 5, 5, 1.5, 1.5, 1.5])
    if with_gripper:
        motor_type = gripper_type.get_motor_type()
        gripper_kp, gripper_kd = gripper_type.get_motor_kp_kd()
        assert motor_type != ""
        logging.info(
            f"adding gripper motor with type: {motor_type}, gripper_kp: {gripper_kp}, gripper_kd: {gripper_kd}"
        )
        motor_list.append([0x07, motor_type])
        motor_offsets.append(0.0)
        motor_directions.append(1)
        kp = np.concatenate([kp, np.array([gripper_kp])])
        kd = np.concatenate([kd, np.array([gripper_kd])])

    motor_chain = DMChainCanInterface(
        motor_list,
        motor_offsets,
        motor_directions,
        channel,
        motor_chain_name="yam_real",
        receive_mode=ReceiveMode.p16,
        get_same_bus_device_driver=get_encoder_chain if with_teaching_handle else None,
        use_buffered_reader=False,
    )
    motor_states = motor_chain.read_states()
    logging.info(f"YAM initial motor_states: {motor_states}")
    get_robot = partial(
        MotorChainRobot,
        motor_chain=motor_chain,
        xml_path=model_path,
        use_gravity_comp=True,
        gravity_comp_factor=1.3,
        joint_limits=joint_limits,
        kp=kp,
        kd=kd,
        zero_gravity_mode=zero_gravity_mode,
        joint_state_saver_factory=joint_state_saver_factory,
        set_realtime_and_pin_callback=set_realtime_and_pin_callback,
    )

    if with_gripper:
        return get_robot(
            gripper_index=6,
            gripper_limits=gripper_type.get_gripper_limits(),
            gripper_type=gripper_type,
            limit_gripper_force=50.0,
        )
    else:
        return get_robot()


def get_whole_robot(
    channel="can0",
    arm_type: ArmType = ArmType.YAM,
    gripper_type: GripperType = GripperType.CRANK_4310,
    zero_gravity_mode: bool = True,
    joint_state_saver_factory: Optional[Callable[[str], Any]] = None,
    set_realtime_and_pin_callback: Optional[Callable[[int], None]] = None,
    gripper_mass: Optional[float] = None,
    gripper_inertia: Optional[Tuple[float, float, float]] = None,
) -> MotorChainRobot:
    with_gripper = True
    with_teaching_handle = False
    if gripper_type == GripperType.YAM_TEACHING_HANDLE:
        with_gripper = False
        with_teaching_handle = True
    if gripper_type == GripperType.NO_GRIPPER:
        with_gripper = False
        with_teaching_handle = False
        
    arm_base_path = arm_type.get_arm_xml_path()
    model_path = save_assembled_robot_xml(gripper_type, arm_base_path)
    model_path = modify_gripper_inertial(model_path, gripper_mass, gripper_inertia)
    
    motor_list = [
        [0x01, "DM4340"],
        [0x02, "DM4340"],
        [0x03, "DM4340"],
        [0x04, "DM4310"],
        [0x05, "DM4310"],
        [0x06, "DM4310"],
    ]
    motor_offsets = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    motor_directions = [1, 1, 1, 1, 1, 1]
    joint_limits = np.array(
        [
            [-2.617, 3.13],
            [0, 3.65],
            [0.0, 3.13],
            [-1.57, 1.57],
            [-1.57, 1.57],
            [-2.09, 2.09],
        ]
    )
    joint_limits[:, 0] += -0.15  # add some buffer to the joint limits
    joint_limits[:, 1] += 0.15

    kp = np.array([80, 80, 80, 40, 10, 10])
    kd = np.array([5, 5, 5, 1.5, 1.5, 1.5])
    if with_gripper:
        motor_type = gripper_type.get_motor_type()
        gripper_kp, gripper_kd = gripper_type.get_motor_kp_kd()
        assert motor_type != ""
        logging.info(
            f"adding gripper motor with type: {motor_type}, gripper_kp: {gripper_kp}, gripper_kd: {gripper_kd}"
        )
        motor_list.append([0x07, motor_type])
        motor_offsets.append(0.0)
        motor_directions.append(1)
        kp = np.concatenate([kp, np.array([gripper_kp])])
        kd = np.concatenate([kd, np.array([gripper_kd])])
        
    motor_chain = DMChainCanInterface(
        motor_list=motor_list,
        motor_offset=motor_offsets,
        motor_direction=motor_directions,
        channel=channel,
        motor_chain_name=f"{arm_type.name}_{gripper_type.name}",
        receive_mode=ReceiveMode.p16,
        get_same_bus_device_driver=get_encoder_chain if with_teaching_handle else None,
        use_buffered_reader=False
    )

    motor_states = motor_chain.read_states()
    logging.info(f"YAM initial motor_states: {motor_states}")
    
    get_robot = partial(
        MotorChainRobot,
        motor_chain=motor_chain,
        xml_path=model_path,
        use_gravity_comp=True,
        gravity_comp_factor=1.3,
        joint_limits=joint_limits,
        kp=kp,
        kd=kd,
        zero_gravity_mode=zero_gravity_mode,
        joint_state_saver_factory=joint_state_saver_factory,
        set_realtime_and_pin_callback=set_realtime_and_pin_callback,
    )
    
    if with_gripper:
        return get_robot(
            gripper_index=6,
            gripper_limits=gripper_type.get_gripper_limits(),
            gripper_type=gripper_type,
            limit_gripper_force=50.0,
        )
    else:
        return get_robot()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get robot with specified arm and gripper type")
    parser.add_argument(
        "--arm",
        type=str,
        default="yam",
        choices=ArmType.available_arms(),
        help=f"Arm type to use. Available: {ArmType.available_arms()}",
    )
    parser.add_argument(
        "--gripper",
        type=str,
        default="crank_4310",
        choices=GripperType.available_grippers(),
        help=f"Gripper type to use. Available: {GripperType.available_grippers()}",
    )
    parser.add_argument(
        "--channel",
        type=str,
        default="can0",
        help="CAN channel to use (default: can0)",
    )
    args = parser.parse_args()

    arm_type = ArmType.from_string_name(args.arm)
    gripper_type = GripperType.from_string_name(args.gripper)

    print(f"Starting robot with arm={arm_type.value}, gripper={gripper_type.value}, channel={args.channel}")
    robot = get_whole_robot(channel=args.channel, arm_type=arm_type, gripper_type=gripper_type)
    while True:
        time.sleep(1)
        print(robot.get_observations())
