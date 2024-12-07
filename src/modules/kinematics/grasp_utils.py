import numpy as np
from pydrake.all import (AngleAxis, PiecewisePolynomial, PiecewisePose,
                         RigidTransform, RotationMatrix)
from typing import (Tuple, Dict)

def MakeGraspFrames(initial_pose: RigidTransform, grasp_pose: RigidTransform, clearance_pose: RigidTransform,
                    t0: float, returning: False) -> Dict[str, Tuple[float,RigidTransform, bool]]:
    """
    Generates a sequence of keypoint poses for a grasp trajectory.
    Inspired by https://github.com/RussTedrake/manipulation/blob/187576150412cd4f7bb1aac540f29d6cfdc76600/manipulation/pick.py#L7:
    
    Trajectory pipelline:
    initial -> prepare -> pregrasp -> grasp_start -> grasp_end -> stall -> clearance

    Args:
    initial_pose: The initial pose of the object X_Ginitial
    grasp_pose: The grasp pose of the object X_Ggrasp (not pose of end effector but pose of gripper)
    clearance_pose: The final clearance pose of the object X_Gclearance 
    t0: Time when at initial pose 
    returning: Whether the object is being returned to its original position (the gripper becomes reversed)

    Returns:
    Dict that matches keypoints to (time, pose, is_open) where is_open represents whether the gripper is opened or closed
    """

    frames = {}

    X_G_grasp_pregrasp = RigidTransform([0, 0, -0.15]) # For panda z-axis is the normal

    pregrasp_pose = grasp_pose @ X_G_grasp_pregrasp

    # Initial to prepare: interpolate halfway orientation by halving the angle
    X_GinitialGpregrasp = initial_pose.inverse() @ pregrasp_pose
    angle_axis = X_GinitialGpregrasp.rotation().ToAngleAxis()
    X_GinitialGprepare = RigidTransform(
        AngleAxis(angle = angle_axis.angle() / 2.0, axis=angle_axis.axis()),
        X_GinitialGpregrasp.translation() / 2.0
    )
    prepare_pose = initial_pose @ X_GinitialGprepare 

    prepare_time = 4.0 * np.linalg.norm(X_GinitialGprepare.translation())
    clearance_time = 5.0 * np.linalg.norm(pregrasp_pose.translation() - clearance_pose.translation())

    postgrasp_pose = pregrasp_pose

    frames = {
        'initial': (t0, initial_pose, not returning),
        'prepare': (t0 + prepare_time, prepare_pose, not returning),
        'pregrasp': (t0 + prepare_time * 2, pregrasp_pose, not returning),
        'grasp_start': (t0 + prepare_time * 2 + 1.5, grasp_pose, not returning),
        'grasp_end': (t0 + prepare_time * 2 + 3.5, grasp_pose, returning),
        'postgrasp': (t0 + prepare_time * 2 + 5.5, postgrasp_pose, returning),
        'stall': (t0 + prepare_time * 2 + 6.5, postgrasp_pose, returning),
        'clearance': (t0 + prepare_time * 2 + 6.5 + clearance_time, clearance_pose, returning)
    }

    return frames
    
def MakeGraspTrajectories(frames: Dict[str, Tuple[float,RigidTransform]]) -> Tuple[PiecewisePose, PiecewisePolynomial]:
    """
    Constructs a panda arm position and hand trajectory from plans generated from MakeGraspFrames
    """
    sample_times = []
    poses = []

    opened = np.array([-0.1, 0.1])
    closed = np.array([0.0, 0.0])
    hand_samples = []

    keypoint_names = ['initial', 'prepare', 'pregrasp', 'grasp_start', 'grasp_end', 'postgrasp', 'stall', 'clearance']

    for keypoint in keypoint_names:
        time, pose, is_open = frames[keypoint]
        sample_times.append(time)
        poses.append(pose)

        hand_samples.append(opened if is_open else closed)

    return PiecewisePose.MakeLinear(sample_times, poses), PiecewisePolynomial.FirstOrderHold(sample_times, np.array(hand_samples).T)