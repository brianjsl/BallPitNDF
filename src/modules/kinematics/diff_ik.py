from pydrake.all import (
    DiagramBuilder,
    MultibodyPlant,
    Frame,
    DifferentialInverseKinematicsIntegrator,
    DifferentialInverseKinematicsParameters,
    ModelInstanceIndex
)
import numpy as np
from typing import Optional

def AddPandaDifferentialIK(builder: DiagramBuilder, plant: MultibodyPlant, frame: Optional[Frame] =None
    , panda_arm: ModelInstanceIndex = None
    ) -> DifferentialInverseKinematicsIntegrator:
    params = DifferentialInverseKinematicsParameters(plant.num_positions(panda_arm),
                                                     plant.num_velocities(panda_arm))
    time_step = plant.time_step()
    q0 = plant.GetPositions(plant.CreateDefaultContext(), panda_arm)
    params.set_nominal_joint_position(q0)
    params.set_end_effector_angular_speed_limit(2)
    params.set_end_effector_translational_velocity_limits([-2, -2, -2],
                                                          [2, 2, 2])

    # Decrease velocity to prevent oscilations in controller.
    # Panda has 7 (arm) joints (exclude the hand)
    panda_velocity_limits = np.array([1.4, 1.4, 1.7, 1.3, 2.2, 2.3, 2.3])
    params.set_joint_velocity_limits(
        (-panda_velocity_limits, panda_velocity_limits))
    params.set_joint_centering_gain(10 * np.eye(7))
    if frame is None:
        frame = plant.GetFrameByName("panda_link8")
    differential_ik = builder.AddSystem(
        DifferentialInverseKinematicsIntegrator(
            plant,
            frame,
            time_step,
            params,
            log_only_when_result_state_changes=True))
    return differential_ik