from pydrake.all import (
    LeafSystem,
    AbstractValue,
    RigidTransform,
    PublishEvent,
    MultibodyPlant,
    PiecewisePose,
    JacobianWrtVariable,
    PiecewisePose,
    RotationMatrix,
)
import numpy as np


class PandaGraspTrajectoryPlanner(LeafSystem):
    def __init__(self, plant: MultibodyPlant):
        super().__init__()
        self.plant = plant
        self.context = plant.CreateDefaultContext()
        self.panda_link8_body = plant.GetBodyByName("panda_link8")

        self.grasp_input_port = self.DeclareAbstractInputPort(
            "grasp_pose", AbstractValue.Make(RigidTransform())
        )
        self.panda_arm_trajectory_output_port = self.DeclareAbstractOutputPort(
            "panda_arm_trajectory",
            lambda: AbstractValue.Make(PiecewisePose()),
            self.CalcPandaArmTrajectory,
        )

    def CalcPandaArmTrajectory(self, context, output):
        X_WH_Init = self.plant.EvalBodyPoseInWorld(self.context, self.panda_link8_body)
        X_WG = self.grasp_input_port.Eval(context)

        X_GPregrasp = RigidTransform(
            R=RotationMatrix(
                np.array(
                    [
                        [0, 1, 0],  # X becomes Z
                        [0, 0, 1],  # Y becomes X
                        [1, 0, 0],  # Z becomes Y
                    ]
                )
            ),
            p=[0, -0.25, 0.25],
        )
        X_WPregrasp = X_WG.multiply(X_GPregrasp)
        # goal_pose = RigidTransform(
        #     p=X_WH_Init.translation() + np.array([0, 0, -0.1]), R=X_WH_Init.rotation()
        # )
        X_WPregrasp = RigidTransform(
            R=X_GPregrasp.rotation(),
            p=X_WPregrasp.translation(),
        )
   
        trajectory = PiecewisePose.MakeLinear(
            times=[0, 0.5],
            poses=[X_WH_Init, X_WPregrasp],
        )

        # spatial velocity
        trajectory_VG = trajectory.MakeDerivative()

        output.set_value(trajectory_VG)


class DebugTrajectoryVisualizer(LeafSystem):
    pass
