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
        self.panda_hand_body = plant.GetBodyByName("panda_hand")

        self.grasp_input_port = self.DeclareAbstractInputPort(
            "grasp_pose", AbstractValue.Make(RigidTransform())
        )
        self.panda_arm_trajectory_output_port = self.DeclareAbstractOutputPort(
            "panda_arm_trajectory",
            lambda: AbstractValue.Make(PiecewisePose()),
            self.CalcPandaArmTrajectory,
        )
        self._traj = None

    def CalcPandaArmTrajectory(self, context, output):
        if self._traj is not None:
            output.set_value(self._traj)
            return

        # X_WH_Init = self.plant.EvalBodyPoseInWorld(self.context, self.panda_link8_body)
        X_WH_Init = self.plant.EvalBodyPoseInWorld(self.context, self.panda_hand_body)
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
            p=[0, -0.2, 0],
            # p=[0, -0.25, 0.25],
        )
        X_WPregrasp = X_WG.multiply(X_GPregrasp)
        # X_WPregrasp = RigidTransform(
        #     R=X_WH_Init.rotation(),
        #     p=X_WH_Init.translation() - np.array([0, 0, 0.05]),
        #     # p=X_WH_Init.translation() - np.array([0, 0, 0.05]),
        # )

        trajectory = PiecewisePose.MakeLinear(
            times=[0, 1.0],
            poses=[X_WH_Init, X_WPregrasp],
        )

        # spatial velocity
        trajectory_VG = trajectory.MakeDerivative()
        self._traj = trajectory_VG

        output.set_value(trajectory_VG)


class DebugTrajectoryVisualizer(LeafSystem):
    pass
