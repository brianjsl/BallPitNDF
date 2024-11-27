from pydrake.all import (
    LeafSystem,
    AbstractValue,
    RigidTransform,
    PublishEvent,
    MultibodyPlant,
    PiecewisePose,
    JacobianWrtVariable,
    RotationMatrix,
    Meshcat
)
import numpy as np
from debug.visualize_utils import draw_grasp_candidate, draw_query_pts


class PandaGraspTrajectoryPlanner(LeafSystem):
    def __init__(self, plant: MultibodyPlant, meshcat: Meshcat):
        super().__init__()
        self.plant = plant
        self.context = plant.CreateDefaultContext()
        self.panda_link8_body = plant.GetBodyByName("panda_link8")

        self.grasp_input_port = self.DeclareAbstractInputPort(
            "grasp_pose", AbstractValue.Make((RigidTransform(), np.ndarray(0)))
        )
        self.panda_arm_trajectory_output_port = self.DeclareAbstractOutputPort(
            "panda_arm_trajectory",
            lambda: AbstractValue.Make(PiecewisePose()),
            self.CalcPandaArmTrajectory,
        )
        self.meshcat = meshcat

    def CalcPandaArmTrajectory(self, context, output):
        X_WH_Init = self.plant.EvalBodyPoseInWorld(self.context, self.panda_link8_body)
        X_WG, final_query_pts = self.grasp_input_port.Eval(context)
        # draw_grasp_candidate(self.meshcat, X_WG)
        draw_query_pts(self.meshcat, final_query_pts)


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