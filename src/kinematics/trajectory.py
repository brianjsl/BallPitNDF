from pydrake.all import (
    LeafSystem,
    AbstractValue,
    RigidTransform,
    PublishEvent,
    MultibodyPlant,
    PiecewisePose,
    JacobianWrtVariable,
    PiecewisePose,
    TrajectorySource,
    DiagramBuilder,
    Integrator,
)
import numpy as np

from .diff_ik import PandaPseudoDiffIKController
from .planner import PandaGraspTrajectoryPlanner


def CreatePandaTrajectoryController(plant: MultibodyPlant, initial_panda_arm_position):
    builder = DiagramBuilder()

    trajectory_evaluator = builder.AddNamedSystem(
        "trajectory_evaluator", TrajectoryEvaluator()
    )

    builder.ExportInput(
        trajectory_evaluator.get_input_port(0),
        "panda_arm_trajectory",
    )

    # diff ik controller
    diff_ik_controller = builder.AddNamedSystem(
        "diff_ik_controller", PandaPseudoDiffIKController(plant)
    )

    builder.ExportInput(diff_ik_controller.GetInputPort("q"), "panda_arm.position")
    builder.Connect(
        trajectory_evaluator.get_output_port(0),
        diff_ik_controller.GetInputPort("spatial_velocity"),
    )

    # panda arm integrator
    integrator = builder.AddNamedSystem("panda_arm_integrator", Integrator(7))
    builder.Connect(
        diff_ik_controller.GetOutputPort("velocity_commanded"),
        integrator.get_input_port(0),
    )
    builder.ExportOutput(
        integrator.get_output_port(0),
        "panda_arm.position_commanded",
    )

    # panda_arm = plant.GetModelInstanceByName("panda_arm")
    # integrator.set_integral_value(
    #     integrator.CreateDefaultContext(),
    #     initial_panda_arm_position,
    #     # plant.GetPositions(plant.GetMyContextFromRoot(context), panda_arm),
    # )

    diagram = builder.Build()
    diagram.set_name("PandaTrajectoryController")

    return diagram


class PandaTrajectoryEvaluator(LeafSystem):
    def __init__(self):
        super().__init__()
        self.trajectory_input = self.DeclareAbstractInputPort(
            "trajectory", AbstractValue.Make(PiecewisePose())
        )

        self.panda_arm_q = self.DeclareVectorOutputPort(
            "panda_arm_q", 7, self.CalcPandaArmQ
        )
        self.panda_hand_q = self.DeclareVectorOutputPort(
            "panda_hand_q", 2, self.CalcPandaHandQ
        )

    def CalcPandaArmQ(self, context, output):
        t = context.get_time()
        trajectory = self.trajectory_input.Eval(context)

        if t < trajectory.end_time():
            q = trajectory.value(t)[:7]
        else:
            q = np.zeros(7)

        output.set_value(q)

    def CalcPandaHandQ(self, context, output):
        t = context.get_time()
        trajectory = self.trajectory_input.Eval(context)

        if t < trajectory.end_time():
            q = trajectory.value(t)[-2:]
        else:
            q = np.zeros(2)

        output.set_value(q)
