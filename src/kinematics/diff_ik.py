from pydrake.all import (
    LeafSystem,
    AbstractValue,
    RigidTransform,
    PublishEvent,
    MultibodyPlant,
    PiecewisePose,
    JacobianWrtVariable,
    MathematicalProgram,
    SnoptSolver,
)
import numpy as np


class PandaPseudoDiffIKController(LeafSystem):
    def __init__(self, plant: MultibodyPlant, panda_arm_name: str = "panda_arm"):
        super().__init__()
        self.plant = plant
        self.plant_context = plant.CreateDefaultContext()
        self.panda_arm = plant.GetModelInstanceByName(panda_arm_name)
        self.link8_frame = plant.GetBodyByName("panda_link8").body_frame()
        self.panda_hand_frame = plant.GetBodyByName("panda_hand").body_frame()

        self.world_frame = plant.world_frame()

        self.q_input_port = self.DeclareVectorInputPort("q", 7)
        self.spatial_velocity_input_port = self.DeclareVectorInputPort(
            "spatial_velocity", 6
        )
        self.velocity_commanded_output_port = self.DeclareVectorOutputPort(
            "velocity_commanded", 7, self.CalcVelocityCommanded
        )

    def CalcVelocityCommanded(self, context, output):
        t = context.get_time()

        q = self.q_input_port.Eval(context)
        V_G = self.spatial_velocity_input_port.Eval(context)
        self.plant.SetPositions(self.plant_context, self.panda_arm, q)

        J_G = self.plant.CalcJacobianSpatialVelocity(
            self.plant_context,
            JacobianWrtVariable.kQDot,
            self.panda_hand_frame,
            [0, 0, 0],
            self.world_frame,
            self.world_frame,
        )
        J_G = J_G[:, 0:7]

        # prog = MathematicalProgram()
        # v = prog.NewContinuousVariables(7, "v")

        # solver = SnoptSolver()
        # result = solver.Solve(prog)

        # v_max = 3.0
        # V_G_desired = V_G
        # prog.AddCost(np.linalg.norm(J_G.dot(v) - V_G_desired) ** 2)
        # prog.AddBoundingBoxConstraint(-v_max, v_max, v)

        # v = result.GetSolution(v)

        v = np.linalg.pinv(J_G).dot(V_G)
        # print("t: ", t, "v: ", v)
        print("t: ", t, "v: ", v, "V_G: ", V_G)
        output.SetFromVector(v)
