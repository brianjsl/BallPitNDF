from pydrake.all import (
    LeafSystem,
    AbstractValue,
    RigidTransform,
    PublishEvent,
    MultibodyPlant,
    PiecewisePose,
    JacobianWrtVariable,
)
import numpy as np


class PandaDiffIKController(LeafSystem):
    def __init__(self, plant: MultibodyPlant, panda_arm_name: str = "panda_arm"):
        super().__init__()
        self.plant = plant
        self.plant_context = plant.CreateDefaultContext()
        self.panda_arm = plant.GetModelInstanceByName(panda_arm_name)
        self.link6_frame = plant.GetBodyByName("panda_link6").body_frame()
        self.world_frame = plant.world_frame()

        self.q_input_port = self.DeclareVectorInputPort("q", 7)
        self.spatial_velocity_input_port = self.DeclareVectorInputPort(
            "spatial_velocity", 6
        )
        self.velocity_commanded_output_port = self.DeclareVectorOutputPort(
            "velocity_commanded", 7, self.CalcVelocityCommanded
        )

    def CalcVelocityCommanded(self, context, output):
        q = self.q_input_port.Eval(context)
        V_G = self.spatial_velocity_input_port.Eval(context)
        self.plant.SetPositions(self.plant_context, self.panda_arm, q)

        # print("q: ", q, "V_G: ", V_G)

        J_G = self.plant.CalcJacobianSpatialVelocity(
            self.plant_context,
            JacobianWrtVariable.kV,
            self.link6_frame,
            [0, 0, 0],
            self.world_frame,
            self.world_frame,
        )
        J_G = J_G[:, :7]

        v = np.linalg.pinv(J_G).dot(V_G)
        output.SetFromVector(v)


class TrajectoryPlanner(LeafSystem):
    def __init__(self, plant: MultibodyPlant, panda_arm_name: str = "panda_arm"):
        super().__init__()
        self.plant = plant
        self.panda_arm = plant.GetModelInstanceByName(panda_arm_name)
        self.link6 = plant.GetBodyByName("panda_link6")
        self.plant_context = plant.CreateDefaultContext()

        self.prev_goal_pose = RigidTransform()
        self.trajectory = None
        self.trajectory_VG = None  # generalized velocity of trajectory

        self.goal_pose_input_port = self.DeclareAbstractInputPort(
            "goal_pose", AbstractValue.Make(RigidTransform())
        )
        # self.trajectory_output_port = self.DeclareAbstractOutputPort(
        #     "trajectory",
        #     lambda: AbstractValue.Make(PiecewisePolynomial()),
        #     self.CalcTrajectory,
        # )
        self.spatial_velocity_output_port = self.DeclareVectorOutputPort(
            "desired_spatial_velocity", 6, self.CalcDesiredSpatialVelocity
        )

        self.DeclarePeriodicEvent(
            period_sec=1.0, offset_sec=0.0, event=PublishEvent(self.PrintGoalPose)
        )

    # def CalcTrajectory(self, context):
    #     print("YO")

    def CalcDesiredSpatialVelocity(self, context, output):
        t = context.get_time()
        goal_pose = self.goal_pose_input_port.Eval(context)

        # print(
        #     goal_pose.translation(),
        #     self.prev_goal_pose.translation(),
        #     goal_pose.IsExactlyEqualTo(self.prev_goal_pose),
        # )
        if not goal_pose.IsExactlyEqualTo(self.prev_goal_pose):
            # create new trajectory
            current_pose = self.plant.EvalBodyPoseInWorld(
                self.plant_context, self.link6
            )

            print("goal chacned")
            # print("current_pose: ", current_pose)
            # print("goal_pose: ", goal_pose)

            trajectory = PiecewisePose.MakeLinear(
                # times=[t, t + 1],
                times=[t, t + 0.5],
                poses=[current_pose, goal_pose],
            )

            self.trajectory = trajectory
            self.trajectory_VG = trajectory.MakeDerivative(1)
            self.prev_goal_pose = RigidTransform(
                p=goal_pose.translation(), R=goal_pose.rotation()
            )

        spatial_velocity = self.trajectory_VG.value(t).ravel()
        output.SetFromVector(spatial_velocity)

    def PrintGoalPose(self, context, event):
        goal_pose = self.goal_pose_input_port.Eval(context)
        # print(goal_pose)
