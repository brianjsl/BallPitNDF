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
    Meshcat,
    KinematicTrajectoryOptimization,
    BsplineTrajectory,
    PositionConstraint,
    OrientationConstraint,
    Solve,
    CompositeTrajectory,
)
import numpy as np
from manipulation.meshcat_utils import AddMeshcatTriad
from manipulation.meshcat_utils import PublishPositionTrajectory


def CreatePickAndPourTrajectory():
    pass


class TrajectoryPlanner(LeafSystem):
    def __init__(self, plant: MultibodyPlant, meshcat: Meshcat):
        super().__init__()

        self.plant = plant
        self.plant_context = plant.CreateDefaultContext()
        self.meshcat = meshcat

        self.grasp_input_port = self.DeclareAbstractInputPort(
            "grasp_pose", AbstractValue.Make((RigidTransform(), np.ndarray(0)))
        )

        self.panda_arm_traj_output = self.DeclareAbstractOutputPort(
            "panda_arm_trajectory",
            lambda: AbstractValue.Make(PiecewisePose()),
            self.CalcPandaArmTrajectory,
        )

        self.traj = None
        self.should_plan = True

    def CalcPandaArmTrajectory(self, context, output):
        plant = self.plant
        plant_context = self.plant_context
        grasp_pose, _ = self.grasp_input_port.Eval(context)

        if self.should_plan:
            # get the initial pose of the panda hand
            panda_hand = plant.GetBodyByName("panda_hand")
            X_WStart = plant.EvalBodyPoseInWorld(self.plant_context, panda_hand)
            X_WGoal = RigidTransform(
                R=X_WStart.rotation(),
                p=grasp_pose.translation() + np.array([0, 0, 0.2]),
            )
            # X_WGoal = RigidTransform(
            #     R=X_WStart.rotation(),
            #     p=X_WStart.translation() + np.array([0, 0, -0.1]),
            # )
            AddMeshcatTriad(self.meshcat, "X_WStart", X_PT=X_WStart)
            AddMeshcatTriad(self.meshcat, "X_WGoal", X_PT=X_WGoal)

            # get gripper frame
            gripper_frame = panda_hand.body_frame()

            num_positions = self.plant.num_positions()
            num_q = plant.num_positions()
            q0 = plant.GetPositions(self.plant_context)

            # create trajectory optimization problem
            trajopt = KinematicTrajectoryOptimization(num_positions, 10)
            prog = trajopt.get_mutable_prog()

            trajopt.AddDurationCost(1.0)
            trajopt.AddPathLengthCost(1.0)
            trajopt.AddPositionBounds(
                plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits()
            )
            trajopt.AddVelocityBounds(
                plant.GetVelocityLowerLimits(), plant.GetVelocityUpperLimits()
            )
            trajopt.AddDurationConstraint(0.5, 5)

            # start constraint
            start_constraint = PositionConstraint(
                plant,
                plant.world_frame(),
                X_WStart.translation(),
                X_WStart.translation(),
                gripper_frame,
                [0, 0.0, 0],
                plant_context,
            )
            trajopt.AddPathPositionConstraint(start_constraint, 0)

            start_orientation_constraint = OrientationConstraint(
                plant,
                frameAbar=plant.world_frame(),
                R_AbarA=X_WStart.rotation(),
                frameBbar=gripper_frame,
                R_BbarB=RigidTransform().rotation(),
                theta_bound=0.1,
                plant_context=plant_context,
            )
            trajopt.AddPathPositionConstraint(start_orientation_constraint, 0)

            prog.AddQuadraticErrorCost(
                np.eye(num_q), q0, trajopt.control_points()[:, 0]
            )

            # goal constraint
            goal_constraint = PositionConstraint(
                plant,
                plant.world_frame(),
                X_WGoal.translation(),
                X_WGoal.translation(),
                gripper_frame,
                [0, 0.0, 0],
                plant_context,
            )
            goal_orientation_constraint = OrientationConstraint(
                plant,
                frameAbar=plant.world_frame(),
                R_AbarA=X_WGoal.rotation(),
                frameBbar=gripper_frame,
                R_BbarB=RigidTransform().rotation(),
                theta_bound=0.1,
                plant_context=plant_context,
            )

            trajopt.AddPathPositionConstraint(goal_orientation_constraint, 1)

            trajopt.AddPathPositionConstraint(goal_constraint, 1)
            prog.AddQuadraticErrorCost(
                np.eye(num_q), q0, trajopt.control_points()[:, -1]
            )

            # start and end with zero velocity
            trajopt.AddPathVelocityConstraint(
                np.zeros((num_q, 1)), np.zeros((num_q, 1)), 0
            )
            trajopt.AddPathVelocityConstraint(
                np.zeros((num_q, 1)), np.zeros((num_q, 1)), 1
            )

            result = Solve(prog)
            traj = trajopt.ReconstructTrajectory(result)

            output.set_value(traj)
            self.traj = traj
            self.should_plan = False

            print("end_time: ", traj.end_time())
        else:
            output.set_value(self.traj)

    def CalcPandaHandTrajectory(self, context, output):
        pass


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

        print("t: ", t, "q: ", q)
        output.set_value(q)

    def CalcPandaHandQ(self, context, output):
        t = context.get_time()
        trajectory = self.trajectory_input.Eval(context)

        if t < trajectory.end_time():
            q = trajectory.value(t)[-2:]
        else:
            q = np.zeros(2)

        output.set_value(q)
