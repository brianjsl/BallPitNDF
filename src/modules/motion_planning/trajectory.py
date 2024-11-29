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
    PiecewisePolynomial,
    PiecewisePose,
    RotationMatrix,
    MinimumDistanceLowerBoundConstraint,
    InverseKinematics,
)
import numpy as np
from manipulation.meshcat_utils import AddMeshcatTriad
from manipulation.meshcat_utils import PublishPositionTrajectory


def CreatePregraspTrajectory_(
    plant: MultibodyPlant,
    plant_context,
    X_WStart: RigidTransform,
    X_WGoal: RigidTransform,
    meshcat: Meshcat,
):
    panda_hand = plant.GetBodyByName("panda_hand")
    X_Grasp = RigidTransform(p=[0, 0, 0])
    X_WPreGrasp = X_WGoal.multiply(X_Grasp)

    AddMeshcatTriad(meshcat, "X_WStart", X_PT=X_WStart)
    AddMeshcatTriad(meshcat, "X_WPreGrasp", X_PT=X_WPreGrasp)
    AddMeshcatTriad(meshcat, "X_WGoal", X_PT=X_WGoal)

    # get gripper frame
    gripper_frame = panda_hand.body_frame()

    num_positions = plant.num_positions()
    num_q = plant.num_positions()
    q0 = plant.GetPositions(plant_context)

    # create trajectory optimization problem
    trajopt = KinematicTrajectoryOptimization(num_positions, 10)
    prog = trajopt.get_mutable_prog()

    # set initial guess
    q_guess = np.tile(q0.reshape((num_positions, 1)), (1, trajopt.num_control_points()))
    q_guess[0, :] = np.linspace(0, -np.pi / 2, trajopt.num_control_points())
    path_guess = BsplineTrajectory(trajopt.basis(), q_guess)
    trajopt.SetInitialGuess(path_guess)

    # add constraints
    trajopt.AddDurationCost(1.0)
    trajopt.AddPathLengthCost(1.0)
    trajopt.AddPositionBounds(
        plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits()
    )

    # vll = plant.GetVelocityLowerLimits()
    # vul = plant.GetVelocityUpperLimits()

    # lower = np.zeros(num_positions)
    # lower[:9] = vll[:9]

    # upper = np.zeros(num_positions)
    # upper[:9] = vul[:9]

    trajopt.AddVelocityBounds(
        plant.GetVelocityLowerLimits(), plant.GetVelocityUpperLimits()
    )
    trajopt.AddDurationConstraint(0.5, 50)

    # start constraint
    start_constraint = PositionConstraint(
        plant,
        plant.world_frame(),
        X_WStart.translation(),
        X_WStart.translation(),
        gripper_frame,
        [0, 0.00, 0],
        plant_context,
    )
    trajopt.AddPathPositionConstraint(start_constraint, 0)
    prog.AddQuadraticErrorCost(np.eye(num_q), q0, trajopt.control_points()[:, 0])

    # goal constraint
    goal_constraint = PositionConstraint(
        plant,
        plant.world_frame(),
        X_WGoal.translation(),
        X_WGoal.translation(),
        gripper_frame,
        [0, 0.05, 0],
        plant_context,
    )

    trajopt.AddPathPositionConstraint(goal_constraint, 1)
    prog.AddQuadraticErrorCost(np.eye(num_q), q0, trajopt.control_points()[:, -1])

    # start and end with zero velocity
    trajopt.AddPathVelocityConstraint(np.zeros((num_q, 1)), np.zeros((num_q, 1)), 0)
    trajopt.AddPathVelocityConstraint(np.zeros((num_q, 1)), np.zeros((num_q, 1)), 1)

    result = Solve(prog)
    traj = trajopt.ReconstructTrajectory(result)

    return traj


def CreatePregraspTrajectory(
    plant: MultibodyPlant,
    plant_context,
    X_WStart: RigidTransform,
    X_WGoal: RigidTransform,
    meshcat: Meshcat,
):
    traj = PiecewisePose.MakeLinear(
        [0, 1.0],
        [X_WStart, X_WGoal],
    )
    # traj = PiecewisePolynomial.FirstOrderHold(
    #     [0, 1.0],
    #     [X_WStart.translation(), X_WGoal.translation()],
    # )

    poses = []
    times = np.linspace(0, 1.0, 30)

    for t in times:
        AddMeshcatTriad(meshcat, path=str(t), X_PT=traj.value(t), opacity=0.2)
        poses.append(traj.value(t))

    world_frame = plant.world_frame()
    gripper_frame = plant.GetFrameByName("panda_hand")
    q_nominal = plant.GetPositions(plant_context)

    def AddOrientationConstraint(ik, R_WG, bounds):
        """Add orientation constraint to the ik problem. Implements an inequality
        constraint where the axis-angle difference between f_R(q) and R_WG must be
        within bounds. Can be translated to:
        ik.prog().AddBoundingBoxConstraint(angle_diff(f_R(q), R_WG), -bounds, bounds)
        """
        ik.AddOrientationConstraint(
            frameAbar=world_frame,
            R_AbarA=R_WG,
            frameBbar=gripper_frame,
            R_BbarB=RotationMatrix(),
            theta_bound=bounds,
        )

    def AddPositionConstraint(ik, p_WG_lower, p_WG_upper):
        """Add position constraint to the ik problem. Implements an inequality
        constraint where f_p(q) must lie between p_WG_lower and p_WG_upper. Can be
        translated to
        ik.prog().AddBoundingBoxConstraint(f_p(q), p_WG_lower, p_WG_upper)
        """
        ik.AddPositionConstraint(
            frameA=world_frame,
            frameB=gripper_frame,
            p_BQ=np.zeros(3),
            p_AQ_lower=p_WG_lower,
            p_AQ_upper=p_WG_upper,
        )

    q_knots = []
    for i, pose in enumerate(poses):
        ik = InverseKinematics(plant)
        q_variables = ik.q()
        prog = ik.prog()

        cost = np.sum((q_variables - q_nominal) ** 2)
        prog.AddCost(cost)

        X_WG = RigidTransform(pose)
        p_WG = X_WG.translation()
        R_WG = X_WG.rotation()

        AddPositionConstraint(ik, p_WG, p_WG)
        AddOrientationConstraint(ik, R_WG, 0.05)

        # set initial guess
        if i == 0:
            prog.SetInitialGuess(q_variables, q_nominal)
        else:
            prog.SetInitialGuess(q_variables, q_knots[i - 1])

        result = Solve(prog)

        assert result.is_success()
        q_knots.append(result.GetSolution(q_variables))

    # create a trajectory from q_knots
    q_knots = np.array(q_knots).T
    print("q_knots: ", q_knots.shape)
    print("times: ", times.shape)
    traj = PiecewisePolynomial.CubicShapePreserving(times, q_knots)
    print(traj)
    return traj


def CreatePickTrajectory():
    pass


def CreatePickAndPourTrajectory(
    plant: MultibodyPlant,
    plant_context,
    X_WStart: RigidTransform,
    X_WGoal: RigidTransform,
    meshcat: Meshcat,
):
    pregrasp_trajectory = CreatePregraspTrajectory(
        plant=plant,
        plant_context=plant_context,
        X_WStart=X_WStart,
        X_WGoal=X_WGoal,
        meshcat=meshcat,
    )

    trajectory = CompositeTrajectory([pregrasp_trajectory])

    return trajectory


class TrajectoryPlanner(LeafSystem):
    """
    Returns a trajectory of joint positions for the panda arm and hand
    """

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

        # translate gripper frame to world frame

        if self.should_plan:
            # get the initial pose of the panda hand
            panda_hand = plant.GetBodyByName("panda_hand")
            X_WStart = plant.EvalBodyPoseInWorld(self.plant_context, panda_hand)
            # X_WGoal = RigidTransform(
            #     # R=X_WStart.rotation(),
            #     R=grasp_pose.rotation(),
            #     # p=X_WStart.translation() + np.array([0, 0, -0.25]),
            #     # p=[-0.25, -0.25, 0.5],
            #     p=grasp_pose.translation(),
            # )

            # # X_WGoal = grasp_pose

            # AddMeshcatTriad(self.meshcat, "X_WStart", X_PT=X_WStart)
            # AddMeshcatTriad(self.meshcat, "X_WPrepare", X_PT=X_WGoal)
            # AddMeshcatTriad(self.meshcat, "X_WGoal", X_PT=grasp_pose)

            # # get gripper frame
            # gripper_frame = panda_hand.body_frame()

            # num_positions = self.plant.num_positions()
            # num_q = plant.num_positions()
            # q0 = plant.GetPositions(self.plant_context)

            # # create trajectory optimization problem
            # trajopt = KinematicTrajectoryOptimization(num_positions, 10)
            # prog = trajopt.get_mutable_prog()

            # # set initial guess
            # q_guess = np.tile(
            #     q0.reshape((num_positions, 1)), (1, trajopt.num_control_points())
            # )
            # q_guess[0, :] = np.linspace(0, -np.pi / 2, trajopt.num_control_points())
            # path_guess = BsplineTrajectory(trajopt.basis(), q_guess)
            # trajopt.SetInitialGuess(path_guess)

            # # add constraints
            # trajopt.AddDurationCost(1.0)
            # trajopt.AddPathLengthCost(1.0)
            # trajopt.AddPositionBounds(
            #     plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits()
            # )

            # vll = plant.GetVelocityLowerLimits()
            # vul = plant.GetVelocityUpperLimits()

            # lower = np.zeros(num_positions)
            # lower[:9] = vll[:9]

            # upper = np.zeros(num_positions)
            # upper[:9] = vul[:9]

            # trajopt.AddVelocityBounds(lower, upper)
            # trajopt.AddDurationConstraint(0.5, 50)

            # # start constraint
            # start_constraint = PositionConstraint(
            #     plant,
            #     plant.world_frame(),
            #     X_WStart.translation(),
            #     X_WStart.translation(),
            #     gripper_frame,
            #     [0, 0.00, 0],
            #     plant_context,
            # )
            # trajopt.AddPathPositionConstraint(start_constraint, 0)
            # prog.AddQuadraticErrorCost(
            #     np.eye(num_q), q0, trajopt.control_points()[:, 0]
            # )

            # # goal constraint
            # goal_constraint = PositionConstraint(
            #     plant,
            #     plant.world_frame(),
            #     X_WGoal.translation(),
            #     X_WGoal.translation(),
            #     gripper_frame,
            #     [0, 0.01, 0],
            #     plant_context,
            # )
            # # goal_orientation_constraint = OrientationConstraint(
            # #     plant,
            # #     frameAbar=plant.world_frame(),
            # #     R_AbarA=X_WGoal.rotation(),
            # #     frameBbar=gripper_frame,
            # #     R_BbarB=RigidTransform().rotation(),
            # #     theta_bound=0.05,
            # #     plant_context=plant_context,
            # # )

            # # trajopt.AddPathPositionConstraint(goal_orientation_constraint, 1)

            # trajopt.AddPathPositionConstraint(goal_constraint, 1)
            # prog.AddQuadraticErrorCost(
            #     np.eye(num_q), q0, trajopt.control_points()[:, -1]
            # )

            # # start and end with zero velocity
            # trajopt.AddPathVelocityConstraint(
            #     np.zeros((num_q, 1)), np.zeros((num_q, 1)), 0
            # )
            # trajopt.AddPathVelocityConstraint(
            #     np.zeros((num_q, 1)), np.zeros((num_q, 1)), 1
            # )

            # # collision_constraint = MinimumDistanceLowerBoundConstraint(
            # #     plant, 0.001, plant_context, None, 0.01
            # # )

            # # evaluate_at_s = np.linspace(0, 1, 5)
            # # for s in evaluate_at_s:
            # #     trajopt.AddPathPositionConstraint(collision_constraint, s)

            # print("yo")
            # result = Solve(prog)
            # traj = trajopt.ReconstructTrajectory(result)
            # print("solved")

            traj = CreatePickAndPourTrajectory(
                plant=plant,
                plant_context=plant_context,
                X_WStart=X_WStart,
                X_WGoal=grasp_pose,
                meshcat=self.meshcat,
            )
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
            q = trajectory.value(t)[:7]
            # q = np.zeros(7)

        print("t: ", t, "q: ", q)
        output.set_value(q)

    def CalcPandaHandQ(self, context, output):
        t = context.get_time()
        trajectory = self.trajectory_input.Eval(context)

        if t < trajectory.end_time():
            q = trajectory.value(t)[-2:]
        else:
            q = trajectory.value(t)[-2:]
            # q = np.zeros(2)

        output.set_value(q)
