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


def CreateTrajectoryFromPoses(
    plant: MultibodyPlant, plant_context, poses: list[RigidTransform], times: np.ndarray
):
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
    q_knots = q_knots[:7, :]
    # print(q_knots.shape)
    traj = PiecewisePolynomial.CubicShapePreserving(times, q_knots)
    return traj


def CreatePregraspTrajectory(
    plant: MultibodyPlant,
    plant_context,
    X_WStart: RigidTransform,
    X_WGoal: RigidTransform,
    meshcat: Meshcat,
):
    X_WPreGrasp = X_WGoal
    X_WGrasp = X_WGoal.multiply(RigidTransform(p=[0, 0, 0.0475]))
    X_WPick = RigidTransform(
        R=X_WGrasp.rotation(),
        p=X_WGrasp.translation() + np.array([0, 0, 0.2]),
    )
    X_WPour = RigidTransform(
        R=(X_WPick.rotation()).multiply(RotationMatrix.MakeYRotation(np.pi / 2)),
        p=X_WPick.translation(),
    )
    traj = PiecewisePose.MakeLinear(
        [0, 1.0, 2.0, 3.0, 5.0, 7.0],
        [X_WStart, X_WPreGrasp, X_WGrasp, X_WGrasp, X_WPick, X_WPour],
    )

    poses = []
    times = np.linspace(0, traj.end_time(), 30)

    for t in times:
        AddMeshcatTriad(meshcat, path=str(t), X_PT=traj.value(t), opacity=0.2)
        poses.append(traj.value(t))

    return CreateTrajectoryFromPoses(
        plant=plant, plant_context=plant_context, poses=poses, times=times
    )


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


def CreateEndEffectorTrajectory():
    gripper_t_lst = np.array(
        [
            0.0,
            1.0,
            2.0,
            3.0,
            5.0,
        ]
    )
    gripper_knots = np.array(
        # [[0.0, -0.05, -0.05, -0.05, -0.05], [0.0, 0.05, 0.05, 0.05, 0.05]]
        [[0.0, -0.05, -0.05, 0.0, 0.0], [0.0, 0.05, 0.05, 0.0, 0.0]]
    )

    g_traj = PiecewisePolynomial.FirstOrderHold(gripper_t_lst, gripper_knots)

    return g_traj


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

        self.panda_hand_traj_output = self.DeclareAbstractOutputPort(
            "panda_hand_trajectory",
            lambda: AbstractValue.Make(PiecewisePose()),
            self.CalcPandaHandTrajectory,
        )

        self.panda_arm_traj = None
        self.panda_hand_traj = None
        self.should_plan_panda_arm_traj = True
        self.should_plan_panda_hand_traj = True

    def CalcPandaArmTrajectory(self, context, output):
        plant = self.plant
        plant_context = self.plant_context
        grasp_pose, _ = self.grasp_input_port.Eval(context)

        if self.should_plan_panda_arm_traj:
            # get the initial pose of the panda hand
            panda_hand = plant.GetBodyByName("panda_hand")
            X_WStart = plant.EvalBodyPoseInWorld(self.plant_context, panda_hand)

            panda_arm_traj = CreatePickAndPourTrajectory(
                plant=plant,
                plant_context=plant_context,
                X_WStart=X_WStart,
                X_WGoal=grasp_pose,
                meshcat=self.meshcat,
            )
            output.set_value(panda_arm_traj)
            self.panda_arm_traj = panda_arm_traj
            self.should_plan_panda_arm_traj = False

            print("end_time: ", panda_arm_traj.end_time())
        else:
            output.set_value(self.panda_arm_traj)

    def CalcPandaHandTrajectory(self, context, output):
        if self.should_plan_panda_hand_traj:
            panda_hand_traj = CreateEndEffectorTrajectory()
            output.set_value(panda_hand_traj)

            self.panda_hand_traj = panda_hand_traj
            self.should_plan_panda_hand_traj = False
        else:
            output.set_value(self.panda_hand_traj)


class PandaTrajectoryEvaluator(LeafSystem):
    def __init__(self, q: int):
        super().__init__()
        self.q = q
        self.trajectory_input = self.DeclareAbstractInputPort(
            "trajectory", AbstractValue.Make(PiecewisePose())
        )

        self.panda_arm_q = self.DeclareVectorOutputPort("q", q, self.CalcQ)

    def CalcQ(self, context, output):
        t = context.get_time()
        trajectory = self.trajectory_input.Eval(context)
        q = trajectory.value(t)

        print("t: ", t, "q: ", q)
        output.set_value(q)
