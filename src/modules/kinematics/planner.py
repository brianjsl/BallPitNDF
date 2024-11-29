from pydrake.all import (
    AbstractValue,
    AddMultibodyPlantSceneGraph,
    Concatenate,
    DiagramBuilder,
    InputPortIndex,
    LeafSystem,
    MeshcatVisualizer,
    Parser,
    PiecewisePolynomial,
    PiecewisePose,
    PointCloud,
    PortSwitch,
    RandomGenerator,
    RigidTransform,
    RollPitchYaw,
    Simulator,
    StartMeshcat,
    UniformlyRandomRotationMatrix,
    MultibodyPlant,
    Meshcat,
    RotationMatrix
)
from enum import Enum
import numpy as np
from copy import copy
import numpy as np
from manipulation.meshcat_utils import AddMeshcatTriad
from debug.visualize_utils import draw_query_pts
from src.modules.kinematics.grasp_utils import MakeGraspFrames, MakeGraspTrajectories

class PlannerState(Enum):
    INITIAL_STATE = 1
    WAIT_FOR_OBJECTS_TO_SETTLE = 2
    MOVE_TO_BASKET = 3
    MOVE_TO_BALLPIT = 4
    POUR_TO_BALLPIT = 5
    RETURN_BASKET = 6
    GO_HOME = 7

# 9 joints: 7 arm joints + 2 hand joints
NUM_POSITIONS = 9
ARM_POSITIONS = 7
HAND_POSITIONS = 2

class Planner(LeafSystem):
    def __init__(self, meshcat: Meshcat, plant: MultibodyPlant):
        super().__init__()

        #debugging
        self._meshcat = meshcat

        self._gripper_body_index = plant.GetBodyByName("panda_link8").index()
        self._panda_joints = [plant.GetJointByName(f"panda_joint{i}") for i in range(1, 8)]
        self._panda_link_indices = [plant.GetBodyByName(f"panda_link{i}").index() for i in range(1, 8)]

        # input port for the arm's body poses
        self._body_poses_index = self.DeclareAbstractInputPort(
            "body_poses", AbstractValue.Make([RigidTransform()])
        ).get_index()

        # Input port for the LNDF grasp pose
        # Format: pose, query points
        self._basket_grasp_index = self.DeclareAbstractInputPort(
            'basket_grasp', AbstractValue.Make((RigidTransform(), np.ndarray(0)))
        ).get_index()

        # where to pour
        # TODO: Find a more general way to specify (?)
        self._bin_pos = np.array([-0.5, -0.5, 0])

        #state of the hand
        self._hand_state_index = self.DeclareVectorInputPort(
            "hand_state", 4
        ).get_index()
        # self._external_torque_index = self.DeclareVectorInputPort(
        #     "external_torque", 7
        # ).get_index()

        # Arm pose
        self.DeclareAbstractOutputPort(
            "X_WG",
            lambda: AbstractValue.Make(RigidTransform()),
            self.CalcGripperPose,
        )
        self.DeclareVectorOutputPort("hand_position", 2, self.CalcHandPosition)

        # control mode
        self.DeclareAbstractOutputPort(
            "control_mode",
            lambda: AbstractValue.Make(InputPortIndex(0)),
            self.CalcControlMode,
        )
        self.DeclareAbstractOutputPort(
            "reset_diff_ik",
            lambda: AbstractValue.Make(False),
            self.CalcDiffIKReset,
        )


        # When going home
        self.DeclareVectorOutputPort(
            "panda_position_command", ARM_POSITIONS, self.CalcPandaPosition
        )

        # Positions of arm joints at beginning
        self._panda_position_index = self.DeclareVectorInputPort(
            "panda_position", ARM_POSITIONS).get_index()

        # State of the joints 
        self._q0_index = self.DeclareDiscreteState(ARM_POSITIONS)  

        # State of the planner
        self._mode_index = self.DeclareAbstractState(
            AbstractValue.Make(PlannerState.INITIAL_STATE)
        )

        # Trajectory States

        # Gripper pose
        self._traj_X_G_index = self.DeclareAbstractState(
            AbstractValue.Make(PiecewisePose())
        )

        # Hand Pose
        self._trag_hand_index = self.DeclareAbstractState(
            AbstractValue.Make(PiecewisePolynomial())
        )

        # Joint positions
        self._traj_q_index = self.DeclareAbstractState(
            AbstractValue.Make(PiecewisePolynomial())
        )

        # Updates
        self.DeclareInitializationDiscreteUpdateEvent(self.Initialize)
        self.DeclarePeriodicUnrestrictedUpdateEvent(0.1, 0.0, self.Update)

        # Times
        self._times_index = self.DeclareAbstractState(
            AbstractValue.Make({"initial": 0.0})
        )
    
        self._is_done = False

    def Update(self, context, state):
        mode = state.get_mutable_abstract_state(int(self._mode_index))
        mode_value = mode.get_value()

        current_time = context.get_time()
        print(f't: {current_time:.1f}, mode: {mode_value}')

        hand_state = self.get_input_port(self._hand_state_index).Eval(context)
        body_poses = self.get_input_port(self._body_poses_index).Eval(context)

        # pose of the gripper body
        X_G = body_poses[int(self._gripper_body_index)]

        match mode_value:
            case PlannerState.INITIAL_STATE:
                self.GoHome(context, state, current_time)
            case PlannerState.WAIT_FOR_OBJECTS_TO_SETTLE:
                start_time = context.get_abstract_state(int(self._times_index)).get_value()["initial"]
                if current_time > start_time + 0.5:
                    self.PlanGrasp(context, state)
            case PlannerState.MOVE_TO_BASKET:
                X_G_traj = context.get_abstract_state(int(self._traj_X_G_index)).get_value()
                if not X_G_traj.is_time_in_range(current_time):
                    self.PlanPour(context, state)
            case PlannerState.MOVE_TO_BALLPIT:
                X_G_traj = context.get_abstract_state(int(self._traj_X_G_index)).get_value()
                if not X_G_traj.is_time_in_range(current_time):
                    self.Pour(context, state)
            case PlannerState.POUR_TO_BALLPIT:
                X_G_traj = context.get_abstract_state(int(self._traj_X_G_index)).get_value()
                if not X_G_traj.is_time_in_range(current_time):
                    self.ReturnBasket(context, state)
            case PlannerState.RETURN_BASKET:
                X_G_traj = context.get_abstract_state(int(self._traj_X_G_index)).get_value()
                if not X_G_traj.is_time_in_range(current_time):
                    self.GoHome(context, state, current_time)
                    self._is_done = True
            case PlannerState.GO_HOME:
                q_traj = context.get_abstract_state(int(self._traj_q_index)).get_value()
                if not q_traj.is_time_in_range(current_time):
                    if not self._is_done:
                        state.get_mutable_abstract_state(int(self._times_index)).set_value({"initial": current_time})
                        state.get_mutable_abstract_state(int(self._mode_index)).set_value(PlannerState.WAIT_FOR_OBJECTS_TO_SETTLE)
                    else:
                        print("Pouring Done.")
                        return
    
    def GoHome(self, context, state, current_time):
        print('GOING HOME')

        # initial Panda position
        q = self.get_input_port(self._panda_position_index).Eval(context)
        home = copy(context.get_discrete_state(int(self._q0_index)).get_value())
        home[0] = q[0] # Don't reset the first joint (breaks)
        state.get_mutable_abstract_state(int(self._traj_q_index)).set_value(
            PiecewisePolynomial.FirstOrderHold([current_time, current_time + 0.1], np.column_stack((q, home)))
        )
        state.get_mutable_abstract_state(int(self._mode_index)).set_value(PlannerState.GO_HOME)
    
    def PlanGrasp(self, context, state):
        print('PLANNING GRASP')
        initial_pose = self.get_input_port(self._body_poses_index).Eval(context)[int(self._gripper_body_index)]
        grasp_pose, final_query_points = self.get_input_port(self._basket_grasp_index).Eval(context)
        draw_query_pts(self._meshcat, final_query_points)

        # TODO: Add clearance pose
        clearance_pose = RigidTransform(RollPitchYaw(-np.pi / 2, 0, np.pi/2), [0, -0.5, 0.3]) 
        frames = MakeGraspFrames(initial_pose, grasp_pose, clearance_pose, context.get_time(), False)

        #debugging: keep track of trajectory frames
        AddMeshcatTriad(self._meshcat, 'initial', X_PT=frames['initial'][1])
        AddMeshcatTriad(self._meshcat, 'prepare', X_PT=frames['prepare'][1])
        AddMeshcatTriad(self._meshcat, 'pre/post grasp', X_PT=frames['pregrasp'][1])
        AddMeshcatTriad(self._meshcat, 'grasp', X_PT=frames['grasp_start'][1])
        AddMeshcatTriad(self._meshcat, 'clearance', X_PT=frames['clearance'][1])

        print(f'Planned a grasp at time {context.get_time()}')

        X_G_traj, hand_traj = MakeGraspTrajectories(frames)

        state.get_mutable_abstract_state(int(self._traj_X_G_index)).set_value(
            X_G_traj
        )
        state.get_mutable_abstract_state(int(self._trag_hand_index)).set_value(
            hand_traj
        )
        state.get_mutable_abstract_state(int(self._mode_index)).set_value(PlannerState.MOVE_TO_BASKET)
    
    def PlanPour(self, context, state):
        print('PLANNING POUR')
        initial_pose = self.get_input_port(self._body_poses_index).Eval(context)[int(self._gripper_body_index)]

        # have final pos be above the bin
        final_pos = self._bin_pos + np.array([0, 0, 0.4])
        final_pose = RigidTransform(final_pos)

        # just linearly interpolate between the grasping phase
        X_G_traj = PiecewisePose.MakeLinear([context.get_time(), context.get_time() + 5], [initial_pose, final_pose])
        state.get_mutable_abstract_state(int(self._traj_X_G_index)).set_value(
            X_G_traj
        )
        # keep the hand closed
        state.get_mutable_abstract_state(int(self._trag_hand_index)).set_value(
            PiecewisePolynomial.FirstOrderHold([context.get_time(), context.get_time() + 5], np.array([[0.0], [0.0]]).T)
        )
        state.get_mutable_abstract_state(int(self._mode_index)).set_value(PlannerState.MOVE_TO_BALLPIT)
    
    def Pour(self, context, state):
        print('POURING')
        initial_pose = self.get_input_port(self._body_poses_index).Eval(context)[int(self._gripper_body_index)]
        final_pose = initial_pose @ RigidTransform(RotationMatrix(RollPitchYaw(0.75*np.pi, 0, 0)))

        X_G_traj = PiecewisePose.MakeLinear([context.get_time(), context.get_time() + 5,
                                             context.get_time() + 10], [initial_pose, final_pose, initial_pose])
        state.get_mutable_abstract_state(int(self._traj_X_G_index)).set_value(
            X_G_traj
        )

        state.get_mutable_abstract_state(int(self._trag_hand_index)).set_value(
            PiecewisePolynomial.FirstOrderHold([context.get_time(), context.get_time() + 5,
                                                context.get_time() + 10], np.array([[0.0, 0.0, 0.0]]).T)
        )
        state.get_mutable_abstract_state(int(self._mode_index)).set_value(PlannerState.POUR_TO_BALLPIT)

    
    def ReturnBasket(self, context, state):
        print('RETURNING BASKET')
        initial_pose = self.get_input_port(self._body_poses_index).Eval(context)[int(self._gripper_body_index)]
        grasp_pose, final_query_points = self.get_input_port(self._basket_grasp_index).Eval(context)

        # TODO: FIX clearance pose
        clearance_pose = RigidTransform(RollPitchYaw(0, 0, np.pi/2), [0, -0.5, 1.0]) 
        frames = MakeGraspFrames(initial_pose, grasp_pose, clearance_pose, context.get_time(), True)

        #debugging: keep track of trajectory frames
        AddMeshcatTriad(self._meshcat, 'returning initial', X_PT=frames['initial'][1])
        AddMeshcatTriad(self._meshcat, 'returning prepare', X_PT=frames['prepare'][1])
        AddMeshcatTriad(self._meshcat, 'returning post/pre grasp', X_PT=frames['pregrasp'][1])
        AddMeshcatTriad(self._meshcat, 'returning ungrasp', X_PT=frames['grasp_start'][1])
        AddMeshcatTriad(self._meshcat, 'clearance', X_PT=frames['clearance'][1])

        print(f'Planned a return at time {context.get_time()}')

        X_G_traj, hand_traj = MakeGraspTrajectories(frames)
        state.get_mutable_abstract_state(int(self._traj_X_G_index)).set_value(
            X_G_traj
        )
        state.get_mutable_abstract_state(int(self._trag_hand_index)).set_value(
            hand_traj
        )
        state.get_mutable_abstract_state(int(self._mode_index)).set_value(PlannerState.GO_HOME)

    def CalcGripperPose(self, context, output):
        """
        Calculate the pose of the panda arm 
        """
        mode_value = context.get_abstract_state(int(self._mode_index)).get_value()
        current_time = context.get_time()
        if mode_value in [PlannerState.INITIAL_STATE, PlannerState.GO_HOME]:
            output.set_value(self.get_input_port(int(self._body_poses_index)).Eval(context)[int(self._gripper_body_index)])
        elif mode_value in [PlannerState.MOVE_TO_BASKET,PlannerState.MOVE_TO_BALLPIT, PlannerState.POUR_TO_BALLPIT, PlannerState.RETURN_BASKET]:
            pose_trajectory = context.get_mutable_abstract_state(int(self._traj_X_G_index)).get_value()
            if pose_trajectory.get_number_of_segments() > 0 and pose_trajectory.is_time_in_range(current_time):
                output.set_value(pose_trajectory.GetPose(current_time))
            else:
                output.set_value(self.get_input_port(int(self._body_poses_index)).Eval(context)[int(self._gripper_body_index)])
        elif mode_value == PlannerState.WAIT_FOR_OBJECTS_TO_SETTLE:
            # TODO: 
            body_poses = self.get_input_port(self._body_poses_index).Eval(context)
            # pose of the gripper body
            X_G = body_poses[int(self._gripper_body_index)]
            output.set_value(X_G)
        else: 
            raise RuntimeError(f"Unknown State; {mode_value}")

    def CalcHandPosition(self, context, output):
        """
        Calculate the position of the panda hand (open or closed) 
        """
        mode_value = context.get_abstract_state(int(self._mode_index)).get_value()
        hand_state = self.get_input_port(int(self._hand_state_index)).Eval(context)
        current_time = context.get_time()

        opened = np.array([-0.1, 0.1])
        closed = np.array([0.0, 0.0])

        if mode_value in [PlannerState.MOVE_TO_BASKET, PlannerState.RETURN_BASKET]:
            hand_trajectory = context.get_abstract_state(int(self._trag_hand_index)).get_value()
            if hand_trajectory.get_number_of_segments() > 0 and hand_trajectory.is_time_in_range(current_time):
                output.SetFromVector(hand_trajectory.value(current_time))
        elif mode_value in [PlannerState.WAIT_FOR_OBJECTS_TO_SETTLE, PlannerState.INITIAL_STATE, PlannerState.GO_HOME]:
            output.SetFromVector(opened)
        elif mode_value in [PlannerState.MOVE_TO_BALLPIT, PlannerState.POUR_TO_BALLPIT]:
            # move to ballpit or pour the hand will be closed
            output.SetFromVector(closed)
        else:
            raise RuntimeError(f"Unknown State; {mode_value}")
    
    def CalcControlMode(self, context, output):
        mode_value = context.get_abstract_state(int(self._mode_index)).get_value()

        if mode_value == PlannerState.GO_HOME:
            output.set_value(InputPortIndex(2))
        else:
            output.set_value(InputPortIndex(1))

    def CalcDiffIKReset(self, context, output):
        mode_value = context.get_abstract_state(int(self._mode_index)).get_value()

        if mode_value == PlannerState.GO_HOME:
            output.set_value(True)
        else:
            output.set_value(False) 
    
    def Initialize(self, context, discrete_state):
        # discrete_state.set_value(
        #     int(self._q0_index),
        #     self.get_input_port(int(self._panda_position_index)).Eval(context))

        # Get the current joint positions
        q = self.get_input_port(int(self._panda_position_index)).Eval(context)
        discrete_state.set_value(int(self._q0_index), q)

    def CalcPandaPosition(self, context, output):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()

        q = self.get_input_port(int(self._panda_position_index)).Eval(context)
        current_time = context.get_time()

        if mode == PlannerState.GO_HOME:
            q_traj = context.get_abstract_state(int(self._traj_q_index)).get_value()
            output.SetFromVector(q_traj.value(current_time))
        else:
            output.SetFromVector(q)