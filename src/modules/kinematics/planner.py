from pydrake.all import (
    AbstractValue,
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

# By UNPOURING, we mean the state of waiting 
# for the bucket to get back to level

class PlannerState(Enum):
    INITIAL_STATE = 1
    WAIT_FOR_OBJECTS_TO_SETTLE = 2
    MOVE_TO_BASKET = 3
    MOVE_TO_BALLPIT = 4
    POUR_TO_BALLPIT = 5
    UNPOUR_BASKET = 6,
    RETURN_BASKET = 7,
    GO_HOME = 8,

# 9 joints: 7 arm joints + 2 hand joints
NUM_POSITIONS = 9
ARM_POSITIONS = 7
HAND_POSITIONS = 2

class Planner(LeafSystem):
    def __init__(self, meshcat: Meshcat, plant: MultibodyPlant, object: str):
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
        self._obj_grasp_index = self.DeclareAbstractInputPort(
            'obj_grasp', AbstractValue.Make((RigidTransform(), np.ndarray(0)))
        ).get_index()

        # where to pour
        # TODO: Find a more general way to specify (?)
        self._bin_pos = np.array([-0.2, -0.5, 0])

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

        #Position to Command for Hand (the calculation is direct and does not go to a diffIK solver)
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


        # Position to Command for Panda
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
        self._traj_hand_index = self.DeclareAbstractState(
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
        self.object = object

    def Update(self, context, state):
        mode = state.get_mutable_abstract_state(int(self._mode_index))
        mode_value = mode.get_value()
        current_time = context.get_time()
        print(f't: {current_time:.1f}, mode: {mode_value}')

        # Store trajectories and times in local variables
        traj_X_G = context.get_abstract_state(int(self._traj_X_G_index)).get_value()
        traj_end_time = traj_X_G.end_time() if traj_X_G.get_number_of_segments() > 0 else None

        if mode_value == PlannerState.INITIAL_STATE:
            self.GoHome(context, state, current_time)
        elif mode_value == PlannerState.WAIT_FOR_OBJECTS_TO_SETTLE:
            start_time = context.get_abstract_state(int(self._times_index)).get_value()["initial"]
            if current_time > start_time + 2.0:
                self.PlanGrasp(context, state)
        elif mode_value in [PlannerState.MOVE_TO_BASKET, PlannerState.MOVE_TO_BALLPIT,
                            PlannerState.POUR_TO_BALLPIT, PlannerState.RETURN_BASKET,
                            PlannerState.UNPOUR_BASKET]:
            if traj_end_time and current_time > traj_end_time:
                next_mode = {
                    PlannerState.MOVE_TO_BASKET: self.PlanPour,
                    PlannerState.MOVE_TO_BALLPIT: self.Pour,
                    PlannerState.POUR_TO_BALLPIT: self.UnPour,
                    PlannerState.UNPOUR_BASKET: self.ReturnBasket,
                    PlannerState.RETURN_BASKET: lambda c, s: self.GoHome(c, s, current_time)
                }
                next_mode[mode_value](context, state)
                if mode_value == PlannerState.MOVE_TO_BALLPIT:
                    state.get_mutable_abstract_state(int(self._mode_index)).set_value(PlannerState.POUR_TO_BALLPIT)
                elif mode_value == PlannerState.POUR_TO_BALLPIT:
                    state.get_mutable_abstract_state(int(self._mode_index)).set_value(PlannerState.UNPOUR_BASKET)
                elif mode_value == PlannerState.RETURN_BASKET:
                    self._is_done = True

        elif mode_value == PlannerState.GO_HOME:
            traj_q = context.get_abstract_state(int(self._traj_q_index)).get_value()
            traj_q_end_time = traj_q.end_time()
            if current_time > traj_q_end_time:
                if not self._is_done:
                    state.get_mutable_abstract_state(int(self._times_index)).set_value({"initial": current_time})
                    state.get_mutable_abstract_state(int(self._mode_index)).set_value(PlannerState.WAIT_FOR_OBJECTS_TO_SETTLE)
                else:
                    print("Pouring Done.")

        else:
            raise RuntimeError(f"Unknown State: {mode_value}")
    
    def GoHome(self, context, state, current_time):
        print('GOING HOME')

        # initial Panda position
        q = self.get_input_port(self._panda_position_index).Eval(context)
        home = np.copy(context.get_discrete_state(int(self._q0_index)).get_value())
        home[0] = q[0] # Don't reset the first joint (breaks)
        state.get_mutable_abstract_state(int(self._traj_q_index)).set_value(
            PiecewisePolynomial.FirstOrderHold([current_time, current_time + 2.0], np.column_stack((q, home)))
        )
        state.get_mutable_abstract_state(int(self._mode_index)).set_value(PlannerState.GO_HOME)
    
    def PlanGrasp(self, context, state):
        print('PLANNING GRASP')
        initial_pose = self.get_input_port(self._body_poses_index).Eval(context)[int(self._gripper_body_index)]
        grasp_pose, final_query_points = self.get_input_port(self._obj_grasp_index).Eval(context)
        draw_query_pts(self._meshcat, final_query_points)

        # TODO: Add clearance pose
        if self.object == 'basket':
            clearance_pose = RigidTransform(RollPitchYaw(np.pi, 0, -np.pi/2), [0.5, -0.2, 0.5]) 
        else:
            clearance_pose = RigidTransform(RollPitchYaw(np.pi, -np.pi/2, -np.pi/2), [0.5, -0.2, 0.5])
        frames = MakeGraspFrames(initial_pose, grasp_pose, clearance_pose, context.get_time(), False, self.object)

        #debugging: keep track of trajectory frames
        AddMeshcatTriad(self._meshcat, 'initial', X_PT=frames['initial'][1])
        AddMeshcatTriad(self._meshcat, 'prepare', X_PT=frames['prepare'][1])
        AddMeshcatTriad(self._meshcat, 'pregrasp', X_PT=frames['pregrasp'][1])
        AddMeshcatTriad(self._meshcat, 'grasp', X_PT=frames['grasp_start'][1])
        AddMeshcatTriad(self._meshcat, 'clearance', X_PT=frames['clearance'][1])

        print(f'Planned a grasp at time {context.get_time()}')

        X_G_traj, hand_traj = MakeGraspTrajectories(frames)

        state.get_mutable_abstract_state(int(self._traj_X_G_index)).set_value(
            X_G_traj
        )
        state.get_mutable_abstract_state(int(self._traj_hand_index)).set_value(
            hand_traj
        )
        state.get_mutable_abstract_state(int(self._mode_index)).set_value(PlannerState.MOVE_TO_BASKET)
    
    def PlanPour(self, context, state):
        print('PLANNING POUR')
        initial_pose = self.get_input_port(self._body_poses_index).Eval(context)[int(self._gripper_body_index)]

        # So we don't crash into the ballpit
        intermediate_pos = self._bin_pos + np.array([0.5, 0, 0.5])
        
        if self.object == 'basket':
            intermediate_pose = RigidTransform(RollPitchYaw(np.pi, 0, -np.pi/2), intermediate_pos)
        else:
            intermediate_pose = RigidTransform(RollPitchYaw(np.pi/2, -np.pi/2, -np.pi/2), intermediate_pos)
        
        # have final pos be above the bin
        if self.object == 'basket':
            final_pos = self._bin_pos + np.array([0.0, 0, 0.5])
        else:
            final_pos = self._bin_pos + np.array([0.2, 0, 0.5])


        if self.object == 'basket':
            final_pose = RigidTransform(RollPitchYaw(np.pi, 0, -np.pi/2), final_pos)
        else:
            final_pose = RigidTransform(RollPitchYaw(np.pi/2, -np.pi/2, -np.pi/2), final_pos)

        X_G_traj = PiecewisePose.MakeLinear([context.get_time(), context.get_time()+3, context.get_time() + 6], [initial_pose, intermediate_pose, final_pose])
        state.get_mutable_abstract_state(int(self._traj_X_G_index)).set_value(
            X_G_traj
        )
        AddMeshcatTriad(self._meshcat, 'pour', X_PT=final_pose)
        AddMeshcatTriad(self._meshcat, 'prepour', X_PT=intermediate_pose)

        # keep the hand closed
        state.get_mutable_abstract_state(int(self._traj_hand_index)).set_value(
            PiecewisePolynomial.FirstOrderHold([context.get_time(), context.get_time() + 3, context.get_time() + 6], np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]).T)
        )
        state.get_mutable_abstract_state(int(self._mode_index)).set_value(PlannerState.MOVE_TO_BALLPIT)
    
    def Pour(self, context, state):
        print('POURING')
        initial_pose = self.get_input_port(self._body_poses_index).Eval(context)[int(self._gripper_body_index)]

        if self.object == 'basket':
            rot_pose = initial_pose @ RigidTransform(RotationMatrix(RollPitchYaw(0.5*np.pi, 0, 0)), [0, 0, 0]) 
            inter_pose = RigidTransform(RotationMatrix(RollPitchYaw(np.pi, 0, -np.pi/2)), rot_pose.translation())
        else:
            rot_pose = initial_pose @ RigidTransform(RotationMatrix(RollPitchYaw(0, -0.75*np.pi, 0)), [0, 0, 0]) 
            inter_pose = RigidTransform(RotationMatrix(RollPitchYaw(np.pi/2, -np.pi/2, -np.pi/2)), rot_pose.translation())

        X_G_traj = PiecewisePose.MakeLinear([context.get_time(), 
                                             context.get_time() + 3, context.get_time() + 10, context.get_time()+13,
                                             context.get_time()+16], [initial_pose, rot_pose, rot_pose, inter_pose, inter_pose])
        state.get_mutable_abstract_state(int(self._traj_X_G_index)).set_value(
            X_G_traj
        )

        state.get_mutable_abstract_state(int(self._traj_hand_index)).set_value(
            PiecewisePolynomial.FirstOrderHold([context.get_time(), context.get_time() + 3,
                                                context.get_time() + 10, context.get_time()+13, 
                                                context.get_time() + 16 
                                                ], np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0,0.0]]).T)
        )
    
    def UnPour(self, context, state):
        print('UNPOURING')
        initial_pose = self.get_input_port(self._body_poses_index).Eval(context)[int(self._gripper_body_index)]

        if self.object == 'basket':
            final_pose = RigidTransform(RotationMatrix(RollPitchYaw(np.pi, 0, -np.pi/2)), [0, -0.5, 0.6])
        else:
            final_pose = RigidTransform(RotationMatrix(RollPitchYaw(np.pi/2, -np.pi/2, -np.pi/2)), [0, -0.5, 0.6])

        X_G_traj = PiecewisePose.MakeLinear([context.get_time(),
                                             context.get_time() + 3], [initial_pose, final_pose])
        state.get_mutable_abstract_state(int(self._traj_X_G_index)).set_value(
            X_G_traj
        )

        state.get_mutable_abstract_state(int(self._traj_hand_index)).set_value(
            PiecewisePolynomial.FirstOrderHold([context.get_time(), context.get_time() + 3], np.array([[0.0, 0.0], [0.0, 0.0]]).T)
        )

    
    def ReturnBasket(self, context, state):
        print('RETURNING BASKET')
        initial_pose = self.get_input_port(self._body_poses_index).Eval(context)[int(self._gripper_body_index)]

        if self.object == 'basket':
            grasp_pose = RigidTransform(RollPitchYaw(np.pi, 0, -np.pi/2), [0.5, 0, 0.9])
        else:
            grasp_pose = RigidTransform(RollPitchYaw(np.pi, -np.pi/2, -np.pi/2), [0.5, 0, 0.9])

        # TODO: FIX clearance pose

        if self.object == 'basket':
            clearance_pose = RigidTransform(RollPitchYaw(np.pi, 0, -np.pi/2), [0.5, -0.2, 0.9]) 
        else:
            clearance_pose = RigidTransform(RollPitchYaw(np.pi, -np.pi/2, -np.pi/2), [0.5, -0.2, 0.9]) 

        frames = MakeGraspFrames(initial_pose, grasp_pose, clearance_pose, context.get_time(), True, self.object)

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
        state.get_mutable_abstract_state(int(self._traj_hand_index)).set_value(
            hand_traj
        )
        state.get_mutable_abstract_state(int(self._mode_index)).set_value(PlannerState.RETURN_BASKET)

    def CalcGripperPose(self, context, output):
        """
        Calculate the pose of the panda arm 
        """
        mode_value = context.get_abstract_state(int(self._mode_index)).get_value()
        current_time = context.get_time()
        if mode_value in [PlannerState.INITIAL_STATE, PlannerState.GO_HOME]:
            output.set_value(self.get_input_port(int(self._body_poses_index)).Eval(context)[int(self._gripper_body_index)])
        elif mode_value in [PlannerState.MOVE_TO_BASKET,PlannerState.MOVE_TO_BALLPIT, PlannerState.POUR_TO_BALLPIT, 
                            PlannerState.RETURN_BASKET, PlannerState.UNPOUR_BASKET]:
            pose_trajectory = context.get_mutable_abstract_state(int(self._traj_X_G_index)).get_value()
            if pose_trajectory.get_number_of_segments() > 0 and pose_trajectory.is_time_in_range(current_time):
                output.set_value(pose_trajectory.GetPose(current_time))
            else:
                output.set_value(self.get_input_port(int(self._body_poses_index)).Eval(context)[int(self._gripper_body_index)])
        elif mode_value == PlannerState.WAIT_FOR_OBJECTS_TO_SETTLE:
            body_poses = self.get_input_port(self._body_poses_index).Eval(context)
            X_G = body_poses[int(self._gripper_body_index)]
            output.set_value(X_G)
        else: 
            raise RuntimeError(f"Unknown State: {mode_value}")

    def CalcHandPosition(self, context, output):
        """
        Calculate the position of the panda hand (open or closed) 
        """
        mode_value = context.get_abstract_state(int(self._mode_index)).get_value()
        current_time = context.get_time()

        opened = np.array([-0.1, 0.1])
        closed = np.array([0.0, 0.0])

        if mode_value in [PlannerState.MOVE_TO_BASKET, PlannerState.RETURN_BASKET]:
            hand_trajectory = context.get_abstract_state(int(self._traj_hand_index)).get_value()
            if hand_trajectory.get_number_of_segments() > 0 and hand_trajectory.is_time_in_range(current_time):
                desired_position = hand_trajectory.value(current_time)
                output.SetFromVector(desired_position)
            else:
                output.SetFromVector(closed)
        elif mode_value in [PlannerState.WAIT_FOR_OBJECTS_TO_SETTLE, PlannerState.INITIAL_STATE, PlannerState.GO_HOME]:
            output.SetFromVector(opened)
        elif mode_value in [PlannerState.MOVE_TO_BALLPIT, PlannerState.POUR_TO_BALLPIT, PlannerState.UNPOUR_BASKET]:
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
    
    @property
    def is_done(self) -> bool:
        return self._is_done