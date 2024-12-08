from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    Parser,
    MeshcatVisualizer,
    InverseDynamicsController,
    PassThrough,
    Demultiplexer,
    StateInterpolatorWithDiscreteDerivative,
    Multiplexer,
    Meshcat,
    MultibodyPlant,
    DepthRenderCamera,
    RenderCameraCore,
    DepthRange,
    CameraInfo,
    ClippingRange,
    RigidTransform,
    Diagram
)
import os
from manipulation.utils import ConfigureParser
from manipulation.scenarios import AddRgbdSensors
from manipulation.station import AddPointClouds
import numpy as np
from IPython.display import SVG, display
import pydot
from omegaconf import DictConfig
import random
from hydra.utils import get_original_cwd

OBJECT_CLASSES = ['mug', 'basket', 'bottle', 'bowl']
object_params = {
    'mug': {
        'folder_name': 'mug',
        'obj_name': 'mug',
        'link_name': 'mug_body_link'
    },
    'basket': {
        'folder_name': 'wooden_basket_big_handle',
        'obj_name': 'wooden_basket',
        'link_name': 'basket_body_link'
    }, 
    'bowl': {
        'folder_name': 'bowl',
        'obj_name': 'bowl',
        'link_name': 'bowl_body_link'
    }
}

def get_directives(directives_cfg: DictConfig) -> tuple[str, str]:
    object = directives_cfg['object']

    assert object in OBJECT_CLASSES, f"Object {object} unknown."

    # description of robot
    robot_directives = """
directives:
    - add_model:
        name: panda_arm
        file: package://drake_models/franka_description/urdf/panda_arm.urdf
        default_joint_positions:
            panda_joint1: [-1.57]
            panda_joint2: [0.1]
            panda_joint3: [0]
            panda_joint4: [-1.2]
            panda_joint5: [0]
            panda_joint6: [ 1.6]
            panda_joint7: [0]
    - add_weld:
        parent: world
        child: panda_arm::panda_link0
    - add_model:
        name: panda_hand
        file: package://drake_models/franka_description/urdf/panda_hand.urdf
    - add_weld:
        parent: panda_arm::panda_link8
        child: panda_hand::panda_hand
        X_PC:
            translation: [0, 0, 0]
            rotation: !Rpy { deg: [0, 0, 0] }
"""

    if object == 'bowl':
        rot = 15* random.random() + 180
    elif object == 'mug':
        rot = -30 * random.random()
    else:
        rot = -90 * random.random()

    # description of objects in env
    env_directives = f"""
directives:
    - add_model:
        name: floor
        file: package://manipulation/floor.sdf
    - add_weld:
        parent: world
        child: floor::box
        X_PC:
            translation: [0, 0, -0.05]
    - add_model:
        name: {object}
        file: file://{get_original_cwd()}/src/assets/{(object_params[object]['folder_name'])}/{object_params[object]['obj_name']}.sdf
        default_free_body_pose:
            {object_params[object]['link_name']}:
                translation: {[0.4, -0.07, 0.32] if object == 'bowl' else [0.4, 0, 0.1]}
                rotation: !Rpy {{ deg: [90, 0, {rot}]}}

    - add_frame:
        name: camera0_origin
        X_PF:
            base_frame: world
            rotation: !Rpy {{deg: [-130.0, 0, 0.0]}}
            translation: [.5, -.6, 0.8]

    - add_model:
        name: camera0
        file: package://manipulation/camera_box.sdf

    - add_weld:
        parent: camera0_origin
        child: camera0::base

    - add_frame:
        name: camera1_origin
        X_PF:
            base_frame: world
            rotation: !Rpy {{deg: [-140., 0, 90.0]}}
            translation: [1.0, 0, 0.8]

    - add_model:
        name: camera1
        file: package://manipulation/camera_box.sdf

    - add_weld:
        parent: camera1_origin
        child: camera1::base

    - add_frame:
        name: camera2_origin
        X_PF:
            base_frame: world
            rotation: !Rpy {{deg: [-130., 0, -180.0]}}
            translation: [.5, .7, .8]

    - add_model:
        name: camera2
        file: package://manipulation/camera_box.sdf

    - add_weld:
        parent: camera2_origin
        child: camera2::base

    # - add_frame:
    #     name: camera3_origin
    #     X_PF:
    #         base_frame: world
    #         rotation: !Rpy {{deg: [0, 0, 90.0]}}
    #         translation: [.5, 0, -.5] 

    # - add_model:
    #     name: camera3
    #     file: package://manipulation/camera_box.sdf

    # - add_weld:
    #     parent: camera3_origin
    #     child: camera3::base
    
    - add_model:
        name: bin0
        file: file://{get_original_cwd()}/src/assets/ballpit/bin.sdf
        default_free_body_pose:
            bin_base:
                translation: [-0.2, -0.5, 0]
                rotation: !Rpy {{deg: [0, 0, 90]}}
"""
    for i in range(directives_cfg.num_balls):
        env_directives += f"""
    - add_model:
        name: ball_{i}
        file: file://{get_original_cwd()}/src/assets/sphere/sphere_small.sdf
        default_free_body_pose:
            sphere_body_link:
                translation: {[0.4, -0.07, 0.35] if object == 'bowl' else [0.4, 0, 0.2]}
"""
        
    if object == 'bowl':
        env_directives += f"""
    - add_frame:
        name: table_origin
        X_PF:
            base_frame: world
            rotation: !Rpy {{deg: [0.0, 0, 0.0]}}
            translation: [.4, -0.1, 0.25]
    - add_model:
        name: table
        file: file://{get_original_cwd()}/src/assets/table/table.sdf

    - add_weld:
        parent: table_origin
        child: table::table
"""
    return robot_directives, env_directives

def MakePandaManipulationStation(
    robot_directives: str,
    env_directives: str,
    meshcat: Meshcat,
    panda_arm_name: str = "panda_arm",
    panda_hand_name: str = "panda_hand",
    time_step: float = 1e-4,
    camera_prefix: str = "camera",
):
    """
    Sets up the environment with a panda arm and hand. Returns a diagram with controls for the panda arm and hand.

    Args:
        - robot_directives: string containing the directives for the panda arm and hand. Add cameras to this
          directive
        - env_directives: string containing the directives for the environment
        - meshcat: meshcat instance
        - panda_arm_name: name of the panda arm
        - panda_hand_name: name of the panda hand
        - time_step: time step for the simulation

    Returns: Diagram
    """
    meshcat.Delete()

    # initialize builder and plant
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=time_step)

    # parse and process the directive
    parser = Parser(plant)
    ConfigureParser(parser)
    parser.AddModelsFromString(robot_directives, ".dmd.yaml")
    parser.AddModelsFromString(env_directives, ".dmd.yaml")

    plant.Finalize()

    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    #Add Cameras
    AddRgbdSensors(builder,
                   plant,
                   scene_graph,
                   also_add_point_clouds=True,
                   model_instance_prefix=camera_prefix, 
                   depth_camera=None)

    # create controllers for the panda arm and hand
    panda_arm = plant.GetModelInstanceByName(panda_arm_name)
    num_panda_arm_positions = plant.num_positions(panda_arm)

    # export input and output port for panda arm position
    panda_arm_position = builder.AddSystem(PassThrough(num_panda_arm_positions))
    panda_arm_position.set_name(f"{panda_arm_name}_position")
    builder.ExportInput(
        panda_arm_position.get_input_port(), f"{panda_arm_name}_position"
    )
    builder.ExportOutput(
        panda_arm_position.get_output_port(),
        f"{panda_arm_name}_position_commanded",
    )

    # export the estimated state, position, and velocity for panda arm from station
    builder.ExportOutput(
        plant.get_state_output_port(panda_arm),
        f"{panda_arm_name}_state_estimated",
    )

    panda_arm_demux = builder.AddSystem(
        Demultiplexer(num_panda_arm_positions * 2, num_panda_arm_positions)
    )  # multiply by 2 because of position and velocity
    builder.Connect(
        plant.get_state_output_port(panda_arm), panda_arm_demux.get_input_port()
    )
   
    builder.ExportOutput(
        panda_arm_demux.get_output_port(0), f"{panda_arm_name}_position_measured"
    )
    builder.ExportOutput(
        panda_arm_demux.get_output_port(1), f"{panda_arm_name}_velocity_estimated"
    )

    # create input port for panda hand position
    panda_hand = plant.GetModelInstanceByName(panda_hand_name)
    num_panda_hand_positions = plant.num_positions(panda_hand)

    panda_hand_position = builder.AddSystem(PassThrough(num_panda_hand_positions))
    panda_hand_position.set_name(f"{panda_hand_name}_position")
    builder.ExportInput(
        panda_hand_position.get_input_port(), f"{panda_hand_name}_position"
    )
    builder.ExportOutput(
        panda_hand_position.get_output_port(), f"{panda_hand_name}_position_commanded"
    )

    # export the estimated state, position, and velocity for panda hand from station
    num_panda_hand_positions = plant.num_positions(panda_hand)
    builder.ExportOutput(
        plant.get_state_output_port(panda_hand), f"{panda_hand_name}_state_estimated"
    )
    panda_hand_demux = builder.AddSystem(
        Demultiplexer(num_panda_hand_positions * 2, num_panda_hand_positions)
    )  # multiply by 2 because of position and velocity
    builder.Connect(
        plant.get_state_output_port(panda_hand), panda_hand_demux.get_input_port()
    )
    builder.ExportOutput(
        panda_hand_demux.get_output_port(0), f"{panda_hand_name}_position_measured"
    )
    builder.ExportOutput(
        panda_hand_demux.get_output_port(1), f"{panda_hand_name}_velocity_estimated"
    )

    # create controller for panda arm and hand
    controller_plant = MultibodyPlant(time_step)
    controller_parser = Parser(controller_plant)
    ConfigureParser(controller_parser)
    controller_parser.AddModelsFromString(robot_directives, ".dmd.yaml")
    controller_plant.Finalize()

    # num_panda_arm_and_hand_positions = num_panda_arm_positions
    num_panda_arm_and_hand_positions = (
        num_panda_arm_positions + num_panda_hand_positions
    )

    kp = [5000] * num_panda_arm_and_hand_positions
    ki = [50] * num_panda_arm_and_hand_positions
    kd = [1000] * num_panda_arm_and_hand_positions

    panda_controller = builder.AddSystem(
        InverseDynamicsController(
            controller_plant,
            kp=kp,
            ki=ki,
            kd=kd,
            has_reference_acceleration=False,
        )
    )
    panda_controller.set_name("panda_controller")

    # create a multiplexer to combine the panda arm and hand state
    panda_arm_and_hand_state = builder.AddSystem(
        Multiplexer([num_panda_arm_positions, num_panda_hand_positions] * 2)
    )

    builder.Connect(
        panda_arm_demux.get_output_port(0),
        panda_arm_and_hand_state.get_input_port(0),
    )

    builder.Connect(
        panda_hand_demux.get_output_port(0),
        panda_arm_and_hand_state.get_input_port(1),
    )

    builder.Connect(
        panda_arm_demux.get_output_port(1),
        panda_arm_and_hand_state.get_input_port(2),
    )

    builder.Connect(
        panda_hand_demux.get_output_port(1),
        panda_arm_and_hand_state.get_input_port(3),
    )

    builder.Connect(
        panda_arm_and_hand_state.get_output_port(),
        panda_controller.get_input_port_estimated_state(),
    )

    builder.Connect(
        panda_controller.get_output_port_control(), plant.get_actuation_input_port()
    )

    # connect desired position to controller
    desired_state_from_position = builder.AddSystem(
        StateInterpolatorWithDiscreteDerivative(
            num_panda_arm_and_hand_positions,
            time_step=time_step,
            suppress_initial_transient=True,
        )
    )

    builder.Connect(
        desired_state_from_position.get_output_port(),
        panda_controller.get_input_port_desired_state(),
    )

    multiplex = builder.AddSystem(
        Multiplexer([num_panda_arm_positions, num_panda_hand_positions])
    )
    builder.Connect(panda_arm_position.get_output_port(), multiplex.get_input_port(0))
    builder.Connect(panda_hand_position.get_output_port(), multiplex.get_input_port(1))

    builder.Connect(
        # panda_arm_position.get_output_port(),
        multiplex.get_output_port(),
        desired_state_from_position.get_input_port(),
    )

    # Create a multiplexer to combine the panda arm and hand positions and velocities
    panda_arm_and_hand_state = builder.AddSystem(
        Multiplexer([num_panda_arm_positions, num_panda_hand_positions] * 2)
    )  # Multiply by 2 because of positions and velocities
    # Connect the positions and velocities to the multiplexer
    builder.Connect(
        panda_arm_demux.get_output_port(0),  # Arm positions
        panda_arm_and_hand_state.get_input_port(0),
    )
    builder.Connect(
        panda_hand_demux.get_output_port(0),  # Hand positions
        panda_arm_and_hand_state.get_input_port(1),
    )
    builder.Connect(
        panda_arm_demux.get_output_port(1),  # Arm velocities
        panda_arm_and_hand_state.get_input_port(2),
    )
    builder.Connect(
        panda_hand_demux.get_output_port(1),  # Hand velocities
        panda_arm_and_hand_state.get_input_port(3),
    )
    builder.ExportOutput(
        panda_arm_and_hand_state.get_output_port(),
        "robot_state_estimated",
    )

    #cheat ports
    builder.ExportOutput(plant.get_body_poses_output_port(), "body_poses")
    builder.ExportOutput(scene_graph.get_query_output_port(), "query_object")


    diagram = builder.Build()
    diagram.set_name("PandaManipulationStation")

    return diagram

def CreateArmOnlyPlant(time_step: float) -> MultibodyPlant:
    arm_only_plant = MultibodyPlant(time_step=time_step)
    parser = Parser(arm_only_plant)
    parser.AddModelsFromUrl("package://drake_models/franka_description/urdf/panda_arm.urdf")
    arm_only_plant.WeldFrames(
        arm_only_plant.world_frame(),
        arm_only_plant.GetFrameByName("panda_link0")
    )
    arm_only_plant.Finalize()
    return arm_only_plant