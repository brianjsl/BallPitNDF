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
from manipulation.utils import ConfigureParser
from manipulation.scenarios import AddRgbdSensors
from manipulation.station import AddPointClouds
import numpy as np
from IPython.display import SVG, display
import pydot


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
        panda_arm_position.get_input_port(), f"{panda_arm_name}.position"
    )
    builder.ExportOutput(
        panda_arm_position.get_output_port(),
        f"{panda_arm_name}.position_commanded",
    )

    # export the estimated state, position, and velocity for panda arm from station
    builder.ExportOutput(
        plant.get_state_output_port(panda_arm),
        f"{panda_arm_name}.state_estimated",
    )

    panda_arm_demux = builder.AddSystem(
        Demultiplexer(num_panda_arm_positions * 2, num_panda_arm_positions)
    )  # multiply by 2 because of position and velocity
    builder.Connect(
        plant.get_state_output_port(panda_arm), panda_arm_demux.get_input_port()
    )
    builder.ExportOutput(
        panda_arm_demux.get_output_port(0), f"{panda_arm_name}.position_estimated"
    )
    builder.ExportOutput(
        panda_arm_demux.get_output_port(1), f"{panda_arm_name}.velocity_estimated"
    )

    # create input port for panda hand position
    panda_hand = plant.GetModelInstanceByName(panda_hand_name)
    num_panda_hand_positions = plant.num_positions(panda_hand)

    panda_hand_position = builder.AddSystem(PassThrough(num_panda_hand_positions))
    panda_hand_position.set_name(f"{panda_hand_name}_position")
    builder.ExportInput(
        panda_hand_position.get_input_port(), f"{panda_hand_name}.position"
    )
    builder.ExportOutput(
        panda_hand_position.get_output_port(), f"{panda_hand_name}.position_commanded"
    )

    # export the estimated state, position, and velocity for panda hand from station
    num_panda_hand_positions = plant.num_positions(panda_hand)
    builder.ExportOutput(
        plant.get_state_output_port(panda_hand), f"{panda_hand_name}.state_estimated"
    )
    panda_hand_demux = builder.AddSystem(
        Demultiplexer(num_panda_hand_positions * 2, num_panda_hand_positions)
    )  # multiply by 2 because of position and velocity
    builder.Connect(
        plant.get_state_output_port(panda_hand), panda_hand_demux.get_input_port()
    )
    builder.ExportOutput(
        panda_hand_demux.get_output_port(0), f"{panda_hand_name}.position_estimated"
    )
    builder.ExportOutput(
        panda_hand_demux.get_output_port(1), f"{panda_hand_name}.velocity_estimated"
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

    kp = [100] * num_panda_arm_and_hand_positions
    ki = [1] * num_panda_arm_and_hand_positions
    kd = [20] * num_panda_arm_and_hand_positions

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


    #cheat ports
    builder.ExportOutput(plant.get_body_poses_output_port(), "body_poses")
    builder.ExportOutput(scene_graph.get_query_output_port(), "query_object")


    diagram = builder.Build()
    diagram.set_name("PandaManipulationStation")

    return diagram

