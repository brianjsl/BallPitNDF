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
)
from manipulation.utils import ConfigureParser


def MakePandaManipulationStation(
    robot_directives: str,
    env_directives: str,
    meshcat: Meshcat,
    panda_arm_name: str = "panda_arm",
    panda_hand_name: str = "panda_hand",
    time_step: float = 1e-4,
):
    """
    Sets up the environment with a panda arm and hand. Returns a diagram with controls for the panda arm and hand.

    Args:
        - robot_directives: string containing the directives for the panda arm and hand
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

    diagram = builder.Build()
    diagram.set_name("PandaManipulationStation")

    return diagram

# from pydrake.geometry import StartMeshcat
# from pydrake.all import (
#     AddMultibodyPlantSceneGraph,
#     DiagramBuilder,
#     Parser,
#     MeshcatVisualizer,
#     InverseDynamicsController,
#     PassThrough,
#     Demultiplexer,
#     StateInterpolatorWithDiscreteDerivative,
#     Multiplexer,
#     Meshcat,
# )
# from manipulation.station import (
#     MakeHardwareStation,
#     LoadScenario,
#     AppendDirectives
# )
# from manipulation.utils import ConfigureParser
# from omegaconf import DictConfig

# def BuildStation(meshcat:Meshcat, cfg: DictConfig):
#     '''
#     Creates a custom manipulation station from a hydra config with a panda robot arm. 
#     '''
#     scenario = """
# directives:
#     - add_model:
#         name: panda
#         file: package://drake_models/franka_description/urdf/panda_arm.urdf
#         default_joint_positions:
#             panda_joint1: [-1.57]
#             panda_joint2: [0.1]
#             panda_joint3: [0]
#             panda_joint4: [-1.2]
#             panda_joint5: [0]
#             panda_joint6: [ 1.6]
#             panda_joint7: [0]
#     - add_weld:
#         parent: world
#         child: panda::panda_link0
# """

#     for i in range(cfg.num_balls):
#         scenario += """
#     - add_model:
#         name: sphere_{i}
#         file: src/assets/ball/ball.sdf
# """
#     scenario = LoadScenario(scenario)
#     return MakeHardwareStation(scenario, meshcat)

# def build_pouring_diagram(
#     meshcat: Meshcat,
#     cfg: DictConfig,
#     timestep: float = 1e-4,
#     panda_name: str = "panda",
#     panda_hand_name: str = "panda_hand",
# ):
#     '''
#     Builds the manipulation station with the given model directives.
#     '''
#     meshcat.Delete()

#     # Initialize the builder and plant
#     builder = DiagramBuilder()
#     manip_station = BuildStation(meshcat, cfg)
#     station = builder.AddSystem(manip_station)
#     plant = station.GetSubsystemByName("plant")

#     # Parse and process the directive
#     parser = Parser(plant)
#     ConfigureParser(parser)

#     panda = plant.GetModelInstanceByName(panda_name)

#     # export the estimated state, position, and velocity for panda from station
#     builder.ExportOutput(
#         plant.get_state_output_port(panda), f"{panda_name}.state_estimated"
#     )

#     num_panda_positions = plant.num_positions(panda)
#     demux = builder.AddSystem(
#         Demultiplexer(num_panda_positions * 2, num_panda_positions)
#     )  # multiply by 2 because of position and velocity
#     builder.Connect(plant.get_state_output_port(panda), demux.get_input_port())
#     builder.ExportOutput(demux.get_output_port(0), f"{panda_name}.position_estimated")
#     builder.ExportOutput(demux.get_output_port(1), f"{panda_name}.velocity_estimated")

#     # create input port for panda position
#     panda_position = builder.AddSystem(PassThrough(num_panda_positions))
#     panda_position.set_name("panda_position")
#     builder.ExportInput(panda_position.get_input_port(), f"{panda_name}.position")
#     builder.ExportOutput(
#         panda_position.get_output_port(), f"{panda_name}.position_commanded"
#     )

#     # create input port for panda hand position
#     panda_hand = plant.GetModelInstanceByName(panda_hand_name)
#     num_panda_hand_positions = plant.num_positions(panda_hand)

#     panda_hand_position = builder.AddSystem(PassThrough(num_panda_hand_positions))
#     panda_hand_position.set_name("panda_hand_position")
#     builder.ExportInput(
#         panda_hand_position.get_input_port(), f"{panda_hand_name}.position"
#     )
#     builder.ExportOutput(
#         panda_hand_position.get_output_port(), f"{panda_hand_name}.position_commanded"
#     )

#     # export the estimated state, position, and velocity for panda hand from station
#     num_panda_hand_positions = plant.num_positions(panda_hand)
#     builder.ExportOutput(
#         plant.get_state_output_port(panda_hand), f"{panda_hand_name}.state_estimated"
#     )
#     demux = builder.AddSystem(
#         Demultiplexer(num_panda_hand_positions * 2, num_panda_hand_positions)
#     )  # multiply by 2 because of position and velocity
#     builder.Connect(plant.get_state_output_port(panda_hand), demux.get_input_port())
#     builder.ExportOutput(
#         demux.get_output_port(0), f"{panda_hand_name}.position_estimated"
#     )
#     builder.ExportOutput(
#         demux.get_output_port(1), f"{panda_hand_name}.velocity_estimated"
#     )

#     # add controller
#     panda_positions = plant.num_positions()

#     kp = [100] * panda_positions
#     ki = [1] * panda_positions
#     kd = [20] * panda_positions

#     panda_controller = builder.AddSystem(
#         InverseDynamicsController(plant, kp, ki, kd, has_reference_acceleration=False)
#     )
#     panda_controller.set_name("panda_controller")

#     builder.Connect(
#         plant.get_state_output_port(),
#         # plant.get_state_output_port(panda),
#         panda_controller.get_input_port_estimated_state(),
#     )
#     builder.Connect(
#         panda_controller.get_output_port_control(), plant.get_actuation_input_port()
#     )

#     # connect desired position to controller
#     num_plant_positions = plant.num_positions()
#     desired_state_from_position = builder.AddSystem(
#         StateInterpolatorWithDiscreteDerivative(
#             num_plant_positions,
#             time_step=timestep,
#             suppress_initial_transient=True,
#         )
#     )

#     builder.Connect(
#         desired_state_from_position.get_output_port(),
#         panda_controller.get_input_port_desired_state(),
#     )

#     multiplex = builder.AddSystem(
#         Multiplexer([num_panda_positions, num_panda_hand_positions])
#     )
#     builder.Connect(panda_position.get_output_port(), multiplex.get_input_port(0))
#     builder.Connect(panda_hand_position.get_output_port(), multiplex.get_input_port(1))

#     builder.Connect(
#         multiplex.get_output_port(),
#         desired_state_from_position.get_input_port(),
#     )

#     # Build the diagram
#     diagram = builder.Build()
#     diagram.set_name("PandaManipulationStation")

#     return diagram, plant

# def visualize_diagram(diagram):
#     """
#     Util to visualize the system diagram
#     """
#     from IPython.display import SVG, display
#     import pydot
#     display(SVG(pydot.graph_from_dot_data(
#         diagram.GetGraphvizString(max_depth=2))[0].create_svg()))