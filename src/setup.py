from pydrake.geometry import StartMeshcat
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
)
from manipulation.station import (
    MakeHardwareStation,
    LoadScenario,
    AppendDirectives
)
from manipulation.utils import ConfigureParser
from omegaconf import DictConfig

def BuildStation(meshcat:Meshcat, cfg: DictConfig):
    '''
    Creates a custom manipulation station from a hydra config with a panda robot arm. 
    '''
    scenario = """
directives:
    - add_model:
        name: panda
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
        child: panda::panda_link0
"""

    for i in range(cfg.num_balls):
        scenario += """
    - add_model:
        name: sphere_{i}
        file: src/assets/ball/ball.sdf
"""
    scenario = LoadScenario(scenario)
    return MakeHardwareStation(scenario, meshcat)

def build_pouring_diagram(
    meshcat: Meshcat,
    cfg: DictConfig,
    timestep: float = 1e-4,
    panda_name: str = "panda",
    panda_hand_name: str = "panda_hand",
):
    '''
    Builds the manipulation station with the given model directives.
    '''
    meshcat.Delete()

    # Initialize the builder and plant
    builder = DiagramBuilder()
    manip_station = BuildStation(meshcat, cfg)
    station = builder.AddSystem(manip_station)
    plant = station.GetSubsystemByName("plant")

    # Parse and process the directive
    parser = Parser(plant)
    ConfigureParser(parser)

    panda = plant.GetModelInstanceByName(panda_name)

    # export the estimated state, position, and velocity for panda from station
    builder.ExportOutput(
        plant.get_state_output_port(panda), f"{panda_name}.state_estimated"
    )

    num_panda_positions = plant.num_positions(panda)
    demux = builder.AddSystem(
        Demultiplexer(num_panda_positions * 2, num_panda_positions)
    )  # multiply by 2 because of position and velocity
    builder.Connect(plant.get_state_output_port(panda), demux.get_input_port())
    builder.ExportOutput(demux.get_output_port(0), f"{panda_name}.position_estimated")
    builder.ExportOutput(demux.get_output_port(1), f"{panda_name}.velocity_estimated")

    # create input port for panda position
    panda_position = builder.AddSystem(PassThrough(num_panda_positions))
    panda_position.set_name("panda_position")
    builder.ExportInput(panda_position.get_input_port(), f"{panda_name}.position")
    builder.ExportOutput(
        panda_position.get_output_port(), f"{panda_name}.position_commanded"
    )

    # create input port for panda hand position
    panda_hand = plant.GetModelInstanceByName(panda_hand_name)
    num_panda_hand_positions = plant.num_positions(panda_hand)

    panda_hand_position = builder.AddSystem(PassThrough(num_panda_hand_positions))
    panda_hand_position.set_name("panda_hand_position")
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
    demux = builder.AddSystem(
        Demultiplexer(num_panda_hand_positions * 2, num_panda_hand_positions)
    )  # multiply by 2 because of position and velocity
    builder.Connect(plant.get_state_output_port(panda_hand), demux.get_input_port())
    builder.ExportOutput(
        demux.get_output_port(0), f"{panda_hand_name}.position_estimated"
    )
    builder.ExportOutput(
        demux.get_output_port(1), f"{panda_hand_name}.velocity_estimated"
    )

    # add controller
    panda_positions = plant.num_positions()

    kp = [100] * panda_positions
    ki = [1] * panda_positions
    kd = [20] * panda_positions

    panda_controller = builder.AddSystem(
        InverseDynamicsController(plant, kp, ki, kd, has_reference_acceleration=False)
    )
    panda_controller.set_name("panda_controller")

    builder.Connect(
        plant.get_state_output_port(),
        # plant.get_state_output_port(panda),
        panda_controller.get_input_port_estimated_state(),
    )
    builder.Connect(
        panda_controller.get_output_port_control(), plant.get_actuation_input_port()
    )

    # connect desired position to controller
    num_plant_positions = plant.num_positions()
    desired_state_from_position = builder.AddSystem(
        StateInterpolatorWithDiscreteDerivative(
            num_plant_positions,
            time_step=timestep,
            suppress_initial_transient=True,
        )
    )

    builder.Connect(
        desired_state_from_position.get_output_port(),
        panda_controller.get_input_port_desired_state(),
    )

    multiplex = builder.AddSystem(
        Multiplexer([num_panda_positions, num_panda_hand_positions])
    )
    builder.Connect(panda_position.get_output_port(), multiplex.get_input_port(0))
    builder.Connect(panda_hand_position.get_output_port(), multiplex.get_input_port(1))

    builder.Connect(
        multiplex.get_output_port(),
        desired_state_from_position.get_input_port(),
    )

    # Build the diagram
    diagram = builder.Build()
    diagram.set_name("PandaManipulationStation")

    return diagram, plant

def visualize_diagram(diagram):
    """
    Util to visualize the system diagram
    """
    from IPython.display import SVG, display
    import pydot
    display(SVG(pydot.graph_from_dot_data(
        diagram.GetGraphvizString(max_depth=2))[0].create_svg()))