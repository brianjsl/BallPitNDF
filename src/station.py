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


def CreatePandaStation(
    model_directives: str,
    env_directives: str,
    meshcat: Meshcat,
    timestep: float = 1e-4,
    # timestep: float = 0.0,
    panda_name: str = "panda",
    panda_hand_name: str = "panda_hand",
):
    meshcat.Delete()

    # Initialize the builder and plant
    builder = DiagramBuilder()
    model_plant, model_scene_graph = AddMultibodyPlantSceneGraph(
        builder, time_step=timestep
    )
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=timestep)

    # Parse and process the directive
    # model_parser = Parser(model_plant)
    # ConfigureParser(model_parser)
    # model_parser.AddModelsFromString(env_directives, ".dmd.yaml")
    # model_parser.AddModelsFromString(model_directives, ".dmd.yaml")

    parser = Parser(plant)
    ConfigureParser(parser)
    parser.AddModelsFromString(env_directives, ".dmd.yaml")
    # parser.AddModelsFromString(model_directives, ".dmd.yaml")
    plant.Finalize()

    # model_plant = MultibodyPlant(timestep)
    model_parser = Parser(model_plant)
    ConfigureParser(model_parser)
    model_parser.AddModelsFromString(model_directives, ".dmd.yaml")
    model_plant.Finalize()

    # Finalize the plant and add visualization
    visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    visualizer = MeshcatVisualizer.AddToBuilder(builder, model_scene_graph, meshcat)

    # panda = plant.GetModelInstanceByName(panda_name)
    panda = model_plant.GetModelInstanceByName(panda_name)

    # export the estimated state, position, and velocity for panda from station
    builder.ExportOutput(
        plant.get_state_output_port(panda), f"{panda_name}.state_estimated"
    )

    # num_panda_positions = plant.num_positions(panda)
    num_panda_positions = model_plant.num_positions(panda)
    print("num_panda_positions", num_panda_positions)

    demux = builder.AddSystem(
        Demultiplexer(num_panda_positions * 2, num_panda_positions)
    )  # multiply by 2 because of position and velocity
    builder.Connect(model_plant.get_state_output_port(panda), demux.get_input_port())
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
    panda_hand = model_plant.GetModelInstanceByName(panda_hand_name)
    num_panda_hand_positions = model_plant.num_positions(panda_hand)

    panda_hand_position = builder.AddSystem(PassThrough(num_panda_hand_positions))
    panda_hand_position.set_name("panda_hand_position")
    builder.ExportInput(
        panda_hand_position.get_input_port(), f"{panda_hand_name}.position"
    )
    builder.ExportOutput(
        panda_hand_position.get_output_port(), f"{panda_hand_name}.position_commanded"
    )

    # export the estimated state, position, and velocity for panda hand from station
    num_panda_hand_positions = model_plant.num_positions(panda_hand)
    builder.ExportOutput(
        model_plant.get_state_output_port(panda_hand),
        f"{panda_hand_name}.state_estimated",
    )
    demux = builder.AddSystem(
        Demultiplexer(num_panda_hand_positions * 2, num_panda_hand_positions)
    )  # multiply by 2 because of position and velocity
    builder.Connect(
        model_plant.get_state_output_port(panda_hand), demux.get_input_port()
    )
    builder.ExportOutput(
        demux.get_output_port(0), f"{panda_hand_name}.position_estimated"
    )
    builder.ExportOutput(
        demux.get_output_port(1), f"{panda_hand_name}.velocity_estimated"
    )

    # add controller
    panda_positions = plant.num_positions()

    # kp = [1] * 9  # replace with variable
    # ki = [1] * 9
    # kd = [1] * 9

    kp = [100] * 9  # replace with variable
    ki = [1] * 9
    kd = [20] * 9

    # kp = [100] * panda_positions
    # ki = [1] * panda_positions
    # kd = [20] * panda_positions

    panda_controller = builder.AddSystem(
        InverseDynamicsController(
            model_plant, kp, ki, kd, has_reference_acceleration=False
        )
    )

    print(dir(panda_controller))

    panda_controller.set_name("panda_controller")

    print(model_plant)

    panda = model_plant.GetModelInstanceByName(panda_name)
    panda_hand = model_plant.GetModelInstanceByName(panda_hand_name)
    num_panda_positions = model_plant.num_positions(panda)
    num_panda_hand_positions = model_plant.num_positions(panda_hand)

    a = builder.AddSystem(
        Multiplexer([num_panda_positions * 2, num_panda_hand_positions * 2])
    )
    builder.Connect(model_plant.get_state_output_port(panda), a.get_input_port(0))
    builder.Connect(model_plant.get_state_output_port(panda_hand), a.get_input_port(1))

    builder.Connect(
        # plant.get_state_output_port(),
        a.get_output_port(),
        panda_controller.get_input_port_estimated_state(),
    )
    builder.Connect(
        panda_controller.get_output_port_control(),
        model_plant.get_actuation_input_port(),
    )

    # connect desired position to controller
    num_plant_positions = model_plant.num_positions()
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

    return diagram


# def CreatePandaStation(
#     model_directives: str,
#     env_directives: str,
#     meshcat: Meshcat,
#     timestep: float = 1e-4,
#     panda_name: str = "panda",
#     panda_hand_name: str = "panda_hand",
# ):
#     meshcat.Delete()

#     # Initialize the builder and plant
#     builder = DiagramBuilder()
#     plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=timestep)

#     # Parse and process the directive
#     parser = Parser(plant)
#     ConfigureParser(parser)

#     parser.AddModelsFromString(env_directives, ".dmd.yaml")


#     model_plant = MultibodyPlant(timestep)
#     model_parser = Parser(model_plant)
#     ConfigureParser(model_parser)
#     model_parser.AddModelsFromString(model_directives, ".dmd.yaml")

#     # Finalize the plant and add visualization
#     plant.Finalize()
#     visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

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

#     return diagram
