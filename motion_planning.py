import argparse
from pydrake.all import (
    Meshcat,
    StartMeshcat,
    Simulator,
    DiagramBuilder,
    Diagram,
    MeshcatVisualizer,
    RigidTransform,
    RotationMatrix,
)
from omegaconf import DictConfig
from src.stations.teleop_station import MakePandaManipulationStation, get_directives
from src.modules.perception import MergePointClouds, LNDFGrasper
import logging
import os
import random
import numpy as np
from manipulation.station import DepthImageToPointCloud
from debug import (
    visualize_camera_images,
    visualize_depth_images,
    visualize_point_cloud,
    draw_grasp_candidate,
    draw_query_pts,
)
import hydra
import plotly.express as px
from hydra.utils import get_original_cwd
from src.kinematics import PandaGraspTrajectoryPlanner, CreatePandaTrajectoryController
from manipulation.utils import RenderDiagram


class NoDiffIKWarnings(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith("Differential IK")


def BuildPouringDiagram(meshcat: Meshcat, cfg: DictConfig) -> tuple[
    Diagram,
    Diagram,
]:
    directives_config = cfg["directives"]
    lndf_config = cfg["lndf"]

    builder = DiagramBuilder()

    robot_directives, env_directives = get_directives(directives_config)
    panda_station = MakePandaManipulationStation(
        robot_directives=robot_directives,
        env_directives=env_directives,
        meshcat=meshcat,
    )
    station = builder.AddSystem(panda_station)

    plant = station.GetSubsystemByName("plant")

    # default arm position
    plant_context = plant.CreateDefaultContext()
    panda_arm = plant.GetModelInstanceByName("panda_arm")

    merge_point_clouds = builder.AddNamedSystem(
        "merge_point_clouds",
        MergePointClouds(
            plant,
            plant.GetModelInstanceByName("basket"),
            camera_body_indices=[
                plant.GetBodyIndices(plant.GetModelInstanceByName("camera0"))[0],
                plant.GetBodyIndices(plant.GetModelInstanceByName("camera1"))[0],
                plant.GetBodyIndices(plant.GetModelInstanceByName("camera2"))[0],
                plant.GetBodyIndices(plant.GetModelInstanceByName("camera3"))[0],
            ],
            meshcat=meshcat,
        ),
    )

    grasper = builder.AddNamedSystem(
        "grasper", LNDFGrasper(lndf_config, plant, meshcat)
    )

    builder.Connect(
        merge_point_clouds.GetOutputPort("point_cloud"),
        grasper.GetInputPort("merged_point_cloud"),
    )

    for i in range(4):
        point_cloud_port = f"camera{i}_point_cloud"
        builder.Connect(
            panda_station.GetOutputPort(point_cloud_port),
            merge_point_clouds.GetInputPort(point_cloud_port),
        )

    builder.Connect(
        panda_station.GetOutputPort("body_poses"),
        merge_point_clouds.GetInputPort("body_poses"),
    )

    # trajectory planner
    trajectory_planner = builder.AddNamedSystem(
        "trajectory_planner", PandaGraspTrajectoryPlanner(plant, meshcat)
    )

    builder.Connect(
        grasper.GetOutputPort("grasp_pose"),
        trajectory_planner.GetInputPort("grasp_pose"),
    )

    # trajectory controller
    panda_arm = plant.GetModelInstanceByName("panda_arm")
    plant_context = plant.CreateDefaultContext()
    initial_panda_arm_position = plant.GetPositions(plant_context, panda_arm)
    trajectory_controller = builder.AddNamedSystem(
        "trajectory_controller",
        CreatePandaTrajectoryController(
            plant, initial_panda_arm_position=initial_panda_arm_position
        ),
    )

    builder.Connect(
        trajectory_planner.GetOutputPort("panda_arm_trajectory"),
        trajectory_controller.GetInputPort("panda_arm_trajectory"),
    )

    builder.Connect(
        station.GetOutputPort("panda_arm.position_estimated"),
        trajectory_controller.GetInputPort("panda_arm.position"),
    )

    builder.Connect(
        trajectory_controller.GetOutputPort("panda_arm.position_commanded"),
        station.GetInputPort("panda_arm.position"),
    )

    # Debug: visualize camera images
    # visualize_camera_images(station)

    # Debug: visualize depth images
    # visualize_depth_images(station)

    visualizer = MeshcatVisualizer.AddToBuilder(
        builder, station.GetOutputPort("query_object"), meshcat
    )

    return builder.Build(), None, visualizer


@hydra.main(config_path="src/config", config_name="pouring")
def pouring_demo(cfg: DictConfig) -> bool:
    meshcat = StartMeshcat()

    diagram, plant, visualizer = BuildPouringDiagram(meshcat, cfg)

    RenderDiagram(diagram)

    # debug: visualize merged point cloud
    # merge_point_clouds = diagram.GetSubsystemByName('merge_point_clouds')
    # context = merge_point_clouds.GetMyContextFromRoot(diagram.CreateDefaultContext())
    # pc = merge_point_clouds.GetOutputPort('point_cloud').Eval(context)
    # np.save(f'{get_original_cwd()}/outputs/basket_merged_point_cloud.npy', pc.xyzs())
    # fig = px.scatter_3d(x = pc.xyzs()[0,:], y=pc.xyzs()[1,:], z=pc.xyzs()[2,:])
    # fig.show()

    simulator = Simulator(diagram)

    simulator.AdvanceTo(0.6)
    meshcat.Flush()  # Wait for the large object meshes to get to meshcat.
    visualizer.StartRecording()

    # run as fast as possible
    simulator.set_target_realtime_rate(0)
    meshcat.AddButton("Stop Simulation", "Escape")
    print("Press Escape to stop the simulation")

    grasper = diagram.GetSubsystemByName("grasper")
    context = diagram.CreateDefaultContext()
    grasper_context = grasper.GetMyContextFromRoot(context)
    # grasp, final_query_pts = grasper.GetOutputPort('grasp_pose').Eval(grasper_context)
    # draw_grasp_candidate(meshcat, grasp)
    # draw_query_pts(meshcat, final_query_pts)

    # trajectory planner
    trajectory_planner = diagram.GetSubsystemByName("trajectory_planner")
    trajectory_planner_context = trajectory_planner.GetMyContextFromRoot(context)

    # draw pose of gripper in initial pose
    station = diagram.GetSubsystemByName("PandaManipulationStation")
    plant = station.GetSubsystemByName("plant")
    plant_context = plant.GetMyContextFromRoot(context)

    # model instances
    panda_arm = plant.GetModelInstanceByName("panda_arm")
    panda_hand = plant.GetModelInstanceByName("panda_hand")

    # bodies
    panda_gripper_body = plant.GetBodyByName("panda_link8")
    panda_hand_body = plant.GetBodyByName("panda_hand")

    plant_mutable_context = plant.GetMyMutableContextFromRoot(context)
    # plant.SetPositions(plant_mutable_context, panda_arm, [-1.57, 0.1, 0, -1.2, 0, 1.6, 0])

    # set initial integrator value
    integrator = diagram.GetSubsystemByName("trajectory_controller").GetSubsystemByName(
        "panda_arm_integrator"
    )
    integrator.set_integral_value(
        integrator.GetMyContextFromRoot(context),
        plant.GetPositions(plant_context, panda_arm),
    )

    print("Running Simulation")

    while meshcat.GetButtonClicks("Stop Simulation") < 1:
        if cfg.max_time != -1 and simulator.get_context().get_time() > cfg.max_time:
            raise Exception("Took too long")
        simulator.AdvanceTo(simulator.get_context().get_time() + 2.0)
        # stats = diagram.get_output_port().Eval(simulator.get_context())
        visualizer.StopRecording()
    visualizer.PublishRecording()
    meshcat.DeleteButton("Stop Simulation")
    return True


if __name__ == "__main__":
    logging.getLogger("drake").addFilter(NoDiffIKWarnings())

    pouring_demo()
