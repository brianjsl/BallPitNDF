import argparse
from pydrake.all import (
    Meshcat,
    StartMeshcat,
    Simulator,
    DiagramBuilder,
    Diagram,
    MeshcatVisualizer,
)
from omegaconf import DictConfig
from src.stations.teleop_station import MakePandaManipulationStation, get_directives
from src.modules.perception import MergePointClouds, LNDFGrasper
import logging
import os
import random
import numpy as np
from manipulation.station import (
    DepthImageToPointCloud
)
from debug import visualize_camera_images, visualize_depth_images, visualize_point_cloud, draw_grasp_candidate, draw_query_pts
import hydra

class NoDiffIKWarnings(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith("Differential IK")

def BuildPouringDiagram(meshcat: Meshcat, cfg: DictConfig) -> tuple[Diagram, Diagram, ]:
    directives_config = cfg['directives']
    lndf_config = cfg['lndf']

    builder = DiagramBuilder()

    robot_directives, env_directives = get_directives(directives_config)
    panda_station = MakePandaManipulationStation(
        robot_directives=robot_directives,
        env_directives=env_directives,
        meshcat=meshcat,
    )
    station = builder.AddSystem(panda_station)

    plant = station.GetSubsystemByName("plant")

    merge_point_clouds = builder.AddNamedSystem(
        "merge_point_clouds",
        MergePointClouds(
            plant,
            plant.GetModelInstanceByName("bowl"),
            camera_body_indices=[
                plant.GetBodyIndices(plant.GetModelInstanceByName("camera0"))[0],
                plant.GetBodyIndices(plant.GetModelInstanceByName("camera1"))[0],
                plant.GetBodyIndices(plant.GetModelInstanceByName("camera2"))[0],
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

    for i in range(3):
        point_cloud_port = f"camera{i}_point_cloud"
        builder.Connect(
            panda_station.GetOutputPort(point_cloud_port),
            merge_point_clouds.GetInputPort(point_cloud_port),
        )

    builder.Connect(
        panda_station.GetOutputPort("body_poses"),
        merge_point_clouds.GetInputPort("body_poses"),
    )

    # Debug: visualize camera images
    # visualize_camera_images(station)

    # Debug: visualize depth images
    # visualize_depth_images(station)

    visualizer = MeshcatVisualizer.AddToBuilder(
        builder, station.GetOutputPort("query_object"), meshcat
    )

    return builder.Build(), None, visualizer


@hydra.main(config_path='src/config', config_name='pouring')
def pouring_demo(cfg: DictConfig) -> bool:
    meshcat = StartMeshcat()

    diagram, plant, visualizer = BuildPouringDiagram(meshcat, cfg)

    # debug

    # merge_point_clouds = diagram.GetSubsystemByName('merge_point_clouds')
    # context = merge_point_clouds.GetMyContextFromRoot(diagram.CreateDefaultContext())
    # pc = merge_point_clouds.GetOutputPort('point_cloud').Eval(context)
    # np.save('outputs/point_cloud.npy', pc.xyzs())

    simulator = Simulator(diagram)

    simulator.AdvanceTo(0.4)
    meshcat.Flush()  # Wait for the large object meshes to get to meshcat.
    visualizer.StartRecording()


    # run as fast as possible
    simulator.set_target_realtime_rate(0)
    meshcat.AddButton("Stop Simulation", "Escape")
    print("Press Escape to stop the simulation")


    grasper = diagram.GetSubsystemByName('grasper')
    context = grasper.GetMyContextFromRoot(diagram.CreateDefaultContext())
    grasp, final_query_pts = grasper.GetOutputPort('grasp_pose').Eval(context)
    draw_grasp_candidate(meshcat, grasp)
    draw_query_pts(meshcat, final_query_pts)

    while meshcat.GetButtonClicks("Stop Simulation") < 1:
        if cfg.max_time != -1 and simulator.get_context().get_time() > cfg.max_time:
            raise Exception("Took too long")
        simulator.AdvanceTo(simulator.get_context().get_time() + 2.0)
        # stats = diagram.get_output_port().Eval(simulator.get_context())
        visualizer.StopRecording()
    visualizer.PublishRecording()
    meshcat.DeleteButton("Stop Simulation")
    return True

if __name__ == '__main__':
    logging.getLogger('drake').addFilter(NoDiffIKWarnings())

    pouring_demo()

