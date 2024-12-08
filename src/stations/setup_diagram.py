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
    PortSwitch,
)
from omegaconf import DictConfig
from src.stations.teleop_station import (
    MakePandaManipulationStation,
    get_directives,
    CreateArmOnlyPlant,
)
from src.modules.perception import MergePointClouds, LNDFGrasper
import logging
import os
import random
import numpy as np
from manipulation.station import DepthImageToPointCloud
import hydra
import plotly.express as px
from src.modules.kinematics import Planner, AddPandaDifferentialIK
import torch


class NoDiffIKWarnings(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith("Differential IK")

def BuildPouringDiagram(meshcat: Meshcat, cfg: DictConfig) -> tuple[Diagram, Diagram, ]:
    directives_config = cfg['directives']
    lndf_config = cfg['lndf']
    SAM_config = cfg['SAM']
    object = directives_config['object']

    builder = DiagramBuilder()

    robot_directives, env_directives = get_directives(directives_config)
    panda_station = MakePandaManipulationStation(
        robot_directives=robot_directives,
        env_directives=env_directives,
        meshcat=meshcat,
    )
    station = builder.AddSystem(panda_station)

    plant = station.GetSubsystemByName("plant")

    # get cameras from station
    num_cameras = 3
    camera_body_indices = [
        plant.GetBodyIndices(plant.GetModelInstanceByName(f"camera{i}"))[0]
        for i in range(num_cameras)
    ]
    cameras = [station.GetSubsystemByName(f"camera{i}") for i in range(3)]

    merge_point_clouds = builder.AddNamedSystem(
        "merge_point_clouds",
        MergePointClouds(
            plant,
            plant.GetModelInstanceByName(object),

            # only above view cameras to get partial point clouds
            camera_body_indices=[
                plant.GetBodyIndices(plant.GetModelInstanceByName("camera0"))[0],
                plant.GetBodyIndices(plant.GetModelInstanceByName("camera1"))[0],
                plant.GetBodyIndices(plant.GetModelInstanceByName("camera2"))[0],
                # plant.GetBodyIndices(plant.GetModelInstanceByName("camera3"))[0], # bottom "cheat" camera
            ],
            cameras=cameras,
            meshcat=meshcat,
            object_prompt=SAM_config['object_prompt'],
            debug=False
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
        rgb_port = f"camera{i}_rgb_image"
        depth_port = f"camera{i}_depth_image"
        builder.Connect(
            panda_station.GetOutputPort(point_cloud_port),
            merge_point_clouds.GetInputPort(point_cloud_port),
        )
        builder.Connect(
            panda_station.GetOutputPort(rgb_port),
            merge_point_clouds.GetInputPort(rgb_port),
        )
        builder.Connect(
            panda_station.GetOutputPort(depth_port),
            merge_point_clouds.GetInputPort(depth_port),
        )


    # remove cheat ports
    # builder.Connect(
    #     panda_station.GetOutputPort("body_poses"),
    #     merge_point_clouds.GetInputPort("body_poses"),
    # )

    # Planner
    planner_system = Planner(meshcat, plant, directives_config['object'])
    planner = builder.AddSystem(planner_system)
    builder.Connect(
        panda_station.GetOutputPort("body_poses"), planner.GetInputPort("body_poses")
    )
    builder.Connect(
        grasper.GetOutputPort("grasp_pose"),
        planner.GetInputPort("obj_grasp")
    )

    # TODO: Fix Port names
    builder.Connect(
        station.GetOutputPort("panda_hand_state_estimated"),
        planner.GetInputPort("hand_state"),
    )
    builder.Connect(
        station.GetOutputPort("panda_arm_position_measured"),
        planner.GetInputPort("panda_position"),
    )

    # DiffIK
    time_step = plant.time_step()
    arm_only_plant = CreateArmOnlyPlant(time_step)

    diff_ik = AddPandaDifferentialIK(builder, arm_only_plant)
    builder.Connect(planner.GetOutputPort("X_WG"), diff_ik.GetInputPort("X_AE_desired"))
    builder.Connect(
        station.GetOutputPort("panda_arm_state_estimated"),
        diff_ik.GetInputPort("robot_state"),
    )
    builder.Connect(
        planner.GetOutputPort("reset_diff_ik"), diff_ik.GetInputPort("use_robot_state")
    )
    builder.Connect(
        planner.GetOutputPort("hand_position"),
        station.GetInputPort("panda_hand_position"),
    )

    # switch between direct control and diff ik
    switch = builder.AddSystem(PortSwitch(7))
    builder.Connect(diff_ik.get_output_port(), switch.DeclareInputPort("diff_ik"))
    builder.Connect(
        planner.GetOutputPort("panda_position_command"),
        switch.DeclareInputPort("position"),
    )
    builder.Connect(
        switch.get_output_port(), station.GetInputPort("panda_arm_position")
    )
    builder.Connect(
        planner.GetOutputPort("control_mode"), switch.get_port_selector_input_port()
    )

    # Debug: visualize camera images
    # visualize_camera_images(station)

    # Debug: visualize depth images
    # visualize_depth_images(station)

    visualizer = MeshcatVisualizer.AddToBuilder(
        builder, station.GetOutputPort("query_object"), meshcat
    )

    return builder.Build(), planner_system, visualizer
