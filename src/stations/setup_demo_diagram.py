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
    PortSwitch
)
from omegaconf import DictConfig
from src.stations.teleop_station import MakePandaManipulationStation, get_directives, CreateArmOnlyPlant
from src.modules.perception import MergePointClouds, LNDFGrasper
import logging
import os
import random
import numpy as np
from manipulation.station import (
    DepthImageToPointCloud
)
import hydra
import plotly.express as px
from src.modules.kinematics import Planner, AddPandaDifferentialIK
import torch

def BuildPouringDiagram(meshcat: Meshcat, cfg: DictConfig) -> tuple[Diagram, Diagram, ]:
    directives_config = cfg['directives']
    lndf_config = cfg['lndf']

    object, obj_name, object_id, link_name = directives_config['object'], directives_config['obj_name'], \
                                            directives_config['object_id'], directives_config['link_name']


    builder = DiagramBuilder()

    robot_directives, env_directives = get_directives(directives_config, object=object, obj_name=obj_name, object_id=object_id, link_name=link_name)
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
            plant.GetModelInstanceByName(obj_name),

            # only above view cameras to get partial point clouds
            camera_body_indices=[
                plant.GetBodyIndices(plant.GetModelInstanceByName("camera0"))[0],
                plant.GetBodyIndices(plant.GetModelInstanceByName("camera1"))[0],
                plant.GetBodyIndices(plant.GetModelInstanceByName("camera2"))[0],
                # plant.GetBodyIndices(plant.GetModelInstanceByName("camera3"))[0], # bottom "cheat" camera
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

    visualizer = MeshcatVisualizer.AddToBuilder(
        builder, station.GetOutputPort("query_object"), meshcat
    )

    return builder.Build(), visualizer