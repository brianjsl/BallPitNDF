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
from src.modules.kinematics import Planner, AddPandaDifferentialIK
import torch
from src.stations.setup_diagram import BuildPouringDiagram
from tqdm import tqdm


class NoDiffIKWarnings(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith("Differential IK")


@hydra.main(config_path="src/config", config_name="pouring")
def run_pouring_experiments(cfg: DictConfig) -> bool:
    assert cfg.num_runs > 0

    meshcat = StartMeshcat()

    num_success_grasp = 0
    num_success_pour = 0
    object_name = cfg.directives.object
    for i in tqdm(range(cfg.num_runs), position=0):

        # run simulation for `duration`
        try:
            diagram, context, meshcat = pouring_demo(cfg, meshcat, cfg.duration)
            
            # get plant
            station = diagram.GetSubsystemByName("PandaManipulationStation")
            plant = station.GetSubsystemByName("plant")
            plant_context = plant.GetMyContextFromRoot(context)

            # get position of object in world frame
            obj_body = plant.GetBodyByName(f"{object_name}_body_link")
            obj_pose = plant.EvalBodyPoseInWorld(plant_context, obj_body)
            obj_z = obj_pose.translation()[2]

            if obj_z >= 0:
                num_success_grasp += 1

            # get position of ball in world frame
            ball_body = plant.GetBodyByName("sphere_body_link")
            ball_pose = plant.EvalBodyPoseInWorld(plant_context, ball_body)
            ball_position = ball_pose.translation()

            # get corners of ball pit in world frame
            ballpit_body = plant.GetBodyByName("bin_base")
            X_WB = plant.EvalBodyPoseInWorld(plant_context, ballpit_body)
            length = 0.63
            width = 0.49
            height = 0.07

            upper_left_local = np.array([-width / 2, length / 2, height])
            lower_right_local = np.array([width / 2, -length / 2, 0])

            # translate ball position to ball pit frame
            X_Ballpit_Ball = X_WB.inverse().multiply(ball_position)

            # check if ball in ball pit
            in_ballpit = (
                upper_left_local[0] <= X_Ballpit_Ball[0] <= lower_right_local[0]
                and upper_left_local[1] >= X_Ballpit_Ball[1] >= lower_right_local[1]
            )

            if in_ballpit:
                num_success_pour += 1

            tqdm.write(f"[Run {i}] Ball in ball pit: {in_ballpit}")
            with open(f"{object_name}_accuracy.txt", "a") as f:
                f.write(f"[Run {i}] Ball in ball pit: {in_ballpit}\n")

        except Exception as e:
            with open(f"{object_name}_accuracy.txt", "a") as f:
                f.write(f"[Run {i}] Crashed due to Error: {e}\n")

    pour_accuracy = num_success_pour / cfg.num_runs
    grasp_accuracy = num_success_grasp / cfg.num_runs

    print(f"Pour Accuracy: {pour_accuracy}")
    print(f"Grasp Accuracy: {grasp_accuracy}")

    # save to txt file
    with open(f"{object_name}_accuracy.txt", "a") as f:
        f.write(f"Pour Accuracy: {pour_accuracy}\n")
        f.write(f"Grasp Accuracy: {grasp_accuracy}\n")

    meshcat.Delete()


def pouring_demo(cfg: DictConfig, meshcat, duration: int) -> bool:
    diagram, _, _ = BuildPouringDiagram(meshcat, cfg)
    context = diagram.CreateDefaultContext()
    simulator = Simulator(diagram, context)
    simulator.AdvanceTo(duration)

    return diagram, context, meshcat


if __name__ == "__main__":
    logging.getLogger("drake").addFilter(NoDiffIKWarnings())

    # for reproducability. When testing grasp poses of the NDF disable.
    # torch.manual_seed(86)
    # np.random.seed(48)
    # random.seed(72)

    run_pouring_experiments()
