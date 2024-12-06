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


class NoDiffIKWarnings(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith("Differential IK")


@hydra.main(config_path="src/config", config_name="pouring")
def run_pouring_experiments(cfg: DictConfig) -> bool:
    assert cfg.num_runs > 0

    num_success = 0
    for i in range(cfg.num_runs):
        # run simulation for `duration`
        diagram, context = pouring_demo(cfg, cfg.duration)

        # get plant
        station = diagram.GetSubsystemByName("PandaManipulationStation")
        plant = station.GetSubsystemByName("plant")
        plant_context = plant.GetMyContextFromRoot(context)

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
        print("upper_left_local: ", upper_left_local)
        print("lower_right_local: ", lower_right_local)
        print("X_Ballpit_Ball: ", X_Ballpit_Ball)

        # check if ball in ball pit
        in_ballpit = (
            upper_left_local[0] <= X_Ballpit_Ball[0] <= lower_right_local[0]
            and upper_left_local[1] >= X_Ballpit_Ball[1] >= lower_right_local[1]
        )

        if in_ballpit:
            num_success += 1

        print(f"[Run {i}] Ball in ball pit: {in_ballpit}")

    accuracy = num_success / cfg.num_runs
    print(f"Accuracy: {accuracy}")

    # save to txt file
    with open("accuracy.txt", "w") as f:
        f.write(f"Accuracy: {accuracy}")


def pouring_demo(cfg: DictConfig, duration: int) -> bool:
    meshcat = StartMeshcat()

    diagram, planner_system, visualizer = BuildPouringDiagram(meshcat, cfg)
    context = diagram.CreateDefaultContext()
    simulator = Simulator(diagram, context)
    simulator.AdvanceTo(duration)

    return diagram, context


if __name__ == "__main__":
    logging.getLogger("drake").addFilter(NoDiffIKWarnings())

    # for reproducability. When testing grasp poses of the NDF disable.
    torch.manual_seed(86)
    np.random.seed(48)
    random.seed(72)

    run_pouring_experiments()
