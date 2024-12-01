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
from debug import visualize_camera_images, visualize_depth_images, visualize_point_cloud, draw_grasp_candidate, draw_query_pts
import hydra
import plotly.express as px
from hydra.utils import get_original_cwd
from src.modules.kinematics import Planner, AddPandaDifferentialIK
import torch
from src.stations.setup_diagram import BuildPouringDiagram

class NoDiffIKWarnings(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith("Differential IK")


@hydra.main(config_path='src/config', config_name='pouring')
def pouring_demo(cfg: DictConfig) -> bool:
    meshcat = StartMeshcat()

    diagram, planner_system, visualizer = BuildPouringDiagram(meshcat, cfg)

    # debug: visualize merged point cloud
    # merge_point_clouds = diagram.GetSubsystemByName('merge_point_clouds')
    # context = merge_point_clouds.GetMyContextFromRoot(diagram.CreateDefaultContext())
    # pc = merge_point_clouds.GetOutputPort('point_cloud').Eval(context)
    # fig = px.scatter_3d(x = pc.xyzs()[0,:], y=pc.xyzs()[1,:], z=pc.xyzs()[2,:])
    # fig.show()

    # debug: save point cloud
    # np.save(f'{get_original_cwd()}/outputs/basket_merged_point_cloud.npy', pc.xyzs())

    simulator = Simulator(diagram)

    simulator.AdvanceTo(0.6)
    meshcat.Flush()  # Wait for the large object meshes to get to meshcat.
    meshcat.StartRecording()

    # run as fast as possible
    simulator.set_target_realtime_rate(0)
    meshcat.AddButton("Stop Simulation", "Escape")
    print("Press Escape to stop the simulation")

    print('Running Simulation')

    while meshcat.GetButtonClicks("Stop Simulation") < 1:
        if planner_system.is_done:
            break
        try: 
            if cfg.max_time != -1 and simulator.get_context().get_time() > cfg.max_time:
                raise Exception("Took too long")
            simulator.AdvanceTo(simulator.get_context().get_time() + 2.0)
        # stats = diagram.get_output_port().Eval(simulator.get_context())
        except Exception as e:
            print(f'Exception caught: {e}')
            break
    meshcat.StopRecording()
    meshcat.PublishRecording()
    meshcat.DeleteButton("Stop Simulation")
    return True

if __name__ == '__main__':
    logging.getLogger('drake').addFilter(NoDiffIKWarnings())

    # for reproducability. When testing grasp poses of the NDF disable.
    torch.manual_seed(86)
    np.random.seed(48)
    random.seed(72)

    pouring_demo()

