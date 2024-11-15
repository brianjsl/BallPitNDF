import argparse
from pydrake.all import (
    Meshcat, StartMeshcat, Simulator, DiagramBuilder, Diagram, MeshcatVisualizer
)
from omegaconf import DictConfig
from src.setup import MakePandaManipulationStation
from src.modules.perception import MergePointClouds
import logging
import os
import random
import numpy as np
from manipulation.station import (
    DepthImageToPointCloud
)
from debug import visualize_camera_images, visualize_depth_images, visualize_point_cloud

class NoDiffIKWarnings(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith("Differential IK")

def get_directives(cfg: DictConfig) -> tuple[str, str]:
    # description of robot
    robot_directives = """
directives:
    - add_model:
        name: panda_arm
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
        child: panda_arm::panda_link0
    - add_model:
        name: panda_hand
        file: package://drake_models/franka_description/urdf/panda_hand.urdf
    - add_weld:
        parent: panda_arm::panda_link8
        child: panda_hand::panda_hand
        X_PC:
            translation: [0, 0, 0]
            rotation: !Rpy { deg: [0, 0, -45] }
"""

    # description of objects in env
    env_directives = f"""
directives:
    - add_model:
        name: floor
        file: package://manipulation/floor.sdf
    - add_weld:
        parent: world
        child: floor::box
        X_PC:
            translation: [0, 0, -0.05]
    - add_model:
        name: basket
        file: file://{os.getcwd()}/src/assets/basket/basket.sdf

    - add_frame:
        name: camera0_origin
        X_PF:
            base_frame: world
            rotation: !Rpy {{deg: [-130.0, 0, 0.0]}}
            translation: [.5, -.6, 0.8]

    - add_model:
        name: camera0
        file: package://manipulation/camera_box.sdf

    - add_weld:
        parent: camera0_origin
        child: camera0::base

    - add_frame:
        name: camera1_origin
        X_PF:
            base_frame: world
            rotation: !Rpy {{deg: [-140., 0, 90.0]}}
            translation: [1.0, 0, 0.8]

    - add_model:
        name: camera1
        file: package://manipulation/camera_box.sdf

    - add_weld:
        parent: camera1_origin
        child: camera1::base

    - add_frame:
        name: camera2_origin
        X_PF:
            base_frame: world
            rotation: !Rpy {{deg: [-130., 0, -180.0]}}
            translation: [.5, .7, .8]

    - add_model:
        name: camera2
        file: package://manipulation/camera_box.sdf

    - add_weld:
        parent: camera2_origin
        child: camera2::base
"""
    for i in range(cfg.num_balls):
        env_directives += f"""
    - add_model:
        name: ball_{i}
        file: file://{os.getcwd()}/src/assets/sphere/sphere.sdf
        default_free_body_pose:
            sphere:
                translation: [{0.5 + np.random.choice([-1,1])*0.03 + 0.1*(random.random()-0.5)}, {np.random.choice([-1,1])*0.03 + 0.1*(random.random()-0.5)}, 0.5]
"""
    return robot_directives, env_directives

def BuildPouringDiagram(meshcat: Meshcat, cfg: DictConfig) -> tuple[Diagram, Diagram, ]:
    builder = DiagramBuilder()

    robot_directives, env_directives = get_directives(cfg)
    panda_station = MakePandaManipulationStation(
        robot_directives=robot_directives,
        env_directives=env_directives,
        meshcat=meshcat
    )
    station = builder.AddSystem(panda_station)
    
    plant = station.GetSubsystemByName('plant')

    merge_point_clouds = builder.AddNamedSystem(
        'merge_point_clouds',
        MergePointClouds(plant, 
                         plant.GetModelInstanceByName('basket'),
                         camera_body_indices=[
                             plant.GetBodyIndices(
                                 plant.GetModelInstanceByName("camera0"))[0],
                             plant.GetBodyIndices(
                                 plant.GetModelInstanceByName("camera1"))[0],
                             plant.GetBodyIndices(
                                 plant.GetModelInstanceByName("camera2"))[0],
                        ],
                        meshcat=meshcat
        )
    )


    for i in range(3):
        point_cloud_port = f"camera{i}_point_cloud"
        builder.Connect(panda_station.GetOutputPort(point_cloud_port),
                        merge_point_clouds.GetInputPort(point_cloud_port))
    
    builder.Connect(panda_station.GetOutputPort("body_poses"),
                    merge_point_clouds.GetInputPort("body_poses"))
    
    # Debug: visualize camera images
    # visualize_camera_images(station)

    # Debug: visualize depth images
    # visualize_depth_images(station)

    visualizer = MeshcatVisualizer.AddToBuilder(
        builder, station.GetOutputPort("query_object"), meshcat)

    return builder.Build(), None, visualizer 

def pouring_demo(cfg: DictConfig, meshcat: Meshcat) -> bool:

    meshcat.Delete()
    diagram, plant, visualizer = BuildPouringDiagram(meshcat, cfg)

    #debug
    merge_point_clouds = diagram.GetSubsystemByName('merge_point_clouds')
    context = merge_point_clouds.GetMyContextFromRoot(diagram.CreateDefaultContext())
    pc = merge_point_clouds.GetOutputPort('point_cloud').Eval(context)
    visualize_point_cloud(pc.xyzs())

    simulator = Simulator(diagram)

    simulator.AdvanceTo(0.1)
    meshcat.Flush()  # Wait for the large object meshes to get to meshcat.
    visualizer.StartRecording()

    # run as fast as possible
    simulator.set_target_realtime_rate(0)
    meshcat.AddButton("Stop Simulation", "Escape")
    print("Press Escape to stop the simulation")
    while meshcat.GetButtonClicks("Stop Simulation") < 1:
        if cfg.max_time and simulator.get_context().get_time() > cfg.max_time:
            raise Exception("Took too long")
        simulator.AdvanceTo(simulator.get_context().get_time() + 2.0)
        # stats = diagram.get_output_port().Eval(simulator.get_context())
        visualizer.StopRecording()
    visualizer.PublishRecording()
    meshcat.DeleteButton("Stop Simulation")
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Local NDF Pouring Robot'
    )
    parser.add_argument('-n', help='Number of balls to add to box', default=20)
    parser.add_argument('-m', help='Basket ID to use', default=1)
    args = parser.parse_args()
    cfg = DictConfig({
        # 'num_balls': args.n,
        'num_balls': 5,
        'basket_id': args.m,
        'max_time': 60.0,
    })

    logging.getLogger('drake').addFilter(NoDiffIKWarnings())

    meshcat = StartMeshcat()
    pouring_demo(cfg, meshcat)
