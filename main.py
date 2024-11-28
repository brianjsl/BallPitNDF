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
import plotly.express as px
from hydra.utils import get_original_cwd
from src.modules.kinematics import Planner, AddPandaDifferentialIK

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
            plant.GetModelInstanceByName("basket"),

            # only above view cameras to get partial point clouds
            camera_body_indices=[
                plant.GetBodyIndices(plant.GetModelInstanceByName("camera0"))[0],
                plant.GetBodyIndices(plant.GetModelInstanceByName("camera1"))[0],
                plant.GetBodyIndices(plant.GetModelInstanceByName("camera2"))[0],
                # plant.GetBodyIndices(plant.GetModelInstanceByName("camera3"))[0], # remove bottom "cheat" camera
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

    # Planner
    planner = builder.AddSystem(Planner(meshcat, plant))
    builder.Connect(
        panda_station.GetOutputPort("body_poses"),
        planner.GetInputPort("body_poses")
    )
    builder.Connect(
        grasper.GetOutputPort("grasp_pose"),
        planner.GetInputPort("basket_grasp")
    )

    #TODO: Fix Port names
    builder.Connect(
        station.GetOutputPort("panda_hand_state_estimated"),
        planner.GetInputPort("hand_state")
    )
    builder.Connect(
        station.GetOutputPort("panda_arm_position_commanded"),
        planner.GetInputPort("panda_position")
    )

    # builder.Connect(
    #     station.GetOutputPort("panda_arm_torque_external"),
    #     planner.GetInputPort("external_torque")
    # )

    robot = station.GetSubsystemByName("panda_controller").get_multibody_plant_for_control()

    # DiffIK
    diff_ik = AddPandaDifferentialIK(builder, robot)
    builder.Connect(
        planner.GetOutputPort("X_WG"),
        diff_ik.GetInputPort(0)
    )
    builder.Connect(
        station.GetOutputPort("panda_arm_state_estimated"),
        diff_ik.GetInputPort("robot_state")
    )
    builder.Connect(planner.GetOutputPort("reset_diff_ik"),
                    diff_ik.GetInputPort("use_robot_state"))
    builder.Connect(planner.GetOutputPort("hand_position"),
                    station.GetInputPort("panda_hand_position"))
    
    # switch between direct control and diff ik
    switch = builder.AddSystem(PortSwitch(7))
    builder.Connect(
        diff_ik.get_output_port(),
        switch.DeclareInputPort("diff_ik")
    )
    builder.Connect(planner.GetOutputPort("panda_position_command"),
                    switch.DeclareInputPort("position"))
    builder.Connect(switch.get_output_port(),
                    station.GetInputPort("panda_arm_position"))
    builder.Connect(planner.GetOutputPort("control_mode"),
                    switch.get_port_selector_input_port())

    #Debug: visualize camera images
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

    # draw pose of gripper in initial pose
    station = diagram.GetSubsystemByName("PandaManipulationStation")

    print('Running Simulation')

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

