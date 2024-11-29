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
from manipulation.meshcat_utils import AddMeshcatTriad
from src.modules.motion_planning.trajectory import (
    TrajectoryPlanner,
    PandaTrajectoryEvaluator,
)


class NoDiffIKWarnings(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith("Differential IK")


DEBUG_GRASP_POSE = RigidTransform(
    R=RotationMatrix(
        [
            [-0.24274149537086487, 1.7477315664291382, 0.9415445923805237],
            [0.891293466091156, 0.943527340888977, -1.521623969078064],
            [-1.773882508277893, 0.2349160611629486, -0.8933901190757751],
        ]
    ),
    p=[0.4168945953249931, 0.21598514015786352, 0.5206834465265274],
)


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
        time_step=5e-4,
    )
    station = builder.AddSystem(panda_station)

    plant = station.GetSubsystemByName("plant")

    # default arm position
    plant_context = plant.CreateDefaultContext()
    panda_arm = plant.GetModelInstanceByName("panda_arm")

    # merge_point_clouds = builder.AddNamedSystem(
    #     "merge_point_clouds",
    #     MergePointClouds(
    #         plant,
    #         plant.GetModelInstanceByName("basket"),
    #         camera_body_indices=[
    #             plant.GetBodyIndices(plant.GetModelInstanceByName("camera0"))[0],
    #             plant.GetBodyIndices(plant.GetModelInstanceByName("camera1"))[0],
    #             plant.GetBodyIndices(plant.GetModelInstanceByName("camera2"))[0],
    #             plant.GetBodyIndices(plant.GetModelInstanceByName("camera3"))[0],
    #         ],
    #         meshcat=meshcat,
    #     ),
    # )

    # grasper = builder.AddNamedSystem(
    #     "grasper", LNDFGrasper(lndf_config, plant, meshcat, debug_pose=DEBUG_GRASP_POSE)
    # )

    # builder.Connect(
    #     merge_point_clouds.GetOutputPort("point_cloud"),
    #     grasper.GetInputPort("merged_point_cloud"),
    # )

    # for i in range(4):
    #     point_cloud_port = f"camera{i}_point_cloud"
    #     builder.Connect(
    #         panda_station.GetOutputPort(point_cloud_port),
    #         merge_point_clouds.GetInputPort(point_cloud_port),
    #     )

    # builder.Connect(
    #     panda_station.GetOutputPort("body_poses"),
    #     merge_point_clouds.GetInputPort("body_poses"),
    # )

    # trajectory planner
    controller_plant = station.GetSubsystemByName(
        "panda_controller"
    ).get_multibody_plant_for_control()
    trajectory_planner = builder.AddNamedSystem(
        "trajectory_planner",
        TrajectoryPlanner(controller_plant, meshcat),
    )

    # builder.Connect(
    #     grasper.GetOutputPort("grasp_pose"),
    #     trajectory_planner.GetInputPort("grasp_pose"),
    # )

    trajectory_evaluator = builder.AddNamedSystem(
        "trajectory_evaluator", PandaTrajectoryEvaluator()
    )
    builder.Connect(
        trajectory_planner.GetOutputPort("panda_arm_trajectory"),
        trajectory_evaluator.GetInputPort("trajectory"),
    )

    builder.Connect(
        trajectory_evaluator.GetOutputPort("panda_arm_q"),
        station.GetInputPort("panda_arm.position"),
    )

    builder.Connect(
        trajectory_evaluator.GetOutputPort("panda_hand_q"),
        station.GetInputPort("panda_hand.position"),
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
    context = diagram.CreateDefaultContext()

    # print trajectory
    trajectory_planner = diagram.GetSubsystemByName("trajectory_planner")
    trajectory_planner_context = trajectory_planner.GetMyContextFromRoot(context)
    trajectory_planner.GetInputPort("grasp_pose").FixValue(
        trajectory_planner_context, (DEBUG_GRASP_POSE, None)
    )

    # panda_arm_traj = trajectory_planner.GetOutputPort("panda_arm_trajectory").Eval(
    #     trajectory_planner_context
    # )

    # print("panda_arm_traj: ", panda_arm_traj)

    # RenderDiagram(diagram)

    # debug: visualize merged point cloud
    # merge_point_clouds = diagram.GetSubsystemByName('merge_point_clouds')
    # context = merge_point_clouds.GetMyContextFromRoot(diagram.CreateDefaultContext())
    # pc = merge_point_clouds.GetOutputPort('point_cloud').Eval(context)
    # np.save(f'{get_original_cwd()}/outputs/basket_merged_point_cloud.npy', pc.xyzs())
    # fig = px.scatter_3d(x = pc.xyzs()[0,:], y=pc.xyzs()[1,:], z=pc.xyzs()[2,:])
    # fig.show()

    simulator = Simulator(diagram, context)
    simulator.set_target_realtime_rate(1)

    meshcat.StartRecording()
    simulator.AdvanceTo(1.0)
    meshcat.PublishRecording()

    import time

    while True:
        time.sleep(1)

    return True


if __name__ == "__main__":
    logging.getLogger("drake").addFilter(NoDiffIKWarnings())

    pouring_demo()
