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
    PiecewisePose,
    PiecewisePolynomial,
    TrajectorySource,
    Integrator,
)
from manipulation.meshcat_utils import AddMeshcatTriad
from omegaconf import DictConfig
from src.setup import MakePandaManipulationStation
from src.modules.perception import MergePointClouds, LNDFGrasper
from src.kinematics.diff_ik import PandaDiffIKController
import logging
import os
import random
import numpy as np
from manipulation.station import DepthImageToPointCloud
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
        default_free_body_pose:
            basket:
                translation: [-0.5, 0.0, 0.0]
                # translation: [-0.25, -0.25, 0.0]

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


    - add_model:
        name: bin
        file: package://manipulation/hydro/bin.sdf

    - add_weld:
        parent: world
        child: bin::bin_base
        X_PC:
            translation: [0.0004596900773833712, -0.5772629832585529, 0]

"""
    for i in range(cfg.num_balls):
        env_directives += f"""
    - add_model:
        name: ball_{i}
        file: file://{os.getcwd()}/src/assets/sphere/sphere.sdf
        default_free_body_pose:
            sphere:
                translation: [{-0.5}, {-0.0}, 0.05]
                # translation: [{-0.23}, {-0.23}, 0.05]
"""
    return robot_directives, env_directives


def MakePouringStation(meshcat: Meshcat, cfg: DictConfig):
    builder = DiagramBuilder()
    robot_directives, env_directives = get_directives(cfg)
    panda_station = MakePandaManipulationStation(
        robot_directives=robot_directives,
        env_directives=env_directives,
        meshcat=meshcat,
    )
    station = builder.AddSystem(panda_station)

    plant = station.GetSubsystemByName("plant")

    # model instances
    panda_arm = plant.GetModelInstanceByName("panda_arm")
    panda_hand = plant.GetModelInstanceByName("panda_hand")
    basket = plant.GetModelInstanceByName("basket")

    # bodies
    panda_gripper_body = plant.GetBodyByName("panda_link8")
    panda_hand_body = plant.GetBodyByName("panda_hand")
    basket_body = plant.GetBodyByName("basket")

    # body frames
    panda_hand_frame = panda_hand_body.body_frame()
    gripper_frame = panda_gripper_body.body_frame()
    basket_frame = basket_body.body_frame()

    context = plant.CreateDefaultContext()
    plant.SetPositions(context, panda_arm, [-1.57, 0.1, 0, -1.2, 0, 1.6, 0])

    # demo trajectory
    X_WB = plant.CalcRelativeTransform(
        context, frame_A=plant.world_frame(), frame_B=basket_frame
    )
    X_WBasket = plant.EvalBodyPoseInWorld(context, basket_body)
    X_WGinit = plant.EvalBodyPoseInWorld(context, panda_hand_body)

    X_BPreGrasp = RigidTransform(
        p=[0, -0.03, 0.7], R=RotationMatrix.MakeXRotation(np.pi)
    )
    X_WBPreGrasp = X_WB.multiply(X_BPreGrasp)

    X_BGrasp = RigidTransform(
        p=[0, -0.03, 0.305], R=RotationMatrix.MakeXRotation(np.pi)
    )
    X_WBGrasp = X_WB.multiply(X_BGrasp)

    # X_WGoal = X_WG.multiply(RigidTransform(R=RotationMatrix.MakeXRotation(np.pi / 4)))
    # X_WGoal = X_WG.multiply(RigidTransform())
    # X_WGoal = X_WG.multiply(RigidTransform(R=RotationMatrix.MakeYRotation(np.pi / 1.75)))
    X_WG = plant.CalcRelativeTransform(
        context, frame_A=plant.world_frame(), frame_B=gripper_frame
    )
    X_WGoal = X_WG.multiply(RigidTransform(R=RotationMatrix.MakeYRotation(np.pi / 4)))

    # X_WGoal = X_WG.multiply(RigidTransform(R=RotationMatrix.MakeYRotation(np.pi / 1.75)))
    # X_WGoal = X_WBPreGrasp.multiply(
    #     RigidTransform(R=RotationMatrix.MakeXRotation(np.pi / 1.9))
    # )

    initial_pose = plant.EvalBodyPoseInWorld(context, panda_hand_body)

    print(initial_pose)

    AddMeshcatTriad(meshcat, path="/X_WBPreGrasp", X_PT=X_WBPreGrasp)
    AddMeshcatTriad(meshcat, path="/X_WBGrasp", X_PT=X_WBGrasp)
    AddMeshcatTriad(meshcat, path="/X_WBasket", X_PT=X_WBasket)
    AddMeshcatTriad(meshcat, path="/X_WGinit", X_PT=initial_pose)

    # create trajectory
    trajectory = PiecewisePose.MakeLinear(
        times=[0, 5.0, 10.0, 13.0, 17.0, 20.0, 25.0],
        poses=[
            X_WGinit,
            X_WBPreGrasp,
            X_WBGrasp,
            X_WBGrasp,
            X_WBPreGrasp,
            X_WGinit,
            X_WGoal,
        ],
    )
    traj_V_G = trajectory.MakeDerivative()

    times = [0, 10.0, 12.0]
    initial_positions = np.array([0.0, 0.0])
    final_positions2 = np.array([0, 0])
    final_positions = np.array([-0.1, 0.1])
    positions = np.column_stack([initial_positions, final_positions, final_positions2])

    gripper_trajectory = PiecewisePolynomial.FirstOrderHold(times, positions)

    # add psuedo inverse controller
    wsg_source = builder.AddSystem(TrajectorySource(gripper_trajectory))
    builder.Connect(
        wsg_source.get_output_port(0), station.GetInputPort("panda_hand.position")
    )

    diff_ik = builder.AddSystem(PandaDiffIKController(plant=plant))
    diff_ik.set_name("PseudoInverseController")

    integrator = builder.AddNamedSystem("integrator", Integrator(7))
    trajectory_source = builder.AddSystem(TrajectorySource(traj_V_G))

    builder.Connect(
        trajectory_source.get_output_port(0), diff_ik.GetInputPort("spatial_velocity")
    )
    builder.Connect(
        station.GetOutputPort("panda_arm.position_estimated"), diff_ik.GetInputPort("q")
    )

    builder.Connect(diff_ik.get_output_port(0), integrator.get_input_port(0))
    builder.Connect(
        integrator.get_output_port(0), station.GetInputPort("panda_arm.position")
    )

    visualizer = MeshcatVisualizer.AddToBuilder(
        builder, station.GetOutputPort("query_object"), meshcat
    )

    diagram = builder.Build()
    diagram.set_name("PandaFunzo")

    return (diagram, None, visualizer)


def BuildPouringDiagram(meshcat: Meshcat, cfg: DictConfig) -> tuple[
    Diagram,
    Diagram,
]:
    builder = DiagramBuilder()

    robot_directives, env_directives = get_directives(cfg)
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
            camera_body_indices=[
                plant.GetBodyIndices(plant.GetModelInstanceByName("camera0"))[0],
                plant.GetBodyIndices(plant.GetModelInstanceByName("camera1"))[0],
                plant.GetBodyIndices(plant.GetModelInstanceByName("camera2"))[0],
            ],
            meshcat=meshcat,
        ),
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


def pouring_demo(cfg: DictConfig, meshcat: Meshcat) -> bool:

    meshcat.Delete()
    diagram, plant, visualizer = MakePouringStation(meshcat, cfg)
    # diagram, plant, visualizer = BuildPouringDiagram(meshcat, cfg)

    # debug
    # merge_point_clouds = diagram.GetSubsystemByName("merge_point_clouds")
    # context = merge_point_clouds.GetMyContextFromRoot(diagram.CreateDefaultContext())
    # pc = merge_point_clouds.GetOutputPort("point_cloud").Eval(context)
    # visualize_point_cloud(pc.xyzs())
    # np.save("outputs/point_cloud.npy", pc.xyzs())

    context = diagram.CreateDefaultContext()
    simulator = Simulator(diagram, context)
    simulator.set_target_realtime_rate(1.0)

    # lil hacky, should move into a func
    station = diagram.GetSubsystemByName("PandaManipulationStation")
    plant = station.GetSubsystemByName("plant")
    integrator = diagram.GetSubsystemByName("integrator")
    panda_arm = plant.GetModelInstanceByName("panda_arm")
    integrator.set_integral_value(
        integrator.GetMyContextFromRoot(context),
        plant.GetPositions(plant.GetMyContextFromRoot(context), panda_arm),
    )

    meshcat.StartRecording()
    simulator.AdvanceTo(35.0)
    meshcat.PublishRecording()

    import time

    while True:
        time.sleep(1)

    # simulator.AdvanceTo(0.1)
    # meshcat.Flush()  # Wait for the large object meshes to get to meshcat.
    # visualizer.StartRecording()

    # # run as fast as possible
    # simulator.set_target_realtime_rate(0)
    # meshcat.AddButton("Stop Simulation", "Escape")
    # print("Press Escape to stop the simulation")
    # while meshcat.GetButtonClicks("Stop Simulation") < 1:
    #     if cfg.max_time and simulator.get_context().get_time() > cfg.max_time:
    #         raise Exception("Took too long")
    #     simulator.AdvanceTo(simulator.get_context().get_time() + 2.0)
    #     # stats = diagram.get_output_port().Eval(simulator.get_context())
    #     visualizer.StopRecording()
    # visualizer.PublishRecording()
    # meshcat.DeleteButton("Stop Simulation")
    # return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Local NDF Pouring Robot")
    parser.add_argument("-n", help="Number of balls to add to box", default=20)
    parser.add_argument("-m", help="Basket ID to use", default=1)
    args = parser.parse_args()
    cfg = DictConfig(
        {
            # 'num_balls': args.n,
            "num_balls": 0,
            "basket_id": args.m,
            "max_time": 60.0,
        }
    )

    logging.getLogger("drake").addFilter(NoDiffIKWarnings())

    meshcat = StartMeshcat()
    pouring_demo(cfg, meshcat)
