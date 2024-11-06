import argparse
from pydrake.all import (
    Meshcat, StartMeshcat, Simulator
)
from omegaconf import DictConfig
from src.setup import MakePandaManipulationStation
import logging
import os
import random
import numpy as np

class NoDiffIKWarnings(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith("Differential IK")

def get_directives(cfg: DictConfig):
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
"""
    for i in range(cfg.num_balls):
        env_directives += f"""
    - add_frame:
        name: ball_{i}_origin
        X_PF: 
            base_frame: world
            translation: [{0.5 + np.random.choice([-1,1])*0.03 + 0.1*(random.random()-0.5)}, {np.random.choice([-1,1])*0.03 + 0.1*(random.random()-0.5)}, 0.5]
    - add_model:
        name: ball_{i}
        file: file://{os.getcwd()}/src/assets/sphere/sphere.sdf
    - add_weld: 
        parent: ball_{i}_origin
        child: ball_{i}::sphere 
"""
    return robot_directives, env_directives

def pouring_demo(cfg: DictConfig, meshcat: Meshcat):

    meshcat.Delete()
    robot_directives, env_directives = get_directives(cfg)
    diagram = MakePandaManipulationStation(
        robot_directives=robot_directives,
        env_directives=env_directives,
        meshcat=meshcat
    )
    simulator = Simulator(diagram)

    simulator.AdvanceTo(0.1)
    meshcat.Flush()  # Wait for the large object meshes to get to meshcat.
    meshcat.StartRecording()

    # run as fast as possible
    simulator.set_target_realtime_rate(0)
    meshcat.AddButton("Stop Simulation", "Escape")
    print("Press Escape to stop the simulation")
    while meshcat.GetButtonClicks("Stop Simulation") < 1:
        if cfg.max_time and simulator.get_context().get_time() > cfg.max_time:
            raise Exception("Took too long")
        simulator.AdvanceTo(simulator.get_context().get_time() + 2.0)
        # stats = diagram.get_output_port().Eval(simulator.get_context())
        meshcat.StopRecording()
    meshcat.PublishRecording()
    meshcat.DeleteButton("Stop Simulation")

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
