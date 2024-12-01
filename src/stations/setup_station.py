from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    Parser,
    MeshcatVisualizer,
    InverseDynamicsController,
    PassThrough,
    Demultiplexer,
    StateInterpolatorWithDiscreteDerivative,
    Multiplexer,
    Meshcat,
    MultibodyPlant,
    Adder,
)
from manipulation.station import MakeHardwareStation, LoadScenario
import os
from manipulation.utils import ConfigureParser
from manipulation.scenarios import AddRgbdSensors
from manipulation.station import AddPointClouds
import numpy as np
from IPython.display import SVG, display
import pydot
from omegaconf import DictConfig
import random
from hydra.utils import get_original_cwd

def get_scenario(scenario_cfg: DictConfig) -> str:
    scenario_data = f"""
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
    file: file://{get_original_cwd()}/src/assets/panda_hand/panda_hand.urdf
- add_weld:
    parent: panda_arm::panda_link8
    child: panda_hand::panda_hand
    X_PC:
        translation: [0, 0, 0]
        rotation: !Rpy {{ deg: [0, 0, -45] }}
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
    file: file://{get_original_cwd()}/src/assets/wooden_basket_big_handle/wooden_basket.sdf
    default_free_body_pose:
        basket_body_link:
            translation: [0.5, 0, 0]
            rotation: !Rpy {{ deg: [90, 0, 0]}}
- add_model:
    name: bin0
    file: package://drake_models/manipulation_station/bin.sdf
    default_free_body_pose:
        bin_base:
            translation: [0.5, -0.5, 0]
            rotation: !Rpy {{deg: [0, 0, 90]}}
- add_frame:
    name: camera0_origin
    X_PF:
        base_frame: world
        rotation: !Rpy {{deg: [-130.0, 0, 0.0]}}
        translation: [.5, -.6, 0.8]
- add_frame:
    name: camera1_origin
    X_PF:
        base_frame: world
        rotation: !Rpy {{deg: [-140., 0, 90.0]}}
        translation: [1.0, 0, 0.8]
- add_frame:
        name: camera2_origin
        X_PF:
            base_frame: world
            rotation: !Rpy {{deg: [-130., 0, -180.0]}}
            translation: [.5, .7, .8]
- add_model:
    name: camera0
    file: package://manipulation/camera_box.sdf
- add_model:
    name: camera1
    file: package://manipulation/camera_box.sdf
- add_model:
    name: camera2
    file: package://manipulation/camera_box.sdf
- add_weld:
    parent: camera0_origin
    child: camera0::base
- add_weld:
    parent: camera1_origin
    child: camera1::base
- add_weld:
    parent: camera2_origin
    child: camera2::base
"""
    for i in range(scenario_cfg.num_balls):
        scenario_data += f"""
- add_model:
    name: ball_{i}
    file: file://{get_original_cwd()}/src/assets/sphere/sphere.sdf
    default_free_body_pose:
        sphere_body_link:
            translation: [{0.5 + 0.1*(random.random()-0.5)}, {0.1*(random.random()-0.5)}, 0.8]
"""
    scenario_data += """
cameras:
    camera0:
        name: camera0
        depth: True
        X_PB: 
            base_frame: camera0::base
    camera1:
        name: camera1
        depth: True
        X_PB: 
            base_frame: camera1::base
    camera2:
        name: camera2
        depth: True
        X_PB: 
            base_frame: camera2::base
"""
    scenario_data += """
model_drivers:
    panda_arm+panda_hand: !InverseDynamicsDriver {}
"""
    return scenario_data
