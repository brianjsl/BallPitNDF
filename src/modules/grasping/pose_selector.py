import torch
import torch.nn as nn
import numpy as np
from src.modules.grasping.lndf_robot.eval.query_points import QueryPoints
from omegaconf import DictConfig
from src.modules.grasping.lndf_robot.models.conv_occupancy_net import ConvolutionalOccupancyNetwork
from src.modules.grasping.lndf_robot.models.vnn_occupancy_net import VNNOccNet
from src.modules.grasping.lndf_robot.opt.optimizer_lite import OccNetOptimizer
import os
import os.path as osp
from pathlib import Path
from src.modules.grasping.lndf_robot.opt.optimizer_lite import Demo
from hydra.utils import get_original_cwd


class LocalNDF:
    def __init__(self, lndf_cfg: DictConfig):

        #create LNDF model
        self.model = self._create_model(lndf_cfg['model'])

        #generate query points 
        self.query_point_cfg = lndf_cfg['query_point']
        self.query_points = self._create_query_pts(self.query_point_cfg['type'], self.query_point_cfg['args'])

        #generate eval_dir
        self.eval_dir = osp.join(get_original_cwd(), lndf_cfg['eval_dir'])
        os.makedirs(self.eval_dir, exist_ok=True)

        #create optimizer
        self.pose_optimizer_cfg = lndf_cfg['pose_optimizer']
        self.optimizer = self._create_optimizer(self.model, self.query_points, self.pose_optimizer_cfg, self.eval_dir)

    # properties
    @property
    def ModelTypes(self):
        return ["CONV_OCC", "VNN_NDF"]

    @property
    def query_pts(self):
        return self.query_points

    @property
    def viz_path(self):
        return self.eval_dir

    @property
    def QueryPointTypes(self):
        return {
            "SPHERE",
            "RECT",
            "CYLINDER",
            "ARM",
            "SHELF",
            "NDF_GRIPPER",
            "NDF_RACK",
            "NDF_SHELF",
        }

    def _create_model(self, model_cfg: DictConfig):
        model_type = model_cfg['type']
        model_args = model_cfg['args']
        model_ckpt = osp.join(get_original_cwd(), 'src/modules/grasping/lndf_robot/ckpts', model_cfg['ckpt'])
        assert model_type in self.ModelTypes, 'Invalid model type'

        if model_type == "CONV_OCC":
            model = ConvolutionalOccupancyNetwork(**model_args)
            print("Using CONV OCC")

        elif model_type == "VNN_NDF":
            model = VNNOccNet(**model_args)
            print("USING NDF")

        print("model_ckpt: ", model_ckpt)

        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        model.load_state_dict(torch.load(model_ckpt, map_location=device))
        return model

    def _create_optimizer(
        self, model: nn.Module, query_pts, optimizer_cfg: DictConfig, eval_dir: Path
    ):
        """
        Creates OccNetOptimizer for a given config.
        """
        if "opt_type" in optimizer_cfg:
            optimizer_type = optimizer_cfg["opt_type"]  # LNDF or GEOM
        else:
            optimizer_type = None

        optimizer_args = optimizer_cfg['args']
        if not optimizer_args:
            optimizer_args = {}
        if eval_dir is not None:
            opt_viz_path = osp.join(eval_dir, "visualization")
        else:
            opt_viz_path = "visualization"

        assert optimizer_type in ["GEOM", "LNDF"], "Invalid optimizer type"

        if optimizer_type == "LNDF":
            print("Using Occ Net optimizer")
            optimizer = OccNetOptimizer(
                model, query_pts, viz_path=opt_viz_path, **optimizer_args
            )
        return optimizer

    def make_cut(self, obj_pcd, r=0.1, sample_pt=None):
        """
        Cut out portion of object that is some distance away from a sample point.

        Args:
            r (float): Radius of cut out.
            sample_pt (np.ndarray, optional): (1 x 3) sample point to cut around.
                Defaults to None.
        """
        if sample_pt is None:
            sample_pt = obj_pcd[np.random.randint(0, obj_pcd.shape[0])][:]

        new_pcd = []
        for pt_idx in range(obj_pcd.shape[0]):
            pt = obj_pcd[pt_idx : pt_idx + 1][0]
            dis = (sum((pt.squeeze() - sample_pt) ** 2)) ** 0.5
            if dis > r:
                new_pcd.append(pt)
        return np.vstack(new_pcd)

    def _create_query_pts(self, query_pts_type, query_pts_args) -> np.ndarray:
        """
        Create query points from given config

        Args:
            query_pts_config(dict): Configs loaded from yaml file.

        Returns:
            np.ndarray: Query point as ndarray
        """

        assert query_pts_type in self.QueryPointTypes, "Invalid query point type"

        if query_pts_type == "SPHERE":
            query_pts = QueryPoints.generate_sphere(**query_pts_args)
        elif query_pts_type == "RECT":
            query_pts = QueryPoints.generate_rect(**query_pts_args)
        elif query_pts_type == "CYLINDER":
            query_pts = QueryPoints.generate_cylinder(**query_pts_args)
        elif query_pts_type == "ARM":
            query_pts = QueryPoints.generate_rack_arm(**query_pts_args)
        elif query_pts_type == "SHELF":
            query_pts = QueryPoints.generate_shelf(**query_pts_args)
        elif query_pts_type == "NDF_GRIPPER":
            query_pts = QueryPoints.generate_ndf_gripper(**query_pts_args)
        elif query_pts_type == "NDF_RACK":
            query_pts = QueryPoints.generate_ndf_rack(**query_pts_args)
        elif query_pts_type == "NDF_SHELF":
            query_pts = QueryPoints.generate_ndf_shelf(**query_pts_args)

        return query_pts

    def load_demo(self, demo: Demo):
        self.optimizer.add_demo(demo)

    def process_demos(self):
        self.optimizer.process_demos()

    def get_pose(self, target_obj_pcd, viz_path):
        return self.optimizer.optimize_transform_implicit(
            target_obj_pcd, ee=True, viz_path=viz_path, return_intermediates=True
        )
