import torch
import torch.nn as nn
import numpy as np
from src.modules.lndf_robot.eval.query_points import QueryPoints
from omegaconf import DictConfig
from src.modules.lndf_robot.models.conv_occupancy_net import ConvolutionalOccupancyNetwork
from src.modules.lndf_robot.models.vnn_occupancy_net import VNNOccNet
from src.modules.lndf_robot.opt.optimizer_lite import OccNetOptimizer
from src.modules.lndf_robot.opt.optimizer_geom import GeomOptimizer
import os
import os.path as osp
from pathlib import Path

class LocalNDF():
    def __init__(self, cfg: DictConfig, obj_pcd):
        self.obj_pcd = obj_pcd

        #create LNDF model
        self.model = self._create_model(cfg['lndf']['model'])

        #generate query points 
        self.query_point_args = cfg['lndf']['query_point_args']
        self.query_points = self.create_query_pts(cfg['lndf']['query_point_type'], self.query_point_args)

        #generate eval_dir
        eval_dir = osp.join(os.get_cwd(), 'src/modules/lndf_robot/', cfg['lndf']['eval_dir'])
        os.makedirs(eval_dir, exist_ok=True)

        self.pose_optimizer_cfg = cfg['lndf']['pose_optimizer']
        self.optimizer = OccNetOptimizer(self.model, self.query_points, eval_dir, **self.pose_optimizer_cfg)
    
    def _create_model(self, model_cfg: DictConfig):
        model_type = model_cfg['type']
        model_args = model_cfg['args']
        model_ckpt = osp.join(os.getcwd(), 'src/modules/lndf_robot/ckpts', model_cfg['ckpt'])
        assert model_type in self.ModelTypes, 'Invalid model type'

        if model_type == 'CONV_OCC':
            model = ConvolutionalOccupancyNetwork(
                **model_args)
            print('Using CONV OCC')

        elif model_type == 'VNN_NDF':
            model = VNNOccNet(**model_args)
            print('USING NDF')

        model.load_state_dict(torch.load(model_ckpt))
    
    def _create_optimizer(self, model: nn.Module, query_pts, optimizer_cfg: DictConfig, eval_dir: Path):
        '''
        Creates OccNetOptimizer for a given config. 
        '''
        if 'opt_type' in optimizer_cfg:
            optimizer_type = optimizer_cfg['opt_type']  # LNDF or GEOM
        else:
            optimizer_type = None

        optimizer_config = optimizer_cfg['args']
        if eval_dir is not None:
            opt_viz_path = osp.join(eval_dir, 'visualization')
        else:
            opt_viz_path = 'visualization'

        if optimizer_type == 'GEOM':
            print('Using geometric optimizer')
            optimizer = GeomOptimizer(model, query_pts, viz_path=opt_viz_path,
                **optimizer_config)
        else:
            print('Using Occ Net optimizer')
            optimizer = OccNetOptimizer(model, query_pts, viz_path=opt_viz_path,
                **optimizer_config)
        return optimizer

    @property
    def ModelTypes():
        return ['CONV_OCC', 'VNN_NDF']

    @property
    def QueryPointTypes():
        return {
            'SPHERE',
            'RECT',
            'CYLINDER',
            'ARM',
            'SHELF',
            'NDF_GRIPPER',
            'NDF_RACK',
            'NDF_SHELF',
        }

    def make_cut(self, r = 0.1, sample_pt = None):
        """
    Cut out portion of object that is some distance away from a sample point.

    Args:
        r (float): Radius of cut out.
        sample_pt (np.ndarray, optional): (1 x 3) sample point to cut around. 
            Defaults to None.
    """
        if sample_pt is None:
            sample_pt = self.obj_pcd[np.random.randint(0, self.obj_pcd.shape[0])][:]

        new_pcd = []
        for pt_idx in range(self.obj_pcd.shape[0]):
            pt = self.obj_pcd[pt_idx:pt_idx + 1][0]
            dis = (sum((pt.squeeze() - sample_pt)**2))**0.5
            if dis > r:
                new_pcd.append(pt)
        return np.vstack(new_pcd)
    
    def create_query_pts(self, query_pts_type, query_pts_args) -> np.ndarray:
        """
        Create query points from given config

        Args:
            query_pts_config(dict): Configs loaded from yaml file.

        Returns:
            np.ndarray: Query point as ndarray
        """

        assert query_pts_type in self.QueryPointTypes, 'Invalid query point type'

        if query_pts_type == 'SPHERE':
            query_pts = QueryPoints.generate_sphere(**query_pts_args)
        elif query_pts_type == 'RECT':
            query_pts = QueryPoints.generate_rect(**query_pts_args)
        elif query_pts_type == 'CYLINDER':
            query_pts = QueryPoints.generate_cylinder(**query_pts_args)
        elif query_pts_type == 'ARM':
            query_pts = QueryPoints.generate_rack_arm(**query_pts_args)
        elif query_pts_type == 'SHELF':
            query_pts = QueryPoints.generate_shelf(**query_pts_args)
        elif query_pts_type == 'NDF_GRIPPER':
            query_pts = QueryPoints.generate_ndf_gripper(**query_pts_args)
        elif query_pts_type == 'NDF_RACK':
            query_pts = QueryPoints.generate_ndf_rack(**query_pts_args)
        elif query_pts_type == 'NDF_SHELF':
            query_pts = QueryPoints.generate_ndf_shelf(**query_pts_args)

        return query_pts 
        
    def get_pose(self, target_obj_pcd, viz_path):
        return self.optimizer.optimize_transform_implicit(target_obj_pcd, ee=True, viz_path = viz_path, 
                                                          return_intermediates=True)