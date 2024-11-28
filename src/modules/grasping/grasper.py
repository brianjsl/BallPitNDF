from src.modules.grasping import pose_selector
from src.modules.grasping.pose_selector import LocalNDF
from omegaconf import DictConfig
import os
import os.path as osp
import numpy as np
import random
from src.modules.grasping.lndf_robot.utils import path_util, util, torch_util
from scipy.spatial.transform import Rotation as R
from src.modules.grasping.lndf_robot.utils.plotly_save import multiplot, plot3d
import plotly.express as px
from src.modules.grasping.lndf_robot.eval.demo_io import DemoIO
from pydrake.all import (
    PointCloud,
    LeafSystem,
    Meshcat,
    System,
    RigidTransform,
    AbstractValue,
    RotationMatrix,
)
from hydra.utils import get_original_cwd


class LNDFGrasper(LeafSystem):
    def __init__(
        self,
        cfg: DictConfig,
        plant: System,
        meshcat: Meshcat,
        debug_pose: RigidTransform = None,
    ):
        super().__init__()
        self._meshcat = meshcat
        self.cut_pcd = False
        self.local_ndf = LocalNDF(cfg)
        self.pc_sample_pts = cfg.pc_sample_pts
        self.debug_pose = debug_pose

        self.DeclareAbstractInputPort(
            "merged_point_cloud", AbstractValue.Make(PointCloud(0))
        )

        self.DeclareAbstractOutputPort(
            "grasp_pose",
            lambda: AbstractValue.Make((RigidTransform(), self.local_ndf.query_pts)),
            self.get_grasp,
            {self.nothing_ticket()},  # doesn't change after initial computation
        )
        self.load_demos()

    def load_demos(self, demo_exp="lndf_mug_handle_demos", n_demos=10):
        demo_load_dir = osp.join(get_original_cwd(), "src", "demos", demo_exp)
        demo_fnames = os.listdir(demo_load_dir)

        assert len(demo_fnames), "No demonstrations found in path: %s!" % demo_load_dir

        # Sort out grasp demos
        grasp_demo_fnames = [
            osp.join(demo_load_dir, fn) for fn in demo_fnames if "grasp_demo" in fn
        ]

        self.demo_shapenet_ids = []
        self.demo_list = []

        # Iterate through all demos, extract relevant information and
        # prepare to pass into optimizer
        random.shuffle(grasp_demo_fnames)
        for grasp_demo_fn in grasp_demo_fnames[:n_demos]:
            print("Loading grasp demo from fname: %s" % grasp_demo_fn)
            grasp_data = np.load(grasp_demo_fn, allow_pickle=True)

            demo = DemoIO.process_grasp_data(grasp_data)
            self.demo_list.append(demo)

            self.local_ndf.load_demo(demo)
            self.demo_shapenet_ids.append(demo.obj_shapenet_id)

        self.local_ndf.process_demos()
        print("Shapenet IDs used in demo:")
        for id in self.demo_shapenet_ids:
            print("  " + id)

    @staticmethod
    def make_cut(obj_pcd, r=0.1, sample_pt=None):
        """
        Cut out portion of object that is some distance away from a sample point.

        Args:
            obj_pcd (np.ndarray): (n x 3) array representing a point cloud
            r (float): Radius of cut out.
            sample_pt (np.ndarray, optional): (1 x 3) sample point to cut around.
                Defaults to None.
        """
        if sample_pt is None:
            sample_pt = obj_pcd[np.random.randint(0, obj_pcd.shape[0])][:]
        print(sample_pt)

        new_pcd = []
        for pt_idx in range(obj_pcd.shape[0]):
            pt = obj_pcd[pt_idx : pt_idx + 1][0]
            dis = (sum((pt.squeeze() - sample_pt) ** 2)) ** 0.5
            if dis > r:
                new_pcd.append(pt)
        return np.vstack(new_pcd)

    def get_grasp(self, context, output):
        if self.debug_pose:
            output.set_value(self.debug_pose)
            return

        point_cloud = self.get_input_port(
            self.GetInputPort("merged_point_cloud").get_index()
        ).Eval(context)

        point_cloud_npy = point_cloud.xyzs().T * 0.5
        # point_cloud_npy = point_cloud_npy[np.random.choice(point_cloud_npy.shape[0], self.pc_sample_pts, replace=True), :]
        # fig = px.scatter_3d(x = point_cloud_npy[:, 0], y=point_cloud_npy[:, 1], z=point_cloud_npy[:, 2])
        # fig.show()
        pose_mats, best_idx, _ = self.local_ndf.get_pose(
            point_cloud_npy, self.local_ndf.viz_path
        )
        idx = best_idx
        best_pose_mat = pose_mats[idx] * 2

        final_query_pts = util.transform_pcd(self.local_ndf.query_pts, best_pose_mat)

        # offset to go from end effector pose to gripper
        offset_transform = RigidTransform([0, 0, -0.1])
        X_WB = RigidTransform(best_pose_mat) @ offset_transform

        output.set_value((X_WB, final_query_pts))
