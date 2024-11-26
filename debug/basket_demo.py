from src.modules.grasping.pose_selector import LocalNDF 
from omegaconf import DictConfig
import os
import os.path as osp
import numpy as np
import random
from src.modules.grasping.lndf_robot.utils import path_util, util, torch_util
import trimesh
from scipy.spatial.transform import Rotation as R
from src.modules.grasping.lndf_robot.utils.plotly_save import multiplot
import plotly.express as px
from src.modules.grasping.lndf_robot.eval.demo_io import DemoIO
from plotly.offline import iplot

os.environ['CKPT_DIR'] = 'src/modules/lndf_robot/ckpts'
seed = 0
np.random.seed(seed)
random.seed(seed)

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
        pt = obj_pcd[pt_idx:pt_idx + 1][0]
        dis = (sum((pt.squeeze() - sample_pt)**2))**0.5
        if dis > r:
            new_pcd.append(pt)
    return np.vstack(new_pcd)

def get_basket_pcd(use_random_rotation=True, cut_pcd=False):
    obj_model = osp.join('src','assets','basket', 'basket.obj')
    obj_scale = 0.002
    n_samples = 1000
    assert osp.exists(obj_model), 'Object model not found'
    obj_mesh = trimesh.load(obj_model, process=False)
    obj_mesh.apply_scale(obj_scale)
    obj_pcd = obj_mesh.sample(n_samples)

    random_rotation = np.eye(4)
    if use_random_rotation:
        random_rotation[:3, :3] = R.random().as_matrix()

    rotated_obj_pcd = util.transform_pcd(obj_pcd, random_rotation)

    if cut_pcd:
        rotated_obj_pcd = make_cut(rotated_obj_pcd)

    fig = multiplot([obj_pcd, rotated_obj_pcd + np.array([[0.2, 0, 0]])], write_html=False) 
    fig.update_layout(coloraxis=dict(cmax=2, cmin=-1))
    iplot(fig)
    return rotated_obj_pcd

def get_local_ndf_cfg():
    config = {
    'lndf': {
        'eval_dir': 'outputs',
        'pose_optimizer': {
            'opt_type': 'LNDF',
            'args': {
                'opt_iterations': 1000,
                'rand_translate': True,
                'use_tsne': False,
                'M_override': 20,
            }
        },
        'query_point': {
            'type': 'RECT',
            'args': {
                'n_pts': 1000,
                'x': 0.08,
                'y': 0.04,
                'z1': 0.05,
                'z2': 0.02,
            }
        },
        'model': {
            'type': 'CONV_OCC',
            'args': {
                'latent_dim': 128,  # Number of voxels in convolutional occupancy network
                'model_type': 'pointnet',  # Encoder type
                'return_features': True,  # Return latent features for evaluation
                'sigmoid': False,  # Use sigmoid activation on last layer
                'acts': 'last',  # Return last activations of occupancy network
            },
            'ckpt': 'lndf_weights.pth'
        }
        }
    }
    return LocalNDF(config)

def load_demos(local_ndf: LocalNDF):
    demo_exp = 'lndf_mug_handle_demos'
    n_demos = 10

    demo_load_dir = osp.join('src', 'demos', demo_exp)
    demo_fnames = os.listdir(demo_load_dir)

    assert len(demo_fnames), 'No demonstrations found in path: %s!' \
        % demo_load_dir

    # Sort out grasp demos
    grasp_demo_fnames = [osp.join(demo_load_dir, fn) for fn in
        demo_fnames if 'grasp_demo' in fn]

    demo_shapenet_ids = []
    demo_list = []


    # Iterate through all demos, extract relevant information and
    # prepare to pass into optimizer
    random.shuffle(grasp_demo_fnames)
    for grasp_demo_fn in grasp_demo_fnames[:n_demos]:
        print('Loading grasp demo from fname: %s' % grasp_demo_fn)
        grasp_data = np.load(grasp_demo_fn, allow_pickle=True)

        demo = DemoIO.process_grasp_data(grasp_data)
        demo_list.append(demo)

        local_ndf.load_demo(demo)
        demo_shapenet_ids.append(demo.obj_shapenet_id)

    local_ndf.process_demos()
    print('Shapenet IDs used in demo:')
    for id in demo_shapenet_ids:
        print('  ' + id)
    
def get_basket_grip_pose(local_ndf: LocalNDF, rotated_obj_pcd: np.ndarray):
    pose_mats, best_idx, intermediates = local_ndf.get_pose(rotated_obj_pcd, local_ndf.viz_path)
    idx = best_idx
    best_pose_mat = pose_mats[idx]

    final_query_pts = util.transform_pcd(local_ndf.query_pts, best_pose_mat)

    # Generate trail of intermediate optimization results
    intermediate_query_pts = []
    query_pts_mean = np.mean(local_ndf.query_pts, axis=0).reshape(1, 3)
    for iter_mats in intermediates:
        iter_query_pts = util.transform_pcd(query_pts_mean, iter_mats[idx])
        intermediate_query_pts.append(iter_query_pts)

    # Plot all the results
    plot_list = [
        rotated_obj_pcd,
        final_query_pts,
    ] + intermediate_query_pts

    plot_pts = np.vstack(plot_list)

    # Assign different colors to different objects
    color = np.ones(plot_pts.shape[0])
    color[rotated_obj_pcd.shape[0]:rotated_obj_pcd.shape[0] + final_query_pts.shape[0]] *= 2
    color[rotated_obj_pcd.shape[0] + final_query_pts.shape[0]:] *= 3
    fig = px.scatter_3d(
        x=plot_pts[:, 0], y=plot_pts[:, 1], z=plot_pts[:, 2], color=color)
    fig.update_layout(coloraxis=dict(cmax=max(color) + 1, cmin=-1))

    iplot(fig)

if __name__ == '__main__':
    rotated_obj_pcd = get_basket_pcd()
    local_ndf = get_local_ndf_cfg()
    load_demos(local_ndf)
    get_basket_grip_pose(local_ndf, rotated_obj_pcd)
