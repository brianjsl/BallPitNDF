from src.modules.grasping.lndf_robot.models.conv_occupancy_net.encoder import (
    pointnet, voxels, pointnetpp
)

encoder_dict = {
    'pointnet_local_pool': pointnet.LocalPoolPointnet,
    'pointnet_crop_local_pool': pointnet.PatchLocalPoolPointnet,
    'pointnet_plus_plus': pointnetpp.PointNetPlusPlus,
    'voxel_simple_local': voxels.LocalVoxelEncoder,
}
