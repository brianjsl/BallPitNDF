### Adapted from https://github.com/thatprogrammer1/stacking-robot/blob/main/src/perception/merge_point_clouds.py

import numpy as np
from pydrake.all import (
    LeafSystem, PointCloud, AbstractValue, RigidTransform, BaseField, Fields, Concatenate
)

class MergePointClouds(LeafSystem):
    def __init__(self, plant, bowl, camera_body_indices, meshcat):
        super().__init__()
        self._meshcat = meshcat
        mug_point_cloud = AbstractValue.Make(PointCloud(0))

        self._num_cameras = len(camera_body_indices)
        self._camera_ports = []
        for i in range(self._num_cameras):
            point_cloud_port = f"camera{i}_point_cloud"
            self._camera_ports.append(self.DeclareAbstractInputPort(
                point_cloud_port, mug_point_cloud).get_index())

        self.DeclareAbstractInputPort(
            "body_poses", AbstractValue.Make([RigidTransform()]))

        self.DeclareAbstractOutputPort(
            "point_cloud",
            lambda: AbstractValue.Make(PointCloud(new_size=0, fields=Fields(
                BaseField.kXYZs | BaseField.kRGBs | BaseField.kNormals))),
            self.GetPointCloud)

        context = plant.CreateDefaultContext()
        bowl_body = plant.GetBodyByName('bowl_body_link', bowl)
        X_B = plant.EvalBodyPoseInWorld(context, bowl_body)
        margin = 0.001  # only because simulation is perfect!
        # TODO: change if we change bin size/location
        a = X_B.multiply([0.2+margin, -1+margin, -0.9+margin])
        b = X_B.multiply([.8-margin, 1-margin, 0.4])
        # print(a, b)
        self._crop_lower = np.minimum(a, b)
        self._crop_upper = np.maximum(a, b)

        meshcat.SetLineSegments(
            "/cropping_box",  self._crop_lower[:, None],
            self._crop_upper[:, None])
        
        self._camera_body_indices = camera_body_indices
    
    def GetPointCloud(self, context, output):
        body_poses = self.get_input_port(
            self.GetInputPort("body_poses").get_index()
        ).Eval(context)

        pcd = []
        for i in range(self._num_cameras):
            port = self._camera_ports[i]
            cloud = self.get_input_port(
                port).Eval(context)

            pcd.append(cloud.Crop(self._crop_lower, self._crop_upper))
            pcd[i].EstimateNormals(radius=0.1, num_closest=30)

            # Flip normals toward camera
            X_WC = body_poses[self._camera_body_indices[i]]
            pcd[i].FlipNormalsTowardPoint(X_WC.translation())

        merged_pcd = Concatenate(pcd)
        down_sampled_pcd = merged_pcd.VoxelizedDownSample(voxel_size=0.005)
        output.set_value(down_sampled_pcd)
        

    