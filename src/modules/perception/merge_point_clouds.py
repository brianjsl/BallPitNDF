### Adapted from https://github.com/thatprogrammer1/stacking-robot/blob/main/src/perception/merge_point_clouds.py

import numpy as np
from pydrake.all import (
    LeafSystem, PointCloud, AbstractValue, RigidTransform, BaseField, Fields, Concatenate
)

class MergePointClouds(LeafSystem):
    def __init__(self, plant, basket, camera_body_indices, meshcat):
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
            self.GetPointCloud, 
            prerequisites_of_calc= set([
                self.input_port_ticket(self.GetInputPort("body_poses").get_index()),
            ] + [
                self.input_port_ticket(port) for port in self._camera_ports
            ])
            )

        context = plant.CreateDefaultContext()
        basket_body = plant.GetBodyByName('basket_body_link', basket)
        X_B = plant.EvalBodyPoseInWorld(context, basket_body)
        margin = 0.001 

        a = X_B.multiply([-0.3+margin, -0.3+margin, -0.3+margin])
        b = X_B.multiply([0.3-margin, 0.7-margin, .3-margin])
        print(a, b)
        self._crop_lower = np.minimum(a, b)
        self._crop_upper = np.maximum(a, b)

        # crop and find bounding box (debug)
        # line segment represents the diagonal of the box that contains the point cloud
        meshcat.SetLineSegments(
            "/cropping_box",  self._crop_lower[:, None],
            self._crop_upper[:, None])
        
        self._camera_body_indices = camera_body_indices
    
    # def GetPointCloud(self, context, output):
    #     # Check if inputs have changed
    #     # Implement caching mechanism here...

    #     # Process point clouds in parallel
    #     from concurrent.futures import ThreadPoolExecutor

    #     def process_cloud(i):
    #         cloud = self.get_input_port(self._camera_ports[i]).Eval(context)
    #         cropped_cloud = cloud.Crop(self._crop_lower, self._crop_upper)
    #         # Early downsampling
    #         downsampled_cloud = cropped_cloud.VoxelizedDownSample(voxel_size=0.01)
    #         # Optional: Skip normal estimation if not needed
    #         downsampled_cloud.EstimateNormals(radius=0.05, num_closest=10)
    #         X_WC = body_poses[self._camera_body_indices[i]]
    #         downsampled_cloud.FlipNormalsTowardPoint(X_WC.translation())
    #         return downsampled_cloud

    #     body_poses = self.get_input_port(self.GetInputPort("body_poses").get_index()).Eval(context)

    #     with ThreadPoolExecutor() as executor:
    #         pcd = list(executor.map(process_cloud, range(self._num_cameras)))

    #     merged_pcd = Concatenate(pcd)
    #     # Final downsampling if necessary
    #     down_sampled_pcd = merged_pcd.VoxelizedDownSample(voxel_size=0.005)
    #     output.set_value(down_sampled_pcd)
    
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
        

    