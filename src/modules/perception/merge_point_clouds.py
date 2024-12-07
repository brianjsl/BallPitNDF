### Adapted from https://github.com/thatprogrammer1/stacking-robot/blob/main/src/perception/merge_point_clouds.py

import torch
import numpy as np
from pydrake.all import (
    LeafSystem,
    PointCloud,
    AbstractValue,
    RigidTransform,
    BaseField,
    Fields,
    Concatenate,
    PixelType,
)
from pydrake.systems.sensors import Image as SensorImage
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    AutoModelForMaskGeneration,
)
from manipulation.meshcat_utils import AddMeshcatTriad
from PIL import ImageDraw, Image
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import cv2


def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    # Find contours in the binary mask
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Extract the vertices of the contour
    polygon = largest_contour.reshape(-1, 2).tolist()

    return polygon


def polygon_to_mask(
    polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Convert a polygon to a segmentation mask.

    Args:
    - polygon (list): List of (x, y) coordinates representing the vertices of the polygon.
    - image_shape (tuple): Shape of the image (height, width) for the mask.

    Returns:
    - np.ndarray: Segmentation mask with the polygon filled.
    """
    # Create an empty mask
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Convert polygon to an array of points
    pts = np.array(polygon, dtype=np.int32)

    # Fill the polygon with white color (255)
    cv2.fillPoly(mask, [pts], color=(255,))

    return mask


def refine_masks(
    masks: torch.BoolTensor, polygon_refinement: bool = False
) -> List[np.ndarray]:
    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    masks = masks.numpy().astype(np.uint8)
    masks = list(masks)

    if polygon_refinement:
        for idx, mask in enumerate(masks):
            shape = mask.shape
            polygon = mask_to_polygon(mask)
            mask = polygon_to_mask(polygon, shape)
            masks[idx] = mask

    return masks


def depth_to_pC(uv: np.ndarray, depth_img, cam_info, mask):
    """Get 3d coordinates of bounding box from uv and depth image. Mask is the segmentation mask of the object"""
    fx = cam_info.focal_x()
    fy = cam_info.focal_y()
    cx = cam_info.center_x()
    cy = cam_info.center_y()

    u = uv[:, 0]
    v = uv[:, 1]

    depth_img_masked = depth_img[mask.astype(bool)]
    finite_mask = np.isfinite(depth_img_masked)
    min_Z = np.min(depth_img_masked[finite_mask])
    max_Z = np.max(depth_img_masked[finite_mask])
    Z_min = np.array([min_Z, min_Z])
    Z_max = np.array([max_Z, max_Z])

    X_min = (u - cx) * Z_min / fx
    Y_min = (v - cy) * Z_min / fy
    X_max = (u - cx) * Z_max / fx
    Y_max = (v - cy) * Z_max / fy

    pC_min = np.column_stack((X_min, Y_min, Z_min))
    pC_max = np.column_stack((X_max, Y_max, Z_max))
    pC = np.concatenate((pC_min, pC_max), axis=0)
    return pC


class MergePointClouds(LeafSystem):
    def __init__(
        self,
        plant,
        basket,
        camera_body_indices,
        cameras,
        meshcat,
        object_prompt: str = "a basket with handle.",
        debug: bool = False,
    ):
        super().__init__()
        self._meshcat = meshcat
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()

        self.object_prompt = object_prompt
        self.debug = debug

        # grounding dino
        model_id = "IDEA-Research/grounding-dino-base"
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        self.processor = processor
        self.model = model

        # grounding dino segmentation
        segmenter_id = "facebook/sam-vit-base"
        segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id)
        segmentator_processor = AutoProcessor.from_pretrained(segmenter_id)
        self.segmentator = segmentator
        self.segmentator_processor = segmentator_processor

        mug_point_cloud = AbstractValue.Make(PointCloud(0))
        rgb_image = AbstractValue.Make(SensorImage(0, 0))
        depth_image = AbstractValue.Make(SensorImage[PixelType.kDepth32F](0, 0))

        self._num_cameras = len(camera_body_indices)
        self._camera_ports = []
        self._camera_rgb_ports = []
        self._camera_depth_ports = []

        for i in range(self._num_cameras):
            point_cloud_port = f"camera{i}_point_cloud"
            self._camera_ports.append(
                self.DeclareAbstractInputPort(
                    point_cloud_port, mug_point_cloud
                ).get_index()
            )

            rgb_port = f"camera{i}_rgb_image"
            self._camera_rgb_ports.append(
                self.DeclareAbstractInputPort(rgb_port, rgb_image).get_index()
            )

            depth_port = f"camera{i}_depth_image"
            self._camera_depth_ports.append(
                self.DeclareAbstractInputPort(depth_port, depth_image).get_index()
            )

        self._cameras = cameras

        for i in range(self._num_cameras):
            X_WC = plant.EvalBodyPoseInWorld(
                self._plant_context, plant.get_body(camera_body_indices[i])
            )
            AddMeshcatTriad(meshcat, f"camera{i}", length=0.1, radius=0.005, X_PT=X_WC)


        # self.DeclareAbstractInputPort(
        #     "body_poses", AbstractValue.Make([RigidTransform()]))

        self.DeclareAbstractOutputPort(
            "point_cloud",
            lambda: AbstractValue.Make(PointCloud(new_size=0, fields=Fields(
                BaseField.kXYZs | BaseField.kRGBs | BaseField.kNormals))),
            self.GetPointCloud, 
            prerequisites_of_calc= set(
            [
                self.input_port_ticket(port) for port in self._camera_ports
            ])
        )

        # cheat ports (debugging)
        # context = plant.CreateDefaultContext()
        # basket_body = plant.GetBodyByName('basket_body_link', basket)
        # X_B = plant.EvalBodyPoseInWorld(context, basket_body)
        # margin = 0.001 

        # a = X_B.multiply([-0.1+margin, -0.2+margin, -0.2+margin])
        # b = X_B.multiply([0.1-margin, 0.3-margin, .2-margin])
        # print(a, b)
        # self._crop_lower = np.minimum(a, b)
        # self._crop_upper = np.maximum(a, b)

        # crop and find bounding box (debug)
        # line segment represents the diagonal of the box that contains the point cloud
        # meshcat.SetLineSegments(
        #     "/cropping_box",  self._crop_lower[:, None],
        #     self._crop_upper[:, None])

        self._camera_body_indices = camera_body_indices

    def GetPointCloud(self, context, output):
        if hasattr(self, "_cached_point_cloud"):
            output.set_value(self._cached_point_cloud)
            return

        # body_poses = self.get_input_port(
        #     self.GetInputPort("body_poses").get_index()
        # ).Eval(context)

        ps = []
        detection_imgs = []
        mask_imgs = []

        for i in range(self._num_cameras):
            # get rgb and depth image from camera
            rgb_port = self._camera_rgb_ports[i]
            rgb_image = self.get_input_port(rgb_port).Eval(context).data
            rgb_image = Image.fromarray(rgb_image).convert("RGB")
            depth_port = self._camera_depth_ports[i]
            depth_image = self.get_input_port(depth_port).Eval(context).data.squeeze()

            # get bounding box from grounding dino
            inputs = self.processor(
                images=rgb_image, text=self.object_prompt, return_tensors="pt"
            )
            with torch.no_grad():
                outputs = self.model(**inputs)

            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=0.3,
                text_threshold=0.3,
                target_sizes=[rgb_image.size[::-1]],
            )

            box = results[0]["boxes"][-1]
            box_list = box.tolist()

            # draw bounding box on the image
            rgb_image_with_box = rgb_image.copy()
            draw = ImageDraw.Draw(rgb_image_with_box)
            draw.rectangle(box_list, outline="red", width=2)

            # get segmentation mask of the object
            with torch.inference_mode():
                inputs = self.segmentator_processor(
                    images=rgb_image, input_boxes=[[box_list]], return_tensors="pt"
                )
                outputs = self.segmentator(**inputs)

            masks = self.segmentator_processor.post_process_masks(
                masks=outputs.pred_masks,
                original_sizes=inputs.original_sizes,
                reshaped_input_sizes=inputs.reshaped_input_sizes,
            )[0]

            masks = refine_masks(masks, polygon_refinement=True)

            # convert 2d coordinates of bounding box to 3d coordinates
            camera_info = self._cameras[i].depth_camera_info()
            uv = box.view(2, 2).numpy()
            mask = masks[0]
            pC = depth_to_pC(uv, depth_image, camera_info, mask=mask)

            mask_img = Image.fromarray(mask, mode="L")

            detection_imgs.append(rgb_image_with_box)
            mask_imgs.append(mask_img)

            # convert pC to world frame
            X_WC = self._plant.EvalBodyPoseInWorld(
                self._plant_context, self._plant.get_body(self._camera_body_indices[i])
            )
            pW = X_WC.multiply(pC.T).T
            ps.append(pW)

        # create 2 x 3 grid of images
        if self.debug:
            import matplotlib.pyplot as plt

            imgs_grid = detection_imgs + mask_imgs

            # Create a figure with a 2x3 grid
            fig, axes = plt.subplots(
                2, self._num_cameras, figsize=(15, 10)
            )  # Adjust figsize as needed

            for i, ax in enumerate(axes.flat):  # Flatten the 2D grid for iteration
                if i < len(imgs_grid):
                    ax.imshow(imgs_grid[i])

            # Save the resulting grid image
            output_path = "debug_vis.png"
            plt.tight_layout()
            plt.savefig(output_path, bbox_inches="tight")
            plt.close(fig)

            print(f"Grid saved as {output_path}")

        # get bounding box of the merged point cloud
        ps = np.concatenate(ps, axis=0)
        lower = np.min(ps, axis=0)
        upper = np.max(ps, axis=0)

        print("lower: ", lower)
        print("upper: ", upper)

        AddMeshcatTriad(
            self._meshcat,
            path=f"lower",
            length=0.1,
            radius=0.005,
            X_PT=RigidTransform(p=lower),
        )
        AddMeshcatTriad(
            self._meshcat,
            path=f"upper",
            length=0.1,
            radius=0.005,
            X_PT=RigidTransform(p=upper),
        )

        pcd = []
        for i in range(self._num_cameras):
            port = self._camera_ports[i]
            cloud = self.get_input_port(port).Eval(context)

            # pcd.append(cloud.Crop(self._crop_lower, self._crop_upper))
            pcd.append(cloud.Crop(lower, upper))
            pcd[i].EstimateNormals(radius=0.1, num_closest=30)

            # Flip normals toward camera
            # X_WC = body_poses[self._camera_body_indices[i]]
            X_WC = self._plant.EvalBodyPoseInWorld(
                self._plant_context, self._plant.get_body(self._camera_body_indices[i])
            )
            pcd[i].FlipNormalsTowardPoint(X_WC.translation())

        merged_pcd = Concatenate(pcd)
        down_sampled_pcd = merged_pcd.VoxelizedDownSample(voxel_size=0.005)
        output.set_value(down_sampled_pcd)

        # Cache the computed point cloud
        self._cached_point_cloud = down_sampled_pcd
        output.set_value(down_sampled_pcd)
