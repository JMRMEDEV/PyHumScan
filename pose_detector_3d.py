import cv2
import numpy as np
import open3d

from utils.cloud_utils import get_joints_points
from utils.color_utils import denormalize_RGB, get_joint_color
from utils.file_utils import save_data
from utils.kinect_utils import create_color_point_cloud, get_align_color_image


def create_intrinsic():
    """
    Creates an intrinsic with default parameters for its use in Open 3D
    """
    # Hardcode the camera intrinsic parameters for backprojection
    depth_height = 424
    depth_width = 512
    ppx = 260.166
    ppy = 205.197
    fx = 367.535
    fy = 367.535
    intrinsic = open3d.camera.PinholeCameraIntrinsic(
        depth_width, depth_height, fx, fy, ppx, ppy
    )
    return intrinsic


def get_3d_poses(
    dlcl_cfg, data_save_cfg, poses_list, color_frames, depth_frame, kinect
):
    """
    Uses all the retrieved data to calculate the 3d joint positions
    """
    if data_save_cfg["original"]:
        save_data("original", color_frames["original"])
    # Final color frame with the calculated and drawn joints' coordinates
    labeled_img = color_frames["original"]
    # List for saving the joints' information
    joints_resized = []
    for i in range(len(poses_list)):
        for joint_key, joint_values in poses_list[i].items():
            # Calculates the coordinates of the detected joints for the
            # deeplabcut-live resized image
            x_resized = int(joint_values[0])
            y_resized = int(joint_values[1])
            # Create a list with the resized joints' coordinates and keys
            # in case we want to store them
            joint_info_resized = [joint_key, x_resized, y_resized]
            joints_resized.append(joint_info_resized)
            # Uses the resize factor to determine the coordinates of the
            # detected joints, considering the retrieved x,y coordinates
            # of the resized image
            x_original = int(x_resized * 3.2)
            y_original = int(y_resized * 3.2)
            # Draw circles into the original coloured image, with the
            # calculated joint locations
            labeled = cv2.circle(
                # Image to draw into
                labeled_img,
                # Circle center coordinates
                (x_original, y_original),
                # Circle radius
                int(round(dlcl_cfg["display_radius"])),
                # Color to use for drawing (BGR)
                get_joint_color(joint_key),
                # thickness = -1 tells to cv to fill the circle
                # with the color
                thickness=-1,
            )
            labeled_img = labeled

    if data_save_cfg["labeled_resized"]:
        save_data("labeled_resized", joints_resized)
    if data_save_cfg["labeled_original"]:
        save_data("labeled_original", labeled_img)

    if labeled_img is not None and depth_frame is not None:
        # Create Open 3D Point Cloud
        point_cloud = open3d.geometry.PointCloud()
        # Default kinect depth scale where 1 unit = 0.001 m = 1 mm
        depth_scale = 0.001
        # Set the maximum distance to display the point cloud data
        clipping_distance_in_meters = 3
        # Open3D
        intrinsic = create_intrinsic()
        align_color_img = get_align_color_image(kinect, labeled_img)
        # Write the point cloud points and colors
        point_cloud.points, point_cloud.colors, aligned_img = create_color_point_cloud(
            align_color_img,
            depth_frame,
            depth_scale,
            clipping_distance_in_meters,
            intrinsic,
        )

        if data_save_cfg["labeled_aligned"]:
            save_data("labeled_aligned", aligned_img)

        # To convert [x,y,z] -> [x.-y,-z]
        flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        point_cloud.transform(flip_transform)

        colors_RGB = np.asarray(point_cloud.colors)
        colors_RGB = denormalize_RGB(colors_RGB)
        # Merge XYZ coordinates and RGB colors
        output_cloud = np.concatenate(
            (np.asarray(point_cloud.points), colors_RGB), axis=1
        )

        pose_3d, output_cloud = get_joints_points(output_cloud)

        if data_save_cfg["colored_cloud"]:
            save_data("colored_cloud", output_cloud)

        if data_save_cfg["pose_3d"]:
            save_data("pose_3d", pose_3d)

        return pose_3d, output_cloud
