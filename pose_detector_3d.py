import cv2
import csv
import numpy as np
import open3d
from PIL import Image
from utils.kinect_utils import get_align_color_image, create_color_point_cloud

def create_intrinsic():
    """
    Creates an intrinsic with default parameters for its use in Open 3D
    """
    # Hardcode the camera intrinsic parameters for backprojection
    depth_height=424
    depth_width=512
    ppx=260.166
    ppy=205.197
    fx=367.535
    fy=367.535
    intrinsic = open3d.camera.PinholeCameraIntrinsic(depth_width, depth_height, fx, fy, ppx, ppy)
    return intrinsic

def get_3d_poses(dlcl_cfg, poses_list, color_frames, depth_frame, kinect):
    """
    Uses all the retrieved data to calculate the 3d joint positions
    """
    # Final color frame with the calculated and drawn joints' coordinates
    labeled_img = None
    for i in range(len(poses_list)):
        for joint_key, joint_values in poses_list[i].items():
            # Calculates the coordinates of the detected joints for the
            # deeplabcut-live resized image
            x_resized = int(joint_values[0])
            y_resized = int(joint_values[1])
            # Uses the resize factor to determine the coordinates of the
            # detected joints, considering the retrieved x,y coordinates
            # of the resized image
            x_original = int(x_resized*3.2)
            y_original = int(y_resized*3.2)
            # Draw circles into the original coloured image, with the
            # calculated joint locations
            labeled = cv2.circle(
                # Image to draw into
                color_frames['original'],
                # Circle center coordinates
                (x_original, y_original),
                # Circle radius
                int(round(dlcl_cfg['display_radius']*3.2)), 
                # Color to use for drawing
                (152, 3, 252),
                # thickness = -1 tells to cv to fill the circle
                # with the color
                thickness=-1
            )
            labeled_img = Image.fromarray(labeled)
    
    labeled_img.show()
        
    if labeled_img is not None and depth_frame is not None:
        labeled_img = np.array(labeled_img)
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
        point_cloud.points, point_cloud.colors = create_color_point_cloud(
            align_color_img, 
            depth_frame, 
            depth_scale, 
            clipping_distance_in_meters, 
            intrinsic
            )
        # To convert [x,y,z] -> [x.-y,-z]
        flip_transform = [[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]]
        point_cloud.transform(flip_transform)

        # Merge XYZ coordinates and RGB colors
        output_cloud = np.concatenate(
            (np.asarray(point_cloud.points), 
            np.asarray(point_cloud.colors)), 
            axis=1
            )

        headers = ['X', 'Y', 'Z', 'R', 'G', 'B']

        if output_cloud.size > 0:
            output_file = 'test_cloud.txt'
            with open(output_file, 'w', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(output_cloud)