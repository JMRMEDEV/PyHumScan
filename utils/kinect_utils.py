import ctypes

import numpy as np
import open3d
from pykinect2.PyKinectV2 import _ColorSpacePoint


def get_align_color_image(kinect, color_img, color_height=1080, color_width=1920, depth_height=424, depth_width=512):
    """
    Takes a color frame and aligns it with the depth
    """
    CSP_Count = kinect._depth_frame_data_capacity
    CSP_type = _ColorSpacePoint * CSP_Count.value
    CSP = ctypes.cast(CSP_type(), ctypes.POINTER(_ColorSpacePoint))

    kinect._mapper.MapDepthFrameToColorSpace(kinect._depth_frame_data_capacity,kinect._depth_frame_data, CSP_Count, CSP)   

    # Convert ctype pointer to array
    colorXYs = np.copy(np.ctypeslib.as_array(CSP, shape=(depth_height*depth_width,)))
    # Convert struct array to regular numpy array
    colorXYs = colorXYs.view(np.float32).reshape(colorXYs.shape + (-1,))
    colorXYs += 0.5
    colorXYs = colorXYs.reshape(depth_height,depth_width,2).astype(np.int)
    colorXs = np.clip(colorXYs[:,:,0], 0, color_width-1)
    colorYs = np.clip(colorXYs[:,:,1], 0, color_height-1)

    align_color_img = np.zeros((depth_height,depth_width, 4), dtype=np.uint8)
    align_color_img[:, :] = color_img[colorYs, colorXs, :]

    return align_color_img

def create_color_point_cloud(align_color_img, depth_img, 
                             depth_scale, clipping_distance_in_meters, intrinsic):
    """
    Takes a RGBD frame and generates a point cloud from it
    """
    # Only get the first three channel
    align_color_img = align_color_img[:,:,0:3]
    # Convert opencv BGR to RGB
    align_color_img = align_color_img[...,::-1]
    aligned_img = align_color_img
    rgbd_image = open3d.geometry.RGBDImage.create_from_color_and_depth(
        open3d.geometry.Image(align_color_img.copy()), 
        open3d.geometry.Image(depth_img),
        depth_scale=1.0/depth_scale,
        depth_trunc=clipping_distance_in_meters,
        convert_rgb_to_intensity = False)
    pcd = open3d.geometry.PointCloud().create_from_rgbd_image(rgbd_image, intrinsic)

    return pcd.points, pcd.colors, aligned_img
