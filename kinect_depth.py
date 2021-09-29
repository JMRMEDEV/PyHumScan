import csv

import keyboard
import numpy as np
from open3d import *
from pykinect2 import PyKinectRuntime, PyKinectV2
from pykinect2.PyKinectV2 import *

kinect = None
color_w = None
color_h = None
depth_w = None
depth_h = None

def init_kinect():
    """
    Initializes the sensor and its readers
    """
    global kinect
    kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color |
                                            PyKinectV2.FrameSourceTypes_Depth)
    global color_w
    color_w = kinect.color_frame_desc.Width
    global color_h
    color_h = kinect.color_frame_desc.Height
    global depth_w
    depth_w = kinect.depth_frame_desc.Width
    global depth_h
    depth_h = kinect.depth_frame_desc.Height

def stop_kinect():
    """
    Disposes the sensor.
    """
    kinect.close()
    
def get_points():
    init_kinect()
    # Default kinect depth scale where 1 unit = 0.001 m = 1 mm
    depth_scale = 0.001
    # Set the maximum distance to display the point cloud data
    clipping_distance_in_meters = 3
    # Convert dist in mm to unit
    clipping_distance = clipping_distance_in_meters / depth_scale
    # Hardcode the camera intrinsic parameters for backprojection
    width=depth_w 
    height=depth_h
    ppx=260.166
    ppy=205.197
    fx=367.535
    fy=367.535

    cloud_count = 0

    # Open3D Visualization

    intrinsic = open3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, ppx, ppy)
    # To convert [x,y,z] -> [x.-y,-z]
    flip_transform = [[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]]

    while not keyboard.is_pressed('q'):
        # Get source data from sensor
        if kinect.has_new_depth_frame:

            depth_frame = kinect.get_last_depth_frame()

            # Reshape from 1D frame to 2D image
            depth_img   = depth_frame.reshape(((depth_h, depth_w))).astype(np.uint16)

            point_cloud = open3d.geometry.PointCloud().create_from_depth_image(
                open3d.geometry.Image(depth_img),
                intrinsic,
                depth_scale=1.0/depth_scale,
                depth_trunc=clipping_distance_in_meters
            )

            point_cloud.transform(flip_transform)

            output_cloud = np.asarray(point_cloud.points)

            headers = ['X', 'Y', 'Z']

            if output_cloud.size > 0 and cloud_count < 6:
                cloud_count += 1
                print("New Cloud: ", output_cloud, "\n\n")
                output_file = 'test_cloud_' + str(cloud_count) + '.csv'
                with open(output_file, 'w', encoding='UTF8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(headers)
                    writer.writerows(output_cloud)
    
    stop_kinect()
