import numpy as np
from PIL import Image
from pykinect2 import PyKinectRuntime, PyKinectV2

kinect = None
color_w = None
color_h = None
depth_w = None
depth_h = None

def init_kinect():
    """
    Initializes the sensor and its data readers
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
    return kinect

def stop_kinect():
    """
    Disposes the sensor.
    """
    kinect.close()

def handle_color_frame():
    """
    Acquire color frame from reader and transforms it for
    its correct analysis with deeplabcut-live
    """
    color_frame = kinect.get_last_color_frame()
    color_img = color_frame.reshape(((color_h, color_w, 4))).astype(np.uint8)
    # Image resize for deeplabcut-live performance
    color_img_resized = Image.fromarray(color_img)
    color_img_resized = color_img_resized.resize((600, 338))
    color_img_resized = np.array(color_img_resized)
    # Use the resized image for deeplabcut-live performance, 
    # keep the orginal one for 3D pose estimation
    color_dictionary = {'resized': color_img_resized, 'original': color_img}
    return color_dictionary

def handle_depth_frame():
    """
    Aquire depth frame
    """
    depth_frame = kinect.get_last_depth_frame()
    # The depth frame is not resized, since it has no interaction with
    # deeplab-cut live
    depth_img = depth_frame.reshape(((depth_h, depth_w))).astype(np.uint16)
    return depth_img

def read_color_data():
    """
    Waits for sensor to acquire a color frame and it transforms
    it for deeplabcut-live integration
    """
    if kinect.has_new_color_frame() and kinect.has_new_depth_frame:
        color_dictionary = handle_color_frame()
        depth_frame = handle_depth_frame()
    else:
        color_dictionary = None
        depth_frame = None
    return color_dictionary, depth_frame
