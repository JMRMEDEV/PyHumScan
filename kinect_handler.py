import numpy as np
from PIL import Image
from pykinect2 import PyKinectRuntime, PyKinectV2

kinect = None
color_w = None
color_h = None

def init_kinect():
    """
    Start sensor and its data source readers.
    """
    # Kinect runtime object, we want only color and body frames 
    global kinect
    kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)
    # Set the width and height (Default: 1920, 1080)
    global color_w
    color_w = kinect.color_frame_desc.Width
    global color_h
    color_h = kinect.color_frame_desc.Height
    print("Kinect ready\n\n")

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
    if kinect.has_new_color_frame():
        color_frame = kinect.get_last_color_frame()
        color_img = color_frame.reshape(((color_h, color_w, 4))).astype(np.uint8)
        color_img = Image.fromarray(color_img)
        # Image resize for deeplabcut-live performance
        color_img = color_img.resize((600, 338))
        color_img = np.array(color_img)
        return color_img
    else:
        return None

def read_color_data():
    """
    Waits for sensor to acquire a color frame and it transforms
    it for deeplabcut-live integration
    """
    color_frame = handle_color_frame()

    return color_frame    
