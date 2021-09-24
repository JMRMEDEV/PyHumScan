import cv2
import numpy as np
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
    color_frame = kinect.get_last_color_frame()
    color_img = color_frame.reshape(((color_h, color_w, 4))).astype(np.uint8)
    # Resize (1080, 1920, 4) into 600 width variable height
    color_img = cv2.resize(color_img, (0,0), fx=0.312, fy=0.312)
    # Convert img to NumPy array
    color_img = np.array(color_img)
    return color_img

def read_color_data():
    """
    Waits for sensor to acquire a color frame and it transforms
    it for deeplabcut-live integration
    """
    while kinect.has_new_color_frame() == False:
        color_frame = handle_color_frame()

    return color_frame    