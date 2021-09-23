from dlc_live_frame import benchmark_frames
import numpy as np
from time import sleep
from os import system, name
import cv2
from pykinect2 import PyKinectRuntime
from pykinect2 import PyKinectV2

def clear():
    # for windows
    if name == 'nt':
        _ = system('cls')
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = system('clear')

def init_sensor():
    # Kinect runtime object, we want only color and body frames 
    kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)
    # Default: 1920, 1080
    color_width, color_height = kinect.color_frame_desc.Width, kinect.color_frame_desc.Height
    while kinect.has_new_color_frame() == False:
        print("Waiting for kinect frame")
        sleep(0.1)
        clear()
    if kinect.has_new_color_frame():
        color_frame = kinect.get_last_color_frame()
        color_img = color_frame.reshape(((color_height, color_width, 4))).astype(np.uint8)
        # Resize (1080, 1920, 4) into 600 width variable height
        color_img = cv2.resize(color_img, (0,0), fx=0.312, fy=0.312)
        cv2.imwrite("./output/images/kinect/kcolor.jpg", color_img)  
    return color_img

def main():
    color_frame = init_sensor()
    print("\n\nHumanScan Backend\n\n")
    model_path = "./dlc-media/DLC_human_dancing_resnet_101_iteration-0_shuffle-1"
    poses = benchmark_frames(model_path, color_frame, save_frame=True, test=True)
    print("\n\nOperation Success\n\n", poses)

if __name__ == "__main__":
    main()