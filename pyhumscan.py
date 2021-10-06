import keyboard

from kinect_handler import init_kinect, read_color_data, stop_kinect
from pose_detector_2d import dlcl_start, dlcl_stop, get_2d_poses
from pose_detector_3d import get_3d_poses
from utils.system_utils import clear
import plotly.graph_objects as go
import numpy as np
import cv2
import io
from PIL import Image


def render_3d(joints_cloud):
    # Extract the columns
    x = [row[0] for row in joints_cloud]
    y = [row[1] for row in joints_cloud]
    z = [row[2] for row in joints_cloud]
    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z,
                                    mode='markers')])
    img_bytes = fig.to_image(format="jpg", width=1280, height=720)
    img = Image.open(io.BytesIO(img_bytes))
    img_arr = np.array(img)
    cv2.imshow('joints', img_arr)
    cv2.waitKey(1)

def main():
    """
    Starts the whole project logic execution.
    """
    # Set the configuration that will be used with deeplabcut-live
    dlcl_cfg = {
        'model_path': "./dlc-media/DLC_human_dancing_resnet_101_iteration-0_shuffle-1",
        'tf_config': None,
        'resize': None,
        'dynamic': (False, 0.5, 10),
        'display': False,
        'pcutoff': 0.0,
        'display_radius': 5,
        'cmap': "bmy"
    }
    # We choose which data we want to storage in case we want to do so
    data_save_cfg = {
        'original': True,
        'labeled_resized': True,
        'labeled_original': {'image': True, 'csv': True},
        'labeled_aligned': True,
        'colored_points': {'pointcloud': True, 'csv': True}
    }
    # We start the processes with 's'
    print("\n\nPlease, press 's' for starting data acquiring, press 'q' to quit\n\n")
    if keyboard.read_key() == 's':
        clear()
        print("\n\nHumanScan Backend\n\n")
        kinect = init_kinect()
        dlcl_start(dlcl_cfg)
        while not keyboard.is_pressed('q'):
            color_dictionary = None
            depth_frame = None
            while color_dictionary is None and depth_frame is None:
                color_dictionary, depth_frame = read_color_data()
            result = get_2d_poses(dlcl_cfg, color_dictionary)
            if result is not None:
                poses_3d, simple_cloud = get_3d_poses(dlcl_cfg, data_save_cfg, result, color_dictionary, depth_frame, kinect)
                render_3d(simple_cloud)
                # In here, we can choose what to do with the 3d pose data
                # print("\n\n", poses_3d, "\n\n")
        stop_kinect()
        dlcl_stop()
        cv2.destroyAllWindows()
        print("\n\nData aquiring stopped\n\n")
        print("\n\nOperation Completed\n\n")

if __name__ == "__main__":
    main()
