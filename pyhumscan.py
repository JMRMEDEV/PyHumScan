import cv2
import keyboard

from kinect_handler import init_kinect, read_color_data, stop_kinect
from pose_detector_2d import dlcl_start, dlcl_stop, get_2d_poses
from pose_detector_3d import get_3d_poses
from utils.render_utils import render_3d
from utils.system_utils import clear


def main():
    """
    Starts the whole project logic execution.
    """
    # Set the configuration that will be used with deeplabcut-live
    dlcl_cfg = {
        'model_path': "./dlc-resources/DLC_human_dancing_resnet_101_iteration-0_shuffle-1",
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
    print("\n\nPlease, press 's' for starting data acquiring, hold 'q' to quit (until execution stops)\n\n")
    if keyboard.read_key() == 's':
        # Clear the console
        clear()
        print("\n\nHumanScan Backend\n\n")
        # Initialize services
        kinect = init_kinect()
        dlcl_start(dlcl_cfg)
        # We loop until we keep pressed 'q'
        while not keyboard.is_pressed('q'):
            color_dictionary = None
            depth_frame = None
            # Force kinect to keep trying to acquire data
            while color_dictionary is None and depth_frame is None:
                color_dictionary, depth_frame = read_color_data()
            # Once the data is retrieved, get the related 2d poses
            result = get_2d_poses(dlcl_cfg, color_dictionary)
            # We only retrieve 3d poses if we succesfully calculated a 2d pose
            if result is not None:
                # Aquire 3d poses and the cloud related with it
                poses_3d, simple_cloud = get_3d_poses(dlcl_cfg, data_save_cfg, result, color_dictionary, depth_frame, kinect)
                # Render the pose based on the cloud
                render_3d(simple_cloud)
                # In here, we can choose what to do with the 3d pose data
                # print("\n\n", poses_3d, "\n\n")
        # We stop all the services
        stop_kinect()
        dlcl_stop()
        cv2.destroyAllWindows()
        print("\n\nData aquiring stopped\n\n")
        print("\n\nOperation Completed\n\n")

if __name__ == "__main__":
    main()
