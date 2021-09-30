import keyboard

from pose_detector_2d import dlcl_start, dlcl_stop, get_2d_poses
from pose_detector_3d import get_3d_poses
from utils.file_utils import save_data
from utils.system_utils import clear
from kinect_handler import init_kinect, stop_kinect, read_color_data

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
    print("\n\nPlease, press 's' for starting data acquiring, press 'q' (focusing window) to quit\n\n")
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
                poses_3d = get_3d_poses(dlcl_cfg, data_save_cfg, result, color_dictionary, depth_frame, kinect)
                #save_data(dlcl_cfg, testing, result['poses_list'], result['frames'], data_save)
        stop_kinect()
        dlcl_stop()
        print("\n\nData aquiring stopped\n\n")
        print("\n\nOperation Completed\n\n")

if __name__ == "__main__":
    main()
