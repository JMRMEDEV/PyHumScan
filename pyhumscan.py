import keyboard

from pose_detector_2d import dlcl_run
from pose_detector_3d import get_poses
from utils.file_utils import save_data
from utils.kinect_utils import get_align_color_image
from utils.system_utils import clear

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
    data_save = {'frames': False, 'csvs': False}
    # We set either if we are performing a test or not
    testing = False
    # We start the processes with 's'
    print("\n\nPlease, press 's' for starting data acquiring, press 'q' (focusing window) to quit\n\n")
    if keyboard.read_key() == 's':
        clear()
        print("\n\nHumanScan Backend\n\n")
        result = dlcl_run(dlcl_cfg, testing)

    if result is not None:
        poses_3d = get_poses(dlcl_cfg, result['poses_list'], result['color_frames'], result['depth_frames'])
        #save_data(dlcl_cfg, testing, result['poses_list'], result['frames'], data_save)
        print("\n\nOperation Completed\n\n")

if __name__ == "__main__":
    main()
