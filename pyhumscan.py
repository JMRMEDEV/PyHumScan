import keyboard

from pose_detector import dlcl_run
from utils.file_utils import save_data
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
    data_save = {'frames': True, 'csvs': False}
    # We set either if we are performing a test or not
    testing = False
    # We start the processes with 's'
    print("\n\nPlease, press 's' for starting color acquiring, press 'q' (focusing window) to quit\n\n")
    if keyboard.read_key() == 's':
        clear()
        print("\n\nHumanScan Backend\n\n")
        result = dlcl_run(dlcl_cfg, testing)

    if result is not None:
        save_data(dlcl_cfg, testing, result['poses_list'], result['frames'], data_save)
        print("\n\nOperation Completed\n\n")

if __name__ == "__main__":
    main()