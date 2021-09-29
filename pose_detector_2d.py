import os

import keyboard
import numpy as np
import pandas as pd
import ruamel
from dlclive import DLCLive
from PIL import Image

from kinect_handler import init_kinect, read_color_data, stop_kinect


def dlcl_run(
    dlcl_cfg,
    testing=False
):
    """
    Analyze DeepLabCut-live exported model on a image frames:
    Calculate poses' location in (x , y) coordinates
    """

    live = DLCLive(
        dlcl_cfg['model_path'],
        tf_config=dlcl_cfg['tf_config'],
        resize=dlcl_cfg['resize'],
        cropping=None,
        dynamic=dlcl_cfg['dynamic'],
        display=dlcl_cfg['display'],
        pcutoff=dlcl_cfg['pcutoff'],
        display_radius=dlcl_cfg['display_radius'],
        display_cmap=dlcl_cfg['cmap'],
    )

    poses = detect_poses(dlcl_cfg, testing, live)
    return poses

def detect_poses(dlcl_cfg, testing, live):
    """
    Performs deeplabcut-live tensorflow operation
    """
    poses = []
    color_frames = []
    depth_frames = []
    # We might want to perform a project test for the given model
    if testing:
        # Open image from local source
        image = Image.open("./dlc-lf-media/human_frame.jpg")
        # Resize the image for optimal performance keeping aspect ratio
        # basewidth = 600
        # wpercent = (basewidth/float(image.size[0]))
        # hsize = int((float(image.size[1])*float(wpercent)))
        # rezised = image.resize((basewidth,hsize), Image.ANTIALIAS)
        # Convert image to NumPy array
        # frame = np.array(rezised)
        # color_frames.append(frame)
        # poses.append(live.init_inference(frame))
        # poses.append(live.get_pose(frame))
    else:
        init_kinect()
        while not keyboard.is_pressed('q'):
            color_dictionary, depth_frame = read_color_data()
            if color_dictionary is not None and depth_frame is not None:
                color_frames.append(color_dictionary)
                depth_frames.append(depth_frame)
                poses.append(live.init_inference(color_dictionary['resized']))
                poses.append(live.get_pose(color_dictionary['resized']))
        print("\n\nData aquiring stopped\n\n")

    # Dispose Kinect sensor
    stop_kinect()

    # Close tensorflow session
    live.close()

    # We read the retrieved data as HDF format
    cfg_path = os.path.normpath(f"{dlcl_cfg['model_path']}/pose_cfg.yaml")
    ruamel_file = ruamel.yaml.YAML()
    dlc_cfg = ruamel_file.load(open(cfg_path, "r"))
    bodyparts = dlc_cfg["all_joints_names"]
    poses = np.array(poses)
    poses = poses.reshape((poses.shape[0], poses.shape[1] * poses.shape[2]))
    pdindex = pd.MultiIndex.from_product(
        [bodyparts, ["x", "y", "likelihood"]], names=["bodyparts", "coords"]
    )
    poses_df = pd.DataFrame(poses, columns=pdindex)

    poses_list = []

    for i in range(poses_df.shape[0]):
        # We create a dictionary with the data that we are interested on
        poses_dict = {
            'forehead': (poses_df['forehead'].x[i].item(), poses_df['forehead'].y[i].item()),
            'chin': (poses_df['chin'].x[i].item(), poses_df['chin'].y[i].item()),
            'shoulder1': (poses_df['shoulder1'].x[i].item(), poses_df['shoulder1'].y[i].item()),
            'shoulder2': (poses_df['shoulder2'].x[i].item(), poses_df['shoulder2'].y[i].item()),
            'elbow1': (poses_df['elbow1'].x[i].item(), poses_df['elbow1'].y[i].item()),
            'elbow2': (poses_df['elbow2'].x[i].item(), poses_df['elbow2'].y[i].item()),
            'wrist1': (poses_df['wrist1'].x[i].item(), poses_df['wrist1'].y[i].item()),
            'wrist2': (poses_df['wrist2'].x[i].item(), poses_df['wrist2'].y[i].item()),
            'hip1': (poses_df['hip1'].x[i].item(), poses_df['hip1'].y[i].item()),
            'hip2': (poses_df['hip2'].x[i].item(), poses_df['hip2'].y[i].item()),
            'ankle1': (poses_df['ankle1'].x[i].item(), poses_df['ankle1'].y[i].item()),
            'ankle2': (poses_df['ankle2'].x[i].item(), poses_df['ankle2'].y[i].item()),
        }
        # We add the created dictionary as the last elment of poses_list
        poses_list.append(poses_dict)

    result = {'poses_list': poses_list, 'color_frames': color_frames, 'depth_frames': depth_frames}

    return result
