import os

import numpy as np
import pandas as pd
import ruamel
from dlclive import DLCLive
from PIL import Image
from kinect_handler import init_kinect, stop_kinect, read_color_data
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

def detect_poses(dlcl_cfg, testing, live=None):
    """
    Performs deeplabcut-live tensorflow operation
    """
    poses = []
    frames = []
    # We might want to perform a project test for the given model
    if testing:
        # Open image from local source
        image = Image.open("./dlc-lf-media/human_frame.jpg")
        # Resize the image for optimal performance keeping aspect ratio
        basewidth = 600
        wpercent = (basewidth/float(image.size[0]))
        hsize = int((float(image.size[1])*float(wpercent)))
        rezised = image.resize((basewidth,hsize), Image.ANTIALIAS)
        # Convert image to NumPy array
        frame = np.array(rezised)
        frames.append(frame)
        poses.append(live.init_inference(frame))
        poses.append(live.get_pose(frame))
    else:
        init_kinect()
        frame = read_color_data()
        frames.append(frame)
        poses.append(live.init_inference(frame))
        poses.append(live.get_pose(frame))

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
    pose_df = pd.DataFrame(poses, columns=pdindex)

    # We create a dictionary with the data that we are interested on
    poses_dict = {
        'forehead': (pose_df['forehead'].x[0].item(), pose_df['forehead'].y[0].item()),
        'chin': (pose_df['chin'].x[0].item(), pose_df['chin'].y[0].item()),
        'shoulder1': (pose_df['shoulder1'].x[0].item(), pose_df['shoulder1'].y[0].item()),
        'shoulder2': (pose_df['shoulder2'].x[0].item(), pose_df['shoulder2'].y[0].item()),
        'elbow1': (pose_df['elbow1'].x[0].item(), pose_df['elbow1'].y[0].item()),
        'elbow2': (pose_df['elbow2'].x[0].item(), pose_df['elbow2'].y[0].item()),
        'wrist1': (pose_df['wrist1'].x[0].item(), pose_df['wrist1'].y[0].item()),
        'wrist2': (pose_df['wrist2'].x[0].item(), pose_df['wrist2'].y[0].item()),
        'hip1': (pose_df['hip1'].x[0].item(), pose_df['hip1'].y[0].item()),
        'hip2': (pose_df['hip2'].x[0].item(), pose_df['hip2'].y[0].item()),
        'ankle1': (pose_df['ankle1'].x[0].item(), pose_df['ankle1'].y[0].item()),
        'ankle2': (pose_df['ankle2'].x[0].item(), pose_df['ankle2'].y[0].item()),
    }

    result = {'poses_df': poses_dict, 'frames': frames}

    return result
