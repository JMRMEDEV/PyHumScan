import os

import numpy as np
import pandas as pd
import ruamel
from dlclive import DLCLive

live = None

def dlcl_start(dlcl_cfg):
    """
    Set deeplabcut-live initial configuration
    """

    global live
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

def dlcl_stop():
    """"
    Close tensorflow session
    """
    live.close()

def get_2d_poses(dlcl_cfg, color_dictionary, ):
    """
    Analyze DeepLabCut-live exported model on a image frames:
    Calculate poses' location in (x , y) coordinates
    """
    poses = []

    poses.append(live.init_inference(color_dictionary['resized']))
    poses.append(live.get_pose(color_dictionary['resized']))

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

    result = poses_list

    return result
