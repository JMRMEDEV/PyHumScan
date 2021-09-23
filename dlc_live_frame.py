import os
import time

import colorcet as cc
import cv2
import numpy as np
import ruamel
from dlclive import DLCLive
from PIL import Image, ImageColor
from tqdm import tqdm
import pandas as pd

def benchmark_frames(
    model_path,
    color_frame,
    tf_config=None,
    resize=None,
    pixels=None,
    cropping=None,
    dynamic=(False, 0.5, 10),
    n_frames=1000,
    print_rate=False,
    display=False,
    pcutoff=0.0,
    display_radius=3,
    cmap="bmy",
    save_poses=False,
    save_frame=False,
    output=None,
    test=False
):
    """
    Analyze DeepLabCut-live exported model on a image frames:
    Calculate poses' location in x,y coordinates

    Returns
    -------
    dictionary
        {
            forehead: (x,y), 
            chin: (x,y),
            shoulder1: (x,y),
            shoulder2: (x,y),
            elbow1: (x,y),
            elbow2: (x,y),
            wrist1: (x,y),
            wrist2: (x,y),
            hip1: (x,y),
            hip2: (x,y),
            knee1: (x,y),
            knee2: (x,y),
            ankle1: (x,y),
            ankle2: (x,y)
        }
    """

    # Get current time for file naming

    t = time.localtime()
    current_time = time.strftime("%H-%M-%S", t)


    if test:
        # Open image from local source
        image = Image.open("./dlc-lf-media/human_frame.jpg")

        # Resize the image for optimal performance keeping aspect ratio
        basewidth = 600
        wpercent = (basewidth/float(image.size[0]))
        hsize = int((float(image.size[1])*float(wpercent)))
        rezised = image.resize((basewidth,hsize), Image.ANTIALIAS)

        # Get image dimensions
        width, height = image.size
        im_size = (width, height)
        # Convert image to NumPy array
        frame = np.array(rezised)
    else:
        # Convert image to NumPy array
        frame = np.array(color_frame)

    n_frames = 1

    inf_times = np.zeros(n_frames)

    poses = []

    live = DLCLive(
        model_path,
        tf_config=tf_config,
        resize=resize,
        cropping=cropping,
        dynamic=dynamic,
        display=display,
        pcutoff=pcutoff,
        display_radius=display_radius,
        display_cmap=cmap,
    )

    poses.append(live.init_inference(frame))

    iterator = range(n_frames) if (print_rate) or (display) else tqdm(range(n_frames))
    for i in iterator:

        start_pose = time.time()
        poses.append(live.get_pose(frame))
        inf_times[i] = time.time() - start_pose

        if save_frame:

            all_colors = getattr(cc, cmap)
            colors = [
                ImageColor.getcolor(c, "RGB")[::-1]
                for c in all_colors[:: int(len(all_colors) / poses[-1].shape[0])]
            ]

            out_dir = "./output/images/labeled/"
            out_img_base = os.path.basename("human_frame")
            out_img_file = os.path.normpath(
                f"{out_dir}/{os.path.splitext(out_img_base)[i]}_{current_time}_DLCLIVE_LABELED.jpg"
            )

            this_pose = poses[-1]
            for j in range(this_pose.shape[0]):
                if this_pose[j, 2] > pcutoff:
                    x = int(this_pose[j, 0])
                    y = int(this_pose[j, 1])
                    frame = cv2.circle(
                        frame, (x, y), display_radius, colors[j], thickness=-1
                    )

            if resize is not None:
                frame = cv2.resize(frame, im_size)

            cv2.imwrite(out_img_file, frame)    

        if print_rate:
            print("pose rate = {:d}".format(int(1 / inf_times[i])))

    if print_rate:
        print("mean pose rate = {:d}".format(int(np.mean(1 / inf_times))))

    ### Close tensorflow session
    live.close()

    cfg_path = os.path.normpath(f"{model_path}/pose_cfg.yaml")
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
    human_poses = {
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

    if save_frame:
        out_dir = "./output/csv/"
        # Save data as csv
        out_vid_base = os.path.basename("human_model")
        pose_df.to_csv(
            os.path.normpath(
                f"{out_dir}/{os.path.splitext(out_vid_base)[0]}_{current_time}_DLCLIVE_POSES.csv"
            )
        )

    return human_poses
