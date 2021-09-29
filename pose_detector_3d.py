import os
import csv
import cv2
from PIL import Image

from utils.system_utils import get_date


def get_poses(dlcl_cfg, poses_list, color_frames, depth_frames):
    """
    Uses all the retrieved data to calculate the 3d joint positions
    """
    for i in range(len(color_frames)):
        labeled_img = None
        if poses_list[i] is not None:
            for joint_key, joint_values in poses_list[i].items():
                # Calculates the coordinates of the detected joints for the
                # deeplabcut-live resized image
                x_resized = int(joint_values[0])
                y_resized = int(joint_values[1])
                # Uses the resize factor to determine the coordinates of the
                # detected joints, considering 
                x_original = int(x_resized*3.2)
                y_original = int(y_resized*3.2)
                # Draw circles into the original coloured image, with the
                # calculated joint locations
                labeled = cv2.circle(
                    # Image to draw into
                    color_frames[i]['original'],
                    # Circle center coordinates
                    (x_original, y_original),
                    # Circle radius
                    int(round(dlcl_cfg['display_radius']*3.2)), 
                    # Color to use for drawing
                    (252, 3, 152),
                    # thickness = -1 tells to cv to fill the circle
                    # with the color
                    thickness=-1
                )
                labeled_img = Image.fromarray(labeled)
        
        labeled_img.show()

    """
    # Save data as jpg images
    if data_save['frames']:

        out_dir = "./output/images/labeled/" + get_date()['date'] + "/" + get_date()['time'] + "/"
        # If the specified path doesn't exist, create it
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        if not testing:

            for i in range(len(frames)):
                if poses_list[i] is not None:
                    for joint_key, joint_values in poses_list[i].items():
                        x = int(round(joint_values[0]))
                        y = int(round(joint_values[1]))
                        frame = cv2.circle(
                            frames[i], (x, y), dlcl_cfg['display_radius'], (255,0,0), thickness=-1
                        )
                        out_img_file = os.path.normpath(f"{out_dir}DLCLIVE_LABELED_{i}.jpg")
                        cv2.imwrite(out_img_file, frame)

        else:

            for i in range(len(poses_list)):
                for joint_key, joint_values in poses_list[i].items():
                    x = int(round(joint_values[0]))
                    y = int(round(joint_values[1]))
                    frame = cv2.circle(
                        frames[i], (x, y), dlcl_cfg['display_radius'], (255,0,0), thickness=-1
                    )
                    out_img_file = os.path.normpath(f"{out_dir}DLCLIVE_LABELED_test.jpg")
                    cv2.imwrite(out_img_file, frame)

    # Save data as csv
    if data_save['csvs']:

        if not testing:

            out_dir = "./output/csv/labeled/" + get_date()['date'] + "/" + get_date()['time'] + "/"

            # If the specified path doesn't exist, create it
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            headers = ['joint_name', 'x', 'y']
            
            for i in range(len(poses_list)):
                joints = []
                for joint_key, joint_values in poses_list[i].items():
                    out_csv_file = os.path.normpath(f"{out_dir}DLCLIVE_LABELED_{i}.csv")
                    joint = joint_key
                    x = int(round(joint_values[0]))
                    y = int(round(joint_values[1]))
                    joint_info = [joint, x, y]
                    joints.append(joint_info)
                    with open(out_csv_file, 'w', encoding='UTF8', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(headers)
                        writer.writerows(joints)
    """