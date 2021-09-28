import os
import csv
import cv2

from utils.system_utils import get_date


def save_data(dlcl_cfg, testing, poses_list, frames, data_save):
    """
    Saves information of the pose detection process, might be
    jpg images and/or comma separated values
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