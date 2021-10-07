import csv
import os

import cv2

from utils.system_utils import get_date

out_dir = "./output/" + get_date()["date"] + "/"


def save_original(color_frame):
    out_img_file = os.path.normpath(f"{out_dir}color_original_{get_date()['time']}.jpg")
    cv2.imwrite(out_img_file, color_frame)
    print("\n\nOriginal color frame (image) saved\n\n")


def save_labeled_resized(joint_array):
    out_csv_file = os.path.normpath(
        f"{out_dir}joints_original_{get_date()['time']}.csv"
    )
    headers = ["joint_name", "x", "y"]
    with open(out_csv_file, "w", encoding="UTF8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(joint_array)
    print("\n\nOriginal joint (csv) info saved\n\n")


def save_labeled_original(labeled_color_frame):
    out_img_file = os.path.normpath(
        f"{out_dir}labeled_original_{get_date()['time']}.jpg"
    )
    cv2.imwrite(out_img_file, labeled_color_frame)
    print("\n\nOriginal labeled frame (image) saved\n\n")


def save_labeled_aligned(labeled_aligned_frame):
    out_img_file = os.path.normpath(
        f"{out_dir}labeled_aligned_{get_date()['time']}.jpg"
    )
    cv2.imwrite(out_img_file, labeled_aligned_frame)
    print("\n\nLabeled aligned (depth + color) frame (image) saved\n\n")


def save_colored_cloud(cloud):
    headers = ["X", "Y", "Z", "R", "G", "B"]
    # If the specified path doesn't exist, create it
    out_txt_file = os.path.normpath(
        f"{out_dir}pointcloud_filtered_{get_date()['time']}.txt"
    )
    with open(out_txt_file, "w", encoding="UTF8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(cloud)
    print("\n\nFiltered point cloud (txt) saved\n\n")


def save_pose_3d(pose_dict):
    out_csv_file = os.path.normpath(f"{out_dir}pose_3d_{get_date()['time']}.csv")
    poses_output = []
    for pose_key, joint_values in pose_dict.items():
        x = joint_values[0]
        y = joint_values[1]
        z = joint_values[2]
        pose_3d = [pose_key, x, y, z]
        poses_output.append(pose_3d)

    headers = ["joint_name", "x", "y", "z"]
    with open(out_csv_file, "w", encoding="UTF8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(poses_output)
    print("\n\n3D Pose (csv) saved\n\n")


def save_data(data_target, data):
    """ """

    # If the specified path doesn't exist, create it
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if data_target == "original":
        save_original(data)

    if data_target == "labeled_resized":
        save_labeled_resized(data)

    if data_target == "labeled_original":
        save_labeled_original(data)

    if data_target == "labeled_aligned":
        save_labeled_aligned(data)

    if data_target == "colored_cloud":
        save_colored_cloud(data)

    if data_target == "pose_3d":
        save_pose_3d(data)
