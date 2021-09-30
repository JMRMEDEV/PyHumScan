import os
import cv2
import csv

from utils.system_utils import get_date


def save_data(data_save_cfg, data):
    """
    """
    out_dir = "./output/" + get_date()['date'] + "/"
    # If the specified path doesn't exist, create it
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if(data_save_cfg['original']):
        out_img_file = os.path.normpath(f"{out_dir}color_original_{get_date()['time']}.jpg")
        cv2.imwrite(out_img_file, data)
        print("\n\nOriginal color frame (image) saved\n\n")

    if(data_save_cfg['labeled_resized']):
        out_csv_file = os.path.normpath(f"{out_dir}joints_resized_{get_date()['time']}.csv")
        headers = ['joint_name', 'x', 'y']
        with open(out_csv_file, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(data)
        print("\n\nResized joint info saved\n\n")

    if(data_save_cfg['labeled_original']):
        print(data_save_cfg['labeled_original'])
        # if(data_save_cfg['labeled_resized']['image']):
        #     out_img_file = os.path.normpath(f"{out_dir}color_original_labeled_{get_date()['time']}.jpg")
        #     cv2.imwrite(out_img_file, data['image'])
        #     print("\n\nOriginal labaled color frame (image) saved\n\n")
        # if(data_save_cfg['labeled_resized']['csv']):
        #     out_csv_file = os.path.normpath(f"{out_dir}joints_original_{get_date()['time']}.csv")
        #     headers = ['joint_name', 'x', 'y']
        #     with open(out_csv_file, 'w', encoding='UTF8', newline='') as f:
        #         writer = csv.writer(f)
        #         writer.writerow(headers)
        #         writer.writerows(data['joints'])
        #     print("\n\nOriginal joint info saved\n\n")

    if(data_save_cfg['labeled_aligned']):
        out_img_file = os.path.normpath(f"{out_dir}aligned_{get_date()['time']}.jpg")
        cv2.imwrite(out_img_file, data)
        print("\n\nAligned (depth+color) frame (image) saved\n\n")

    if(data_save_cfg['colored_points']):
        print(data_save_cfg['colored_points'])
        headers = ['X', 'Y', 'Z', 'R', 'G', 'B']
        # if(data_save_cfg['colored_points']['pointcloud']):
        #     out_txt_file = os.path.normpath(f"{out_dir}pointcloud_{get_date()['time']}.txt")
        #     with open(out_txt_file, 'w', encoding='UTF8', newline='') as f:
        #         writer = csv.writer(f)
        #         writer.writerow(headers)
        #         writer.writerows(data)
        # if(data_save_cfg['colored_points']['csv']):
        #     out_txt_file = os.path.normpath(f"{out_dir}pointcloud_{get_date()['time']}.csv")
        #     with open(out_txt_file, 'w', encoding='UTF8', newline='') as f:
        #         writer = csv.writer(f)
        #         writer.writerow(headers)
        #         writer.writerows(data)

