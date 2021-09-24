import os

import cv2

from utils.system_utils import get_date, get_miliseconds


def save_data(dlcl_cfg, testing, poses_df, frames, data_save):
    """
    Saves information of the pose detection process, might be
    jpg images and/or comma separated values
    """
    # Save data as jpg images
    if data_save['frames']:

        out_dir = "./output/images/labeled/"

        if not testing:

            out_img_file = os.path.normpath(
                f"{out_dir}/{get_miliseconds()}_DLCLIVE_LABELED_.jpg"
            )
            for i in range(len(frames)):
                for joint_key, joint_values in poses_df.items():
                    x = int(round(joint_values[0]))
                    y = int(round(joint_values[1]))
                    frame = cv2.circle(
                        frames[i], (x, y), dlcl_cfg['display_radius'], (255,0,0), thickness=-1
                    )
                    cv2.imwrite(out_img_file, frame)

        else:

            out_img_file = os.path.normpath(
                f"{out_dir}/test_{get_date()}_DLCLIVE_LABELED_.jpg"
            )
            for i in range(len(frames)):
                for joint_key, joint_values in poses_df.items():
                    x = int(round(joint_values[0]))
                    y = int(round(joint_values[1]))
                    frame = cv2.circle(
                        frames[i], (x, y), dlcl_cfg['display_radius'], (255,0,0), thickness=-1
                    )
                    cv2.imwrite(out_img_file, frame)

    # Save data as csv
    # if data_save['csvs']:

    #     out_dir = "./output/csv/"

    #     if not testing:
    #         poses_df.to_csv(
    #             os.path.normpath(
    #                 f"{out_dir}/{get_miliseconds()}_DLCLIVE_POSES.csv"
    #             )
    #         )
    #     else:
    #         poses_df.to_csv(
    #             os.path.normpath(
    #                 f"{out_dir}/test_{get_date()}_DLCLIVE_POSES.csv"
    #             )
    #         )
