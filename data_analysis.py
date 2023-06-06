import argparse
import time
import pandas as pd
import os
from pathlib import Path
import numpy as np

from utils.torch_utils import time_synchronized
from ana_tools.statistics import Covrt_TXT2Dic, Cal_frame_stamp, Cal_sleep
from ana_tools.mkvedio import Read_vedio


def start():
    t0 = time_synchronized()
    save_EXCLfile, cal_matrix, show_path, thickness, show_counts, cal_sleep = opt.save_EXCLfile, opt.cal_matrix, opt.show_path, opt.thickness, opt.show_counts, opt.cal_sleep

    source_video_path = opt.source
    source_folder = Path(opt.coordi_folder)
    (source_folder / 'res').mkdir(parents=True, exist_ok=True)  # make dir
    # initialize fly_counts
    fly_counts = []

    coordi_dic = Covrt_TXT2Dic(
        source_video_path, save_EXCLfile, source_folder, cal_matrix)

    if save_EXCLfile:
        print(f"All lables excel file saved to {source_folder}/res")

    if cal_matrix:
        # Cal_save_matrix(source, fly_coordi_long, source_folder)
        print(
            f"Coordinate matrix txt file and figure saved to {source_folder}/res")

    if show_path or show_counts:
        Read_vedio(show_path, show_counts, source_video_path, coordi_dic,
                   source_folder, thickness, fly_counts)
        if show_counts:
            print(
                f"The number of flies which flew into the diet was saved to {source_folder}/res/fly_counts.txt")

    if cal_sleep:
        frame_stamp, dis_dic, last_center_dic = {}, {}, {}
        for frame in range(min([int(k) for k in coordi_dic.keys()]), max([int(k) for k in coordi_dic.keys()])):
            data = coordi_dic[str(frame)]
            frame_stamp, dis_dic, last_center_dic = Cal_frame_stamp(frame,
                                                                    data, frame_stamp, dis_dic, last_center_dic)

        sleep_mean, rest_mean, sleep_stay_time, rest_stay_time, sleep_break = Cal_sleep(
            max([int(k) for k in coordi_dic.keys()]), frame_stamp)

        with open(source_folder / 'res' / 'fly_sleep.txt', 'a') as f:
            f.write("frame_stamp: {}; \ndistance:{}; \nsleep_mean: {}; \nsleep_stay_time:{}; \nrest_mean: {}; \nrest_stay_time:{}; \nsleep_break:{} \n ".format(
                frame_stamp, dis_dic, sleep_mean, sleep_stay_time, rest_mean, rest_stay_time, sleep_break))
        print(
            f"The mean sleep time and relative info was saved to {source_folder}/res/fly_sleep.txt")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str,
                        default='data/images', help='Path of source vedio')

    parser.add_argument('--coordi-folder', type=str,
                        default='data/labels', help='Path of the folder with coordinate txts')

    parser.add_argument('--thickness', type=int,
                        default=3, help='Thickness of the bounding box strokes')
    parser.add_argument('--cal-matrix', action='store_true',
                        help='Calculate fly position matrix and generate heatmap.')

    parser.add_argument('--save-EXCLfile', action='store_true',
                        help='show path on video')
    parser.add_argument('--show-path', action='store_true',
                        help='Show path on video')
    parser.add_argument('--show-counts', action='store_true',
                        help='Show counts of fly which fly into the diet on video ad both save the TXT file')
    parser.add_argument('--cal-sleep', action='store_true',
                        help='sleep behavior')

    # parser.add_argument('--path_vedio', type=str,
    #                     default='data/vedio', help='The vedio you want to add path')

    opt = parser.parse_args()
    print(opt)

    start()
