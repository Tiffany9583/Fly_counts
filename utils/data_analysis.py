import argparse
import time
import glob
import pandas as pd
import os
from pathlib import Path
import numpy as np
import cv2

from utils.torch_utils import time_synchronized
from utils.plots import plot_fly_coordi_matrix


def covrt_TXT2Dic(save_EXCLfile, source_folder):

    # csv files in the path
    file_list = glob.glob(str(source_folder) + "/*.txt")

    # {track.track_id:[(Center_x,Center_y),(Center_x2,Center_y2)...]}
    fly_coordi = {}
    diet_coordi = {}

    # list of excel files we want to merge.
    excl_list = []

    for file in file_list:
        data = pd.read_csv(file, sep=";", header=None)

        targets = ["frame: ", " track: ", " class: ",
                   " BBox X \(xmin, ymin\): ", " Center \(x,y\): "]
        for i in range(len(targets)):
            data[i] = data[i].str.replace(targets[i], '', regex=True)

        data.columns = ["frame", "track", "class",
                        "BBox X (xmin, ymin)", "Center (x,y)"]
        # ======== add coordi to dic ========
        for i in range(len(data)):
            track_id = data["track"][i]
            Center = data["Center (x,y)"][i][1:-1]
            s = Center.split(",")
            Center = [float(i) for i in s]

            if str(data["class"][i]) == "Diet":
                if diet_coordi.get(track_id) == None:
                    diet_coordi[track_id] = []
                    diet_coordi[track_id] += [tuple(Center)]
                else:
                    diet_coordi[track_id] += [tuple(Center)]

            elif str(data["class"][i]) == "Fly":
                if fly_coordi.get(track_id) == None:
                    fly_coordi[track_id] = []
                    fly_coordi[track_id] += [tuple(Center)]
                else:
                    fly_coordi[track_id] += [tuple(Center)]

        if save_EXCLfile:
            excl_list.append(data)

    if save_EXCLfile:
        # create a new dataframe to store the merged excel file.
        excl_merged = pd.DataFrame()

        for excl_file in excl_list:
            # appends the data into the excl_merged dataframe.
            excl_merged = pd.concat(
                [excl_merged, excl_file], ignore_index=True)

        # exports the dataframe into excel file with specified name.
        columns = ["frame", "track", "class",
                   "BBox X (xmin, ymin)", "Center (x,y)"]
        excl_merged.to_excel(source_folder / 'res' / 'all_files.xlsx',
                             index=False, columns=columns)

    return fly_coordi, diet_coordi


def cal_save_matrix(source, fly_coordi, source_folder):
    # ======== calculate fly_coordi_matrix ========
    # Read video and first image to plot matrix
    cap = cv2.VideoCapture(source)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        first_img = frame
        cap.release()

    im0_shape = (first_img.shape[1], first_img.shape[0])

    if fly_coordi_matrix.shape != im0_shape:
        # Initialize fly_coordi_matrix
        fly_coordi_matrix = np.zeros(im0_shape)
    else:
        # add fly coordi into fly_coordi_matrix
        for track_id in fly_coordi:
            for coordi in fly_coordi[track_id]:
                fly_coordi_matrix[int(coordi[0])-1][int(coordi[1])-1] += 1

    plot_fly_coordi_matrix(fly_coordi_matrix, source,
                           save_dir=source_folder / 'res', first_img=first_img)


def start():
    t0 = time_synchronized()
    save_EXCLfile, cal_matrix = opt.save_EXCLfile, opt.cal_matrix

    source_folder = Path(opt.source)
    source = Path(opt.coordi_folder)
    (source_folder / 'res' if save_EXCLfile else source_folder).mkdir(parents=True,
                                                                      exist_ok=True)  # make dir

    fly_coordi, diet_coordi = covrt_TXT2Dic(save_EXCLfile, source_folder)

    if save_EXCLfile:
        print(f"All lables excel file saved to {source_folder}/res")

    if cal_matrix:
        cal_save_matrix(source, fly_coordi, save_EXCLfile, source_folder)
        print(f"Coordinate matrix txt file and \
              figure saved to {source_folder}/res")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str,
                        default='data/images', help='Path of source vedio')
    parser.add_argument('--coordi-folder', type=str,
                        default='data/labels', help='Path of the folder with coordinate txts')

    parser.add_argument('--thickness', type=int,
                        default=3, help='Thickness of the bounding box strokes')

    # function for fly project
    parser.add_argument('--save-EXCLfile', action='store_true',
                        help='show path on video')

    parser.add_argument('--show-path', action='store_true',
                        help='show path on video')
    parser.add_argument('--cal-matrix', action='store_true',
                        help='Calculate fly position matrix and generate heatmap.')

    opt = parser.parse_args()
    print(opt)

    start()
