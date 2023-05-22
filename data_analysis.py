import argparse
import time
import glob
import pandas as pd
import os
from pathlib import Path
import numpy as np
import cv2

from utils.torch_utils import time_synchronized
from utils.plots import plot_fly_coordi_matrix, plot_path_line, plot_counts_text
from utils.general_fly import get_color_for, fly_not_in_diet, flyin


def Covrt_TXT2Dic(save_EXCLfile, source_folder):

    # csv files in the path
    file_list = glob.glob(str(source_folder) + "/*.txt")

    # {track.track_id:[(Center_x,Center_y),(Center_x2,Center_y2)...]}
    fly_coordi_long = {}
    # diet_coordi_long = {}

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

            # if str(data["class"][i]) == "Diet":
            #     if diet_coordi_long.get(track_id) == None:
            #         diet_coordi_long[track_id] = []
            #         diet_coordi_long[track_id] += [tuple(Center)]
            #     else:
            #         diet_coordi_long[track_id] += [tuple(Center)]

            if str(data["class"][i]) == "Fly":
                if fly_coordi_long.get(track_id) == None:
                    fly_coordi_long[track_id] = []
                    fly_coordi_long[track_id] += [tuple(Center)]
                else:
                    fly_coordi_long[track_id] += [tuple(Center)]

        excl_list.append(data)

    # create a new dataframe to store the merged excel file.
    excl_merged = pd.DataFrame()

    for excl_file in excl_list:
        # appends the data into the excl_merged dataframe.
        excl_merged = pd.concat(
            [excl_merged, excl_file], ignore_index=True)

    # exports the dataframe into excel file with specified name.
    columns = ["frame", "track", "class",
               "BBox X (xmin, ymin)", "Center (x,y)"]

    if save_EXCLfile:
        excl_merged.to_excel(source_folder / 'res' / 'all_files.xlsx',
                             index=False, columns=columns)

    return excl_merged, fly_coordi_long


def Cal_save_matrix(source, fly_coordi_long, source_folder):
    # ======== calculate fly_coordi_matrix ========
    # Read video and first image to plot matrix
    fly_coordi_matrix = np.array([])
    cap = cv2.VideoCapture(source)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        first_img = frame
        cap.release()

    im0_shape = (first_img.shape[1], first_img.shape[0])
    fly_coordi_matrix = np.zeros(im0_shape)
    # add fly coordi into fly_coordi_matrix
    for track_id in fly_coordi_long:
        for coordi in fly_coordi_long[track_id]:
            fly_coordi_matrix[int(coordi[0])-1][int(coordi[1])-1] += 1

    plot_fly_coordi_matrix(fly_coordi_matrix, source,
                           save_dir=source_folder / 'res', first_img=first_img)


def Read_vedio(show_path, show_counts, source, coordi_data, source_folder, thickness, fly_counts):
    # {track.track_id: [(Center_x, Center_y), (Center_x2, Center_y2)...]}

    fly_coordi_short = {}
    # diet_coordi_short = {}

    frame_count = 0
    cap = cv2.VideoCapture(source)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = 'mp4v'  # output video codec
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_writer = cv2.VideoWriter(
        str(source_folder / 'res'/'Output.mp4'), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        try:
            # frame_list = [str(i) for i in range(frame_count)]
            # data = coordi_data.loc[coordi_data['frame'] .isin(frame_list)]
            data = coordi_data.loc[coordi_data['frame'] == str(frame_count)]
            data = data.reset_index(drop=True)
            diet__center = []  # [(Center_x,Center_y,radius)]

            for i in range(len(data)):
                track_id = data.loc[i, 'track']
                class_name = data.loc[i, 'class']
                Center = data.loc[i, 'Center (x,y)'][1:-1]
                s = Center.split(",")

                if str(data["class"][i]) == "Diet":
                    # if diet_coordi_short.get(track_id) == None:
                    #     diet_coordi_short[track_id] = []
                    #     diet_coordi_short[track_id] += [tuple(Center)]
                    # else:
                    #     if tuple(Center) == diet_coordi_short[track_id][-1]:
                    #         continue
                    #     else:
                    #         diet_coordi_short[track_id] += [tuple(Center)]
                    # ======== make "diet__center" list========
                    Center = [float(i) for i in s]

                    xymin = data["BBox X (xmin, ymin)"][i][1:-1]
                    s1 = xymin.split(",")
                    xymin = [float(i) for i in s1]

                    radius = Center[0]-xymin[0]
                    diet__center.append((Center[0], Center[1], radius))

                elif str(data["class"][i]) == "Fly":
                    Center = [float(i) for i in s]
                    print("A", Center)
                    if show_counts:
                        xymin = data["BBox X (xmin, ymin)"][i][1:-1]
                        s1 = xymin.split(",")
                        xymin = [float(i) for i in s1]
                        fly_radius = Center[0]-xymin[0]

                        fly_counts = Cal_count(fly_counts, diet__center,
                                               Center, fly_radius, fly_coordi_short)

                    # ======== update fly coordinate for print fly path line========
                    # put fly coordinate into fly_coordi
                    Center = (int(Center[0]), int(Center[1]))
                    if fly_coordi_short.get(track_id) == None:
                        fly_coordi_short[track_id] = []
                        fly_coordi_short[track_id] += [tuple(Center)]
                        print("C")
                    else:
                        if tuple(Center) == fly_coordi_short[track_id][-1]:
                            print("C1")
                            continue
                        else:
                            fly_coordi_short[track_id] += [tuple(Center)]
                            print("C2")
                    if show_path:
                        label = f'{class_name} #{track_id}'
                        if fly_coordi_short.get(track_id):
                            fly_coordi_list = fly_coordi_short[track_id]
                            plot_path_line(fly_coordi_list, frame, color=get_color_for(
                                label), line_thickness=thickness)
                            print("D")

        except:
            pass
        if show_counts:
            plot_counts_text(frame, fly_numbers=fly_counts,
                             line_thickness=thickness)
            with open(source_folder / 'res' / 'fly_counts.txt', 'a') as f:
                f.write("frame: {}; fly_counts:{}\n".format(
                    frame_count, fly_counts))

        frame_count = frame_count + 1
        print(f" Processing  {frame_count}/{video_length}")
        vid_writer.write(frame)

    cap.release()
    vid_writer.release()


def Cal_count(fly_counts, diet__center, fly_Center, fly_radius, fly_coordi_short):
    # ======== calculate fly_counts ========
    if len(fly_counts) < len(diet__center):
        for i in range(len(diet__center)-len(fly_counts)):
            fly_counts.append(0)  # initialize fly_counts

    for i in range(len(diet__center)):
        diet_coordi = diet__center[i]

        # if flyin
        if flyin(tuple(fly_Center), (diet_coordi[0], diet_coordi[1]), diet_coordi[2], fly_radius) == True:
            # fly_coordi= {} # {track.track_id:[(Center_x,Center_y),(Center_x2,Center_y2)...]}
            # print("flyin")
            if len(fly_coordi_short) > 0:

                if fly_not_in_diet(tuple(fly_Center), fly_coordi_short) == True:
                    # print("flyin and  add")
                    fly_counts[i] += 1
                else:
                    # print("pass")
                    pass
            else:
                # print("flyin and  add")
                fly_counts[i] += 1

    return fly_counts


def start():
    t0 = time_synchronized()
    save_EXCLfile, cal_matrix, show_path, thickness, show_counts = opt.save_EXCLfile, opt.cal_matrix, opt.show_path, opt.thickness, opt.show_counts

    source = opt.source
    source_folder = Path(opt.coordi_folder)
    (source_folder / 'res' if save_EXCLfile else source_folder).mkdir(parents=True,
                                                                      exist_ok=True)  # make dir
    # initialize fly_counts
    fly_counts = []

    coordi_data, fly_coordi_long = Covrt_TXT2Dic(
        save_EXCLfile, source_folder)

    if save_EXCLfile:
        print(f"All lables excel file saved to {source_folder}/res")

    if cal_matrix:
        Cal_save_matrix(source, fly_coordi_long, source_folder)
        print(
            f"Coordinate matrix txt file and figure saved to {source_folder}/res")

    if show_path or show_counts:
        Read_vedio(show_path, show_counts, source, coordi_data,
                   source_folder, thickness, fly_counts)
        if show_counts:
            print(
                f"The number of flies which flew into the diet was saved to {source_folder}/res/fly_counts.txt")

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
                        help='show path on video')
    parser.add_argument('--show-counts', action='store_true',
                        help='show counts of fly which fly into the diet on video ad both save the TXT file')

    # parser.add_argument('--path_vedio', type=str,
    #                     default='data/vedio', help='The vedio you want to add path')

    opt = parser.parse_args()
    print(opt)

    start()
