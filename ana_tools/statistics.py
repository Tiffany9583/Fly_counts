import time
import glob
import os
from pathlib import Path
import numpy as np
import cv2
from math import dist
import json

from utils.torch_utils import time_synchronized
from utils.plots import plot_fly_coordi_matrix, plot_path_line, plot_counts_text
from utils.general_fly import get_color_for, fly_not_in_diet, Flyin, Sleep_or_rest


def Covrt_TXT2Dic(source, save_EXCLfile, source_folder, cal_matrix):

    # csv files in the path
    file_list = glob.glob(str(source_folder) + "/*.txt")

    # {track.track_id:[(Center_x,Center_y),(Center_x2,Center_y2)...]}
    fly_coordi_long = {}
    # diet_coordi_long = {}

    # list of excel files we want to merge.
    excl_list = []
    D = {}
    first_img, im0_shape, video_length = Get_fiset_frame(source)
    fly_coordi_matrix = np.zeros(im0_shape)
    """
D = {'98': [{
     'track': '57', 
     'class': 'Fly', 
     'BBox X (xmin, ymin)': '(1, 467)', 
     'Center (x,y)': '(4.0, 470.99950080273146)'
     }, ... ]
    }
    """
    for file in file_list:
        with open(file) as file:
            file_lis = []
            for line in file:
                d = {}
                items = []
                data = line.split('; ')

                for i in data:
                    items = i.split(': ')
                    items = list(filter(None, [x.strip('\n') for x in items]))
                    if items[0] == 'frame':
                        frame_number = items[1]
                    else:
                        d[items[0]] = items[1]
                    if len(d) == 4:

                        if cal_matrix and int(frame_number) % 30 == 0 and d['class'] == 'Fly':
                            # print("cal_matrix")
                            Center = d['Center (x,y)'][1:-1].split(",")
                            Center = [float(i) for i in Center]
                            fly_coordi_matrix[int(
                                Center[0])-1][int(Center[1])-1] += 1

                        file_lis.append(d)
            D[frame_number] = file_lis
            print(f" Data processing  {len(D)}/{video_length}")

    if cal_matrix:
        plot_fly_coordi_matrix(fly_coordi_matrix, source,
                               save_dir=source_folder / 'res', first_img=first_img)

    return D


def Get_fiset_frame(source):
    # ======== calculate fly_coordi_matrix ========
    # Read video and first image to plot matrix
    print("Get first frame")
    # fly_coordi_matrix = np.array([])
    cap = cv2.VideoCapture(source)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        first_img = frame
        cap.release()

    im0_shape = (first_img.shape[1], first_img.shape[0])
    return first_img, im0_shape, video_length


def Cal_in_diet_counts(track_id, fly_counts, diet__center, fly_Center, fly_radius, in_diet_fly_center):
    # ======== calculate fly_counts ========

    if len(fly_counts) < len(diet__center):
        for i in range(len(diet__center)-len(fly_counts)):
            fly_counts.append(0)  # initialize fly_counts

    for i in range(len(diet__center)):
        diet_coordi = diet__center[i]

        # if flyin
        if Flyin(tuple(fly_Center), (diet_coordi[0], diet_coordi[1]), diet_coordi[2], fly_radius) == True:
            # print(fly_Center, "in the diet")
            # fly_coordi= {} # {track.track_id:[(Center_x,Center_y),(Center_x2,Center_y2)...]}
            # print(in_diet_fly_center)
            if fly_not_in_diet(tuple(fly_Center), in_diet_fly_center) == True:
                # print("flyin and  add")
                fly_counts[i] += 1
            else:
                # print("pass")
                pass

            in_diet_fly_center[track_id] = tuple(fly_Center)
            # print(in_diet_fly_center)
            # print(fly_counts)

    return fly_counts, in_diet_fly_center


# ====================calulate sleep behavior====================
def Cal_frame_stamp(frame, data, frame_stamp, dis_dic, last_center_dic):
    for i in range(len(data)):
        dis = 0
        item_dic = data[i]
        track_id = item_dic['track']
        class_name = item_dic['class']

        Center = item_dic['Center (x,y)'][1:-1].split(",")
        Center = [float(i) for i in Center]

        if class_name == "Fly":
            if last_center_dic.get(track_id) == None:
                last_center_dic[track_id] = tuple(Center)

            else:
                last_center = last_center_dic[track_id]
                if dist(last_center, tuple(Center)) > 1:
                    # print(tuple(Center))
                    dis = dist(last_center, tuple(Center))
                    if dis_dic.get(track_id) == None:
                        dis_dic[track_id] = dis
                    else:
                        last_dis = dis_dic[track_id]
                        dis_dic[track_id] = last_dis + dis

                    frame_stamp.setdefault(track_id, []).append(frame)

            last_center_dic[track_id] = tuple(Center)

    return frame_stamp, dis_dic, last_center_dic


def Cal_sleep(video_length, frame_stamp):

    sleep_stay_time = {}
    # {track_id: [sec1,sec2....] }
    rest_stay_time = {}
    # {track_id: [sec1,sec2....] }

    sleep_break = {}
    # {track_id: counts }

    sleep_mean = {}
    rest_mean = {}

    for track_id in frame_stamp:
        sleep = False
        if len(frame_stamp[track_id]) == 1:
            # cal stay time (sec)
            stay_time = (video_length - int(frame_stamp[track_id][0]))*0.01111
            sleep_stay_time, rest_stay_time, sleep_break, sleep = Sleep_or_rest(
                track_id, sleep, stay_time, sleep_stay_time, rest_stay_time, sleep_break)

        else:
            for i in range(len(frame_stamp[track_id])-1):
                stay_time = (
                    int(frame_stamp[track_id][i+1]) - int(frame_stamp[track_id][i]))*0.01111
                sleep_stay_time, rest_stay_time, sleep_break, sleep = Sleep_or_rest(
                    track_id, sleep, stay_time, sleep_stay_time, rest_stay_time, sleep_break)

            stay_time = (video_length - int(frame_stamp[track_id][-1]))*0.01111
            sleep_stay_time, rest_stay_time, sleep_break, sleep = Sleep_or_rest(
                track_id, sleep, stay_time, sleep_stay_time, rest_stay_time, sleep_break)

    # use sleep_stay_time, rest_stay_time to cal mean
    for key in sleep_stay_time:
        sleep_mean[key] = sum(sleep_stay_time[key]) / len(sleep_stay_time[key])
    for key in rest_stay_time:
        rest_mean[key] = sum(rest_stay_time[key]) / len(rest_stay_time[key])

    return sleep_mean, rest_mean, sleep_stay_time, rest_stay_time, sleep_break
