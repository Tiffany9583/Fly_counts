import argparse
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from math import dist


from utils.plots import plot_one_box, plot_fly_coordi_matrix, plot_counts_text, plot_path_line


# Settings
classes = []
names = []

# {track.track_id:[(Center_x,Center_y),(Center_x2,Center_y2)...]}
fly_coordi = {}


def get_color_for(class_num):
    colors = [
        "#4892EA",
        "#00EEC3",
        "#FE4EF0",
        "#F4004E",
        "#FA7200",
        "#EEEE17",
        "#90FF00",
        "#78C1D2",
        "#8C29FF"
    ]

    num = hash(class_num)  # may actually be a number or a string
    hex = colors[num % len(colors)]

    # adapted from https://stackoverflow.com/questions/29643352/converting-hex-to-rgb-value-in-python
    rgb = tuple(int(hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

    return rgb


def flyin(fly_center, diet__center, diet_radius, fly_radius):
    ori = diet_radius+fly_radius
    new = dist(
        (0, 0), (abs(fly_center[0]-diet__center[0]), abs(fly_center[1]-diet__center[1])))
    # new = ()**2 + \
    #     ()**2
    if ori >= new:
        return True


def fly_not_in_diet(fly_center, old_coordi):
    d = []
    for coordi in old_coordi.values():
        if dist(fly_center, coordi[-1]) <= 1:
            d.append(1)
        else:
            d.append(0)
    if 1 not in d:
        return True


def update_tracks(detection_engine, tracker, frame_count, save_txt, txt_path, save_img, view_img, im0, gn,
                  fly_counts, fly_coordi_matrix, thickness, show_path, info, cal_matrix):
    diet__center = []  # [(Center_x,Center_y,radius)]

    if len(tracker.tracks):
        print("[Tracks]", len(tracker.tracks))

    for track in tracker.tracks:

        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        xyxy = track.to_tlbr()
        class_num = track.class_num
        bbox = xyxy
        class_name = names[int(
            class_num)] if detection_engine == "yolov5" or "yolov7" else class_num
        # print(str(class_name),"bbox =", xyxy)
        Center_x = bbox[0]+(bbox[2]-bbox[0])*0.5
        Center_y = bbox[1]+(bbox[3]-bbox[1])*0.5
        fly_radius = ((bbox[2]-bbox[0])*0.5 + (bbox[3]-bbox[1])*0.5)*0.5

        if str(class_name) == "Diet":
            radius = (bbox[2]-bbox[0])*0.5
            diet__center.append((Center_x, Center_y, radius))

        if str(class_name) == "Fly":
            # diet__center = [(coordi_x,coordi_y,radius)]

            # ======== calculate fly_coordi_matrix ========
            if cal_matrix:
                im0_shape = (im0.shape[1], im0.shape[0])

                if fly_coordi_matrix.shape != im0_shape:
                    # Initialize fly_coordi_matrix
                    fly_coordi_matrix = np.zeros(im0_shape)
                else:
                    # add fly coordi into fly_coordi_matrix
                    fly_coordi_matrix[int(Center_x)-1][int(Center_y)-1] += 1

            # ======== calculate fly_counts ========
            if len(fly_counts) < len(diet__center):
                for i in range(len(diet__center)-len(fly_counts)):
                    fly_counts.append(0)  # initialize fly_counts

            for i in range(len(diet__center)):
                diet_coordi = diet__center[i]

                # if flyin
                if flyin((Center_x, Center_y), (diet_coordi[0], diet_coordi[1]), diet_coordi[2], fly_radius) == True:
                    # fly_coordi= {} # {track.track_id:[(Center_x,Center_y),(Center_x2,Center_y2)...]}
                    # print("flyin")
                    if len(fly_coordi) > 0:

                        if fly_not_in_diet((Center_x, Center_y), fly_coordi) == True:
                            # print("flyin and  add")
                            fly_counts[i] += 1

                        else:
                            # print("pass")
                            pass
                    else:
                        # print("flyin and  add")
                        fly_counts[i] += 1

             # ======== update fly coordinate for print fly path line========
             # put fly coordinate into fly_coordi
            if fly_coordi.get(track.track_id) == None:
                fly_coordi[track.track_id] = []
                fly_coordi[track.track_id] += [(Center_x, Center_y)]
            else:
                fly_coordi[track.track_id] += [(Center_x, Center_y)]

            label = f'{class_name} #{track.track_id}'
            if show_path:
                if fly_coordi.get(track.track_id):
                    fly_coordi_list = fly_coordi[track.track_id]
                    plot_path_line(fly_coordi_list, im0, color=get_color_for(
                        label), line_thickness=thickness)

        if info:
            print("Tracker ID: {}, Class: {}, BBox Coords (xmin, ymin, xmax, ymax): {}".format(
                str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

        if save_txt:  # Write to file
            # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            with open(txt_path + '.txt', 'a') as f:
                # f.write('frame: {}; track: {}; class: {}; bbox: {};\n'.format(frame_count, track.track_id, class_num,
                #                                                               *xywh))
                f.write('frame: {}; track: {}; class: {}; BBox X and Center (xmin, ymin, Center_x, Center_y): {};\n'.format(frame_count, track.track_id, class_num,
                        (int(bbox[0]), int(bbox[1]), Center_x, Center_y)))

        if save_img or view_img:  # Add bbox to image
            label = f'{class_name} #{track.track_id}'
            plot_one_box(xyxy, im0, label=label,
                         color=get_color_for(label), line_thickness=thickness)

    with open(txt_path + '.txt', 'a') as f:
        f.write("fly_counts:{}\n".format(fly_counts))

    plot_counts_text(im0, fly_numbers=fly_counts, line_thickness=thickness)

    fly_coordi.update(fly_coordi)

    return fly_counts, fly_coordi_matrix
