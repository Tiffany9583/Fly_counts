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
