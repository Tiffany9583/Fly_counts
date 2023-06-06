from math import dist

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


def Flyin(fly_center, diet__center, diet_radius, fly_radius):

    ori = diet_radius+fly_radius
    new = dist(
        (0, 0), (abs(fly_center[0]-diet__center[0]), abs(fly_center[1]-diet__center[1])))
    # new = ()**2 + \
    #     ()**2
    if ori >= new:
        return True
    else:
        return False


def fly_not_in_diet(fly_center, in_diet_fly_center):
    # if fly not in diet befor
    d = []
    if len(in_diet_fly_center) == 0:
        return True
    else:
        for coordi in in_diet_fly_center.values():

            if dist(fly_center, coordi) <= 1:
                d.append(1)
            else:
                d.append(0)
        if 1 not in d:
            return True


# determine fly is sleep or not
def Sleep_or_rest(track_id, sleep, stay_time, sleep_stay_time, rest_stay_time, sleep_break):
    if stay_time >= 10:
        # if stay time up to 10 sec determine this fly is sleeping
        sleep_stay_time.setdefault(track_id, []).append(stay_time)
        sleep_upate = True

    else:
        # if rest
        rest_stay_time.setdefault(track_id, []).append(stay_time)
        sleep_upate = False

    if sleep == True and sleep_upate == False:
        sleep_break.setdefault(track_id, 0)
        sleep_break[track_id] += 1

    sleep = sleep_upate

    return sleep_stay_time, rest_stay_time, sleep_break, sleep
