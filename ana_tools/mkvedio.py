import cv2

from utils.torch_utils import time_synchronized
from utils.plots import plot_fly_coordi_matrix, plot_path_line, plot_counts_text
from utils.general_fly import get_color_for
from ana_tools.statistics import Cal_in_diet_counts


def Read_vedio(show_path, show_counts, source, coordi_data, source_folder, thickness, fly_counts):
    # {track.track_id: [(Center_x, Center_y), (Center_x2, Center_y2)...]}

    fly_coordi_short = {}
    # diet_coordi_short = {}
    in_diet_fly_center = {}
    # {'tracker_id':(x,y)}

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

            data = coordi_data[str(frame_count)]
            diet__center = []  # [(Center_x,Center_y,radius)]
            last_diet__center = None  # [(Center_x,Center_y,radius)]

            # cal total Diet_number in this frame
            Diet_number = 0
            Diet_number = [Diet_number +
                           1 for i in data if i['class'] == 'Diet']

            for i in range(len(data)):
                item_dic = data[i]
                track_id = item_dic['track']
                class_name = item_dic['class']
                Center = item_dic['Center (x,y)'][1:-1].split(",")
                Center = [float(i) for i in Center]
                xymin = item_dic["BBox X (xmin, ymin)"][1:-1].split(",")
                xymin = [float(i) for i in xymin]

                if class_name == "Diet":
                    radius = Center[0]-xymin[0]
                    diet__center.append((Center[0], Center[1], radius))

                    if len(diet__center) == len(Diet_number):
                        last_diet__center = diet__center

                elif class_name == "Fly":
                    # Center = [float(i) for i in s]

                    if show_counts:
                        if last_diet__center == None:
                            pass
                        else:
                            fly_radius = Center[0]-xymin[0]

                            fly_counts, in_diet_fly_center = Cal_in_diet_counts(track_id, fly_counts, last_diet__center,
                                                                                Center, fly_radius, in_diet_fly_center)

                    # ======== update fly coordinate for print fly path line========
                    # put fly coordinate into fly_coordi
                    if show_path:
                        Center = (int(Center[0]), int(Center[1]))
                        if fly_coordi_short.get(track_id) == None:
                            fly_coordi_short.setdefault(
                                track_id, []).append(tuple(Center))

                        else:
                            if tuple(Center) == fly_coordi_short[track_id][-1]:
                                pass
                            else:
                                fly_coordi_short[track_id] += [tuple(Center)]
                    # print(fly_coordi_short)

                        label = f'{class_name} #{track_id}'
                        if fly_coordi_short.get(track_id):
                            plot_path_line(fly_coordi_short[track_id], frame, color=get_color_for(
                                label), line_thickness=thickness)

        except:
            pass
        if show_counts:
            plot_counts_text(frame, fly_numbers=fly_counts,
                             line_thickness=thickness)
            with open(source_folder / 'res' / 'fly_counts.txt', 'a') as f:
                f.write("frame: {}; fly_counts:{}\n".format(
                    frame_count, fly_counts))

        frame_count = frame_count + 1
        print(f"Video processing  {frame_count}/{video_length}")
        vid_writer.write(frame)

    cap.release()
    vid_writer.release()
