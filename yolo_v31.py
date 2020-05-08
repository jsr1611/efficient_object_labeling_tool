# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
# Added functionalities:
    # Manually remove bboxes, and draw new ones
    # Auto identify the duration of the video input file in number of frames => frame_count_total variable
    # Handle missing objects and new objects by removal and addition functions
    # Generate output video
    # Display Two screens, an editor and replay of the interpolated frames
"""
import _tkinter
import colorsys
import os
from os.path import isfile
from timeit import default_timer as timer
import datetime as dtime
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model
import cv2
import easygui as ez
from tkinter.filedialog import askopenfilename, Tk, Button, X
from SecondVid_v01 import camera1_tracker, dispay_info

# Global variables
counter_ = 1
interval = 10
frames_total = 0
final_coor_name = "./coordinates_.txt"
final_coor_name = final_coor_name[:-4] + dtime.datetime.today().strftime('%Y%m%d%H%M') + final_coor_name[-4:]
bboxNxt = []
bboxPrev = []  # previous frame data

frameNum = 0
intrpltd_box = [frameNum, 0, (0, 0, 0, 0)]  # temp variable for full data on one object
my_flag2 = False  # return img and bbox in the '2nd video labeling'
cam2_flag = 0
output_path3 = ""
id_max = 0
status = ""  # Window status indicator, ex: Active or Inactive


# A function for writing bbox coordinates into a file
def write2File(final_bbox):
    for f in range(len(final_bbox)):
        # print("Frame:", final_bbox[f])
        with open(final_coor_name,
                  "a") as text_file:  # "a" for appending; "w" for overwriting
            print(
                "{0:<6d} {1:<6d} {2:<6d} {3:<6d} {4:<6d} {5:<6d}".format(final_bbox[f][0], final_bbox[f][1],
                                                                         final_bbox[f][2][0],
                                                                         final_bbox[f][2][1], final_bbox[f][2][2],
                                                                         final_bbox[f][2][3]), file=text_file)

    print("The human generated bbox coordinates are saved at: {}".format(final_coor_name))


# A function to truncate erroneous zeros
def trunc_0s(data_in):
    data_out = []
    # print("data_in before:", data_in)
    try:
        if len(data_in[2]) == 4:
            if data_in[2][0] == 0 and data_in[2][1] == 0:
                # print("YES\n")
                if data_in[2][2] != 0:
                    data_out.append(data_in)
                data_in = data_out
        else:
            for a in range(len(data_in)):
                if data_in[a][2][0] == 0 and data_in[a][2][1] == 0:
                    print("YES\n")
                    for i in range(len(data_in)):
                        if data_in[i][2][0] != 0 and data_in[i][2][1] != 0 and data_in[i][2][2] != 0:
                            data_out.append(data_in[i])
                    data_in = data_out
    except IndexError:
        print("Index Error inside eucl_sort_add() function")
        # print("data_in after:", data_in)
    return data_in


# A function for identifying and matching objects between 2 different frames. LENGTH of both objects should be the SAME
def eucl_sort(data_0, data_x):
    import math
    # coor1, bbox = eucl_sort(coor1, bbox)
    # provide coordinates (any_nym, top, left, bottom, right) of two frames
    # and get the corrected array of coordinates for the frame(array)
    # ------------Frame x-1 -----------------------------------------
    print("previous_data_0 ", data_0, '\n')
    print("current_data_x ", data_x, '\n')

    data_0 = trunc_0s(data_0)
    data_x = trunc_0s(data_x)

    data_return = []
    data_xx = data_x
    data_00 = data_0

    data_0_new = []
    data_x_new = []
    if (len(data_xx) > 0 and len(data_00)):
        for el in data_0:
            data_0_new.append(el[2])
        for el2 in data_x:
            data_x_new.append(el2[2])
        data0r_np = data_0  # to be reused
        dataxr_np = data_x  # to be reused
        return_np = data_x  # to be returned
        data_0 = data_0_new
        data_x = data_x_new

        data_0 = replaceAll(" ".join(str(x) for x in data_0))
        data_x = replaceAll(" ".join(str(x) for x in data_x).replace("[", ""))
        # ------------Frame 0-----------------------------------------
        data_0 = [int(i) for i in data_0.split()]
        frame_obj_0 = int(len(data_0) / 4)
        data0_np = np.array(data_0).reshape(frame_obj_0, 4)
        # ------------Frame x-----------------------------------------
        data_x = [int(i) for i in data_x.split()]
        frame_obj_x = int(len(data_x) / 4)
        datax_np = np.array(data_x).reshape(frame_obj_x, 4)
        centroid_xy_0 = []  #
        centroid_x_0 = []  #
        centroid_y_0 = []  #
        # -----------------
        for k in range(data0_np.shape[0]):
            Left_0, Top_0, Right_0, Bottom_0 = data0_np[k, :]
            centroid_xy_0.append((Left_0 + Right_0) / 2)
            centroid_xy_0.append((Top_0 + Bottom_0) / 2)
        centroid_xy_0 = np.array(centroid_xy_0).reshape(data0_np.shape[0], 2)
        # -----------------------Frame x   centroid-----------------------------
        centroid_xy_x = []  #
        centroid_x_x = []  #
        centroid_y_x = []  #
        # -----------------
        for k in range(datax_np.shape[0]):
            Left_x, Top_x, Right_x, Bottom_x = datax_np[k, :]
            centroid_xy_x.append((Left_x + Right_x) / 2)
            centroid_xy_x.append((Top_x + Bottom_x) / 2)

        centroid_xy_x = np.array(centroid_xy_x).reshape(datax_np.shape[0], 2)
        # ---------------------------------Euclidean distance calculation------------------------------------------
        index_mat = []
        duplicate_index = []
        tmp_arr = []
        for k in range(data0_np.shape[0]):
            index = 0
            for l in range(datax_np.shape[0]):
                if l == 0:
                    cx, cy = centroid_xy_0[k, :] - centroid_xy_x[l, :]
                    Euc_Dis = math.sqrt(cx ** 2 + cy ** 2)
                else:
                    cx, cy = centroid_xy_0[k, :] - centroid_xy_x[l, :]
                    Euc_Dis_new = math.sqrt(cx ** 2 + cy ** 2)
                    if Euc_Dis_new > Euc_Dis:
                        Euc_Dis = Euc_Dis
                        index = index
                    else:
                        Euc_Dis = Euc_Dis_new
                        index = l
            if k == 0:
                index_mat.append(index)
                tmp_arr.append(data_00[index][1])
            for i in range(k):
                if index not in index_mat:
                    index_mat.append(index)
                    tmp_arr.append(data_00[index][1])
        old_indices = []
        try:
            for o in range(len(data_00)):
                old_indices.append(data_00[o][1])

            for i in range(len(old_indices)):
                if old_indices[i] not in tmp_arr:
                    index_mat.append(i)
                    tmp_arr.append(old_indices[i])

            index_mat = np.array(index_mat, np.int16)
            for i in range(len(data_xx)):
                data_return.append(data_xx[index_mat[i]])
                data_return[i][1] = data_00[i][1]
        except IndexError:
            print("Index Error inside eucl_sort() function")
        # print("index_mat:", index_mat)
        print("Output data_0 data_x:\n", data_00, data_return)
    else:
        pass
    return data_00, data_return


# replaceAll removes all unnecessary literals in a given string
def replaceAll(txt):
    print("Input txt data:", txt)
    txt = txt.replace(",", " ").replace("]", " ").replace("[", " ").replace(")", " ").replace("(", " ")
    print("Output txt data:", txt)
    return txt


# euc_remove function removes object bbox coordinates inside a given area specified by another (bigger) bbox coordinates
def euc_remove(input_0, target):
    print("input data ", input_0, '\n')
    print("target for removal ", target, '\n')
    global rm_labels
    output_data = []
    cxy = []
    for i in range(len(input_0)):
        cxy.append((int((int(input_0[i][2][0]) + int(input_0[i][2][2])) / 2),
                    int((int(input_0[i][2][1]) + int(input_0[i][2][3])) / 2)))
    index2 = 0
    index = 0
    myflag_0 = False
    rm_labels = []
    for i in range(len(input_0)):
        for j in range(len(target)):
            if target[j][0] < cxy[i][0] < target[j][2] and target[j][1] < cxy[i][1] < target[j][3]:
                myflag_0 = True
                index2 = i
            else:
                try:
                    if not myflag_0:
                        index = i
                except IndexError:
                    print("Index Error")
        if not myflag_0:
            output_data.append(input_0[index])
        else:
            rm_labels.append(index2)
            print("deleted label: ", input_0[index2])
        myflag_0 = False
    print("rm_label(len)=", len(rm_labels))
    print("Output data: ", output_data)
    return output_data


# eucl_sort_remove removes missing object(s) in the past frame (data_0) and matches the objects between frames
def eucl_sort_remove(data_0, data_x):
    # coor1, bbox = eucl_sort_remove(coor1, bbox)
    import math
    # provide coordinates (any_nym, top, left, bottom, right) of two frames
    # and get the corrected array of coordinates for the frame(array)
    # ------------Frame x-1 -----------------------------------------
    print("previous_data_0 ", data_0, '\n')
    print("current_data_x ", data_x, '\n')
    data_x = trunc_0s(data_x)
    data_0 = trunc_0s(data_0)

    data_return = []
    data_xx = data_x
    data_00 = data_0

    data_0_new = []
    data_x_new = []
    if (len(data_00) > 0 and len(data_xx) > 0):
        for el in data_0:
            data_0_new.append(el[2])
        for el2 in data_x:
            data_x_new.append(el2[2])
        data0r_np = data_0  # to be reused
        dataxr_np = data_x  # to be reused
        return_np = data_x  # to be returned
        data_0 = data_0_new
        data_x = data_x_new

        data_0 = replaceAll(" ".join(str(x) for x in data_0))
        data_x = replaceAll(" ".join(str(x) for x in data_x).replace("[", ""))
        # ------------Frame 0-----------------------------------------
        data_0 = [int(i) for i in data_0.split()]
        frame_obj_0 = int(len(data_0) / 4)
        data0_np = np.array(data_0).reshape(frame_obj_0, 4)
        # ------------Frame x-----------------------------------------
        data_x = [int(i) for i in data_x.split()]
        frame_obj_x = int(len(data_x) / 4)
        datax_np = np.array(data_x).reshape(frame_obj_x, 4)

        centroid_xy_0 = []  #
        centroid_x_0 = []  #
        centroid_y_0 = []  #
        # -----------------
        for k in range(data0_np.shape[0]):
            Left_0, Top_0, Right_0, Bottom_0 = data0_np[k, :]
            centroid_xy_0.append((Left_0 + Right_0) / 2)
            centroid_xy_0.append((Top_0 + Bottom_0) / 2)
        centroid_xy_0 = np.array(centroid_xy_0).reshape(data0_np.shape[0], 2)

        # -----------------------Frame x   centroid-----------------------------
        centroid_xy_x = []  #
        centroid_x_x = []  #
        centroid_y_x = []  #
        # -----------------
        for k in range(datax_np.shape[0]):
            Left_x, Top_x, Right_x, Bottom_x = datax_np[k, :]
            centroid_xy_x.append((Left_x + Right_x) / 2)
            centroid_xy_x.append((Top_x + Bottom_x) / 2)

        centroid_xy_x = np.array(centroid_xy_x).reshape(datax_np.shape[0], 2)

        # ---------------------------------Euclidean distance calculation------------------------------------------
        index_mat = []
        l = 0

        for k in range(datax_np.shape[0]):
            index = 0
            for l in range(data0_np.shape[0]):
                if (l == 0):
                    cx, cy = centroid_xy_x[k, :] - centroid_xy_0[l, :]
                    Euc_Dis = math.sqrt(cx ** 2 + cy ** 2)
                    # print("Euc_Dis", Euc_Dis)
                else:
                    cx, cy = centroid_xy_x[k, :] - centroid_xy_0[l, :]
                    Euc_Dis_new = math.sqrt(cx ** 2 + cy ** 2)
                    # print("Euc_Dis_new", Euc_Dis_new)
                    if Euc_Dis_new > Euc_Dis:
                        Euc_Dis = Euc_Dis
                        index = index
                    else:
                        Euc_Dis = Euc_Dis_new
                        index = l

            index_mat = np.append(index_mat, index)
        index_mat = np.array(index_mat, np.int16)
        # print(index_mat)
        for i in range(len(data_xx)):
            data_return.append(data_00[index_mat[i]])
            data_return[i][1] = data_00[index_mat[i]][1]
        print("Output data_0 data_x:\n", data_return, "\n", data_xx)
    else:
        pass
    return data_return, data_xx


# eucl_sort_add adds new objects at the end of the array (data_x)and needed for matching the objects between frames
def eucl_sort_add(data_0, data_x):
    import math
    # provide coordinates (frame num, id, top, left, bottom, right) of two frames
    # and get the corrected array of coordinates for the frame(array)
    # ------------Frame x-1 -----------------------------------------
    print("previous_data_0 ", data_0, '\n')
    print("current_data_x ", data_x, '\n')

    data_x = trunc_0s(data_x)
    data_0 = trunc_0s(data_0)

    data_return = []
    data_xx = data_x
    data_00 = data_0

    data_0_new = []
    data_x_new = []
    if (len(data_00) > 0 and len(data_xx) > 0):
        for el in data_0:
            data_0_new.append(el[2])
        for el2 in data_x:
            data_x_new.append(el2[2])
        data0r_np = data_0  # to be reused
        dataxr_np = data_x  # to be reused
        return_np = data_x  # to be returned
        data_0 = data_0_new
        data_x = data_x_new

        data_0 = replaceAll(" ".join(str(x) for x in data_0))
        data_x = replaceAll(" ".join(str(x) for x in data_x).replace("[", ""))
        # ------------Frame 0-----------------------------------------
        data_0 = [int(i) for i in data_0.split()]
        frame_obj_0 = int(len(data_0) / 4)
        data0_np = np.array(data_0).reshape(frame_obj_0, 4)
        # ------------Frame x-----------------------------------------
        data_x = [int(i) for i in data_x.split()]
        frame_obj_x = int(len(data_x) / 4)
        datax_np = np.array(data_x).reshape(frame_obj_x, 4)

        centroid_xy_0 = []  #
        centroid_x_0 = []  #
        centroid_y_0 = []  #
        # -----------------
        for k in range(data0_np.shape[0]):
            Left_0, Top_0, Right_0, Bottom_0 = data0_np[k, :]
            centroid_xy_0.append((Left_0 + Right_0) / 2)
            centroid_xy_0.append((Top_0 + Bottom_0) / 2)
        centroid_xy_0 = np.array(centroid_xy_0).reshape(data0_np.shape[0], 2)

        # -----------------------Frame x   centroid-----------------------------
        centroid_xy_x = []  #
        centroid_x_x = []  #
        centroid_y_x = []  #
        # -----------------
        for k in range(datax_np.shape[0]):
            Left_x, Top_x, Right_x, Bottom_x = datax_np[k, :]
            centroid_xy_x.append((Left_x + Right_x) / 2)
            centroid_xy_x.append((Top_x + Bottom_x) / 2)

        centroid_xy_x = np.array(centroid_xy_x).reshape(datax_np.shape[0], 2)

        # ---------------------------------Euclidean distance calculation------------------------------------------
        index_mat = []
        l = 0
        test_mat = []

        for k in range(data0_np.shape[0]):
            index = 0
            for l in range(datax_np.shape[0]):
                if (l == 0):
                    cx, cy = centroid_xy_0[k, :] - centroid_xy_x[l, :]
                    Euc_Dis = math.sqrt(cx ** 2 + cy ** 2)
                    # print("Euc_Dis", Euc_Dis)
                else:
                    cx, cy = centroid_xy_0[k, :] - centroid_xy_x[l, :]
                    Euc_Dis_new = math.sqrt(cx ** 2 + cy ** 2)
                    # print("Euc_Dis_new", Euc_Dis_new)
                    if Euc_Dis_new > Euc_Dis:
                        Euc_Dis = Euc_Dis
                        index = index
                    else:
                        Euc_Dis = Euc_Dis_new
                        index = l

            index_mat = np.append(index_mat, index)
        index_mat = np.array(index_mat, np.int16)
        # print("existing:", end=" ")
        # for x in range(0, len(data_xx)):
        #    print(x, end=" ")
        # print("\nMatching index:\t", index_mat)

        data_new = []
        index_mat_new = []

        for i in range(len(data_00)):
            data_return.append(data_xx[index_mat[i]])
            data_return[i][1] = data_00[i][1]

        arr = [0] * len(data_xx)
        for i in range(len(data_xx)):
            try:
                for j in range(len(index_mat)):
                    if i == index_mat[j]:
                        arr[i] = 1
            except IndexError:
                print("IndexError. Unmatching index:", i)
        for k in range(len(arr)):
            if arr[k] == 0:
                data_new.append(data_xx[k])
                index_mat_new.append(k)
        try:
            idn = data_return[-1][1]  # assign last element's id to idn,
        except IndexError:
            print("Index Error")
        # #and check if it is max value, if not assign max value to idn using the below for loop
        for i in range(len(data_return)):
            if data_return[i][1] > idn:
                idn = data_return[i][1]
        # assign new id to the new objects, get id from idn (max val of previous frame id)
        for m in range(len(index_mat_new)):
            idn += 1
            data_return.append(data_xx[index_mat_new[m]])
            data_return[len(index_mat) + m][1] = idn

        # print("index_mat_new:", index_mat_new)
        print("Return data_0, data_x: \n", data_00, "\n", data_return)
        # print("data_new", data_new)
    else:
        pass
    return data_00, data_return


def eucl_find(data_0, data_x):
    import math
    # ------------Frame x-1 -----------------------------------------
    print("data_previous 0 ", data_0, '\n')
    print("data_current x ", data_x, '\n')
    data_0_new = []
    #data_x_new = []
    data_return = []
    data_00 = data_0  # to be reused
    for el in data_0:
        data_0_new.append(el[2])
    data_x = data_x[2]
    data_0 = data_0_new
    #data_x = data_x_new

    data_0 = replaceAll(" ".join(str(x) for x in data_0))
    data_x = replaceAll(" ".join(str(x) for x in data_x).replace("[", ""))

    # ------------Frame 0-----------------------------------------
    data_0 = [int(i) for i in data_0.split()]
    frame_obj_0 = int(len(data_0) / 4)
    data0_np = np.array(data_0).reshape(frame_obj_0, 4)

    # ------------Frame x-----------------------------------------
    data_x = [int(i) for i in data_x.split()]
    frame_obj_x = int(len(data_x) / 4)
    datax_np = np.array(data_x).reshape(frame_obj_x, 4)
    centroid_xy_0 = []  #
    # -----------------
    for k in range(data0_np.shape[0]):
        Left_0, Top_0, Right_0, Bottom_0 = data0_np[k, :]
        centroid_xy_0.append((Left_0 + Right_0) / 2)
        centroid_xy_0.append((Top_0 + Bottom_0) / 2)
    centroid_xy_0 = np.array(centroid_xy_0).reshape(data0_np.shape[0], 2)
    # -----------------------Frame x   centroid-----------------------------
    centroid_xy_x = []  #
    # -----------------
    for k in range(datax_np.shape[0]):
        Left_x, Top_x, Right_x, Bottom_x = datax_np[k, :]
        centroid_xy_x.append((Left_x + Right_x) / 2)
        centroid_xy_x.append((Top_x + Bottom_x) / 2)
    centroid_xy_x = np.array(centroid_xy_x).reshape(datax_np.shape[0], 2)
    # ---------------------------------Euclidean distance calculation------------------------------------------
    index_mat = []
    try:
        for k in range(datax_np.shape[0]):
            index = 0
            for l in range(data0_np.shape[0]):
                if (l == 0):
                    cx, cy = centroid_xy_x[k, :] - centroid_xy_0[l, :]
                    Euc_Dis = math.sqrt(cx ** 2 + cy ** 2)
                else:
                    cx, cy = centroid_xy_x[k, :] - centroid_xy_0[l, :]
                    Euc_Dis_new = math.sqrt(cx ** 2 + cy ** 2)
                    if Euc_Dis_new > Euc_Dis:
                        Euc_Dis = Euc_Dis
                        index = index
                    else:
                        Euc_Dis = Euc_Dis_new
                        index = l
            index_mat = np.append(index_mat, index)
        index_mat = np.array(index_mat, np.int16)
        data_return = data_00[index_mat[0]]

        del data_00[index_mat[0]]
    except Exception as e:
        print(e)
    print("data_return", data_return)
    return data_00, data_return

def fileOpenClicked(default_path):
    root = Tk()
    video_path = str(
        askopenfilename(filetypes=[("MP4 files", "*.mp4"), ("AVI files", "*.avi"), ("All files", "*.*")]))
    video_path = "./" + os.path.split(video_path)[1]
    root.destroy()
    if len(video_path) >= 5:
        print("Input file path selected by the user: ", video_path)
    else:
        vid_tmp = cv2.VideoCapture(default_path)
        if vid_tmp.isOpened():
            video_path = default_path
        else:
            raise IOError("Error. User canceled input file selection.")
        print("User selected default input file path:", video_path)
    return video_path


def myRect(img, mybbox, i):
    try:
        if i == 'one':
            cv2.rectangle(img, (int(mybbox[2][0]), int(mybbox[2][1])),
                          (int(mybbox[2][2]), int(mybbox[2][3])), (0, 255, 0), 2)
            cv2.rectangle(img, (int(mybbox[2][0]), int(mybbox[2][1]) - 20),
                          ((int(mybbox[2][0]) + 30), int(mybbox[2][1])), (0, 255, 0), -1)
            cv2.putText(img, str(mybbox[1]), (int(mybbox[2][0]) + 1, (int(mybbox[2][1]) - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        else:
            cv2.rectangle(img, (int(mybbox[i][2][0]), int(mybbox[i][2][1])),
                          (int(mybbox[i][2][2]), int(mybbox[i][2][3])), (0, 255, 0), 2)
            cv2.rectangle(img, (int(mybbox[i][2][0]), int(mybbox[i][2][1]) - 20),
                          ((int(mybbox[i][2][0]) + 30), int(mybbox[i][2][1])), (0, 255, 0), -1)
            cv2.putText(img, str(mybbox[i][1]), (int(mybbox[i][2][0]) + 1, (int(mybbox[i][2][1]) - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    except IndexError:
        print("Index Error")
    except ValueError:
        print("Value Error")
    return img


class YOLO(object):
    _defaults = {
        "model_path": 'model_data/yolo.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        "score": 0.6,
        "iou": 0.45,
        "model_image_size": (416, 416),
        "gpu_num": 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)  # set up default values
        self.__dict__.update(kwargs)  # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors == 6  # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors // 2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
            self.yolo_model.load_weights(self.model_path)  # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                   num_anchors / len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2,))
        if self.gpu_num >= 2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           len(self.class_names), self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        global bbox, my_label, frameNum
        start = timer()
        img = np.asarray(image)
        bbox = []
        id_num = 0
        my_label = "veh"
        centroid_xy = []

        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            ybox = [left, top, right, bottom]
            # print('yolo:', label+str(id_num), (left, top), (right, bottom))
            bbox.append([frame_counter, id_num, (left, top, right, bottom)])
            centroid_xy.append(int((left + right) / 2))
            centroid_xy.append(int((top + bottom) / 2))
            print(frame_counter, id_num, (left, top, right, bottom))
            # print("counter_=1:", my_label + str(id_num))
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])
            try:
                cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(img, (left, top - 20), ((left + 30), top), (0, 255, 0), -1)
                cv2.circle(img, (centroid_xy[0], centroid_xy[1]), 3, (0, 255, 0), 2)
                cv2.putText(img, str(id_num), (left + 1, top - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                            cv2.LINE_AA)
            except IndexError:
                print("Index Error at detect_image()")
            id_num += 1
            centroid_xy = []
            left = 0
            right = 0
            top = 0
            bottom = 0

        end = timer()
        # print(end - start)
        if my_flag2:
            return img, bbox
        return img

    def close_session(self):
        self.sess.close()

    ############ Drawing function ##################################
    def drawRect(self, img1, img2):
        # img1 - input image with bboxes for displaying and drawing on
        # img2 - orignal image to be output with bboxes only
        global bbox, bboxPrev, label, id_n, my_label, rm_labels, numOfObj, tmp_bbox, myflag, cam2_flag, id_max, eucfind_data0
        id_n = 1
        label = len(coor1) + id_n
        tmp_bbox = []
        myflag = False
        newObjCounter = 1
        eucfind_data0 = coor1
        eucfind_data1 = coor1 # checked
        while True:
            r = cv2.selectROI("Label Editor", img1, fromCenter=False, showCrosshair=True)
            x1 = int(r[0])
            y1 = int(r[1])
            x2 = int(r[0] + r[2])
            y2 = int(r[1] + r[3])
            if Cam2_flag == 1:  # working for second camera
                while True:
                    if counter_ == 1:
                        label = ez.enterbox("Please enter the id of the object")
                    elif counter_ > 1:
                        data_x = [frameNum, 0, (x1, y1, x2, y2)]
                        if data_x[2][0] == 0 and data_x[2][1] == 0 and data_x[2][2] == 0:
                            print("jumped?")
                            break
                        else:
                            eucfind_data0, bbox_ = eucl_find(eucfind_data0, data_x)
                            label = bbox_[1]
                            msg = "Selected ID is  {}. Is this object ID correct?".format(label)
                            title = "Object ID selection process"
                            choices = ["Yes", "No", "Modify"]
                            key = ez.buttonbox(msg, title, choices)
                            if key == "Yes":
                                eucfind_data1 = eucfind_data0
                                pass
                            if key == "Modify":
                                eucfind_data0 = eucfind_data1
                                label = ez.enterbox("Please enter the ID of the object")
                            else:
                                pass
                    else:
                        label = 0
                    try:
                        if type(int(label)) is int:
                            print("label given: {}, my_flag3={}, counter_={}".format(label, Cam2_flag, counter_))
                            break
                        else:
                            print("Please enter the object id correctly. The id should be an integer number!", label)
                    except TypeError:
                        print("Type Error at bbox appending frameNum={}".format(counter_))
                    except ValueError:
                        print("Value Error")

            else:  # working for first camera
                if counter_ > 1:  # when working for consecutive frames
                    if len(rm_labels) == 1 and newObjCounter == 1:
                        label = rm_labels[0]
                        print("label:{},myflg3={},counter={},rm_lbls#={}".format(label, Cam2_flag, counter_,
                                                                                 len(rm_labels)))
                    elif len(rm_labels) > 1 or newObjCounter > 1:
                        for i in range(len(coor1)):
                            if id_max < coor1[i][1]:
                                id_max = coor1[i][1]
                        label = id_max + id_n
                        id_n += 1
                        print("label:{},myflg3={},countr={},rm_lbl# = {}".format(label, Cam2_flag, counter_,
                                                                                 len(rm_labels)))
                else:  # when working for first frame
                    label = id_n - 1
                    id_n += 1
            try:
                bbox_ = [frameNum, int(label), (x1, y1, x2, y2)]
                bbox_2 = trunc_0s(bbox_)
                print("bbox_2, bbox_", bbox_2, bbox_)
                if bbox_ != bbox_2:
                    pass  # print("jumped?")
                else:
                    bbox.append([frameNum, int(label), (x1, y1, x2, y2)])
                    myRect(img1, bbox_, 'one')  # Generating bboxes on an image
                    print("Drawn bbox:", [frameNum, label, (x1, y1, x2, y2)])
            except ValueError:
                print("Value Error")
            newObjCounter += 1
            if cv2.waitKey(0) & 0xff == 27:
                if Cam2_flag == 1:  # when working for second camera
                    if counter_ > 1 and len(bbox) > len(coor1):
                        myflag = True
                        print("\n\n\n\nmyflag=True: bbox, coor1 before eucl_sort_add() \n", bbox, "\n", coor1)
                        coor1, bbox = eucl_sort_add(coor1, bbox)
                        tmp_bbox = bbox
                    elif counter_ > 1 and len(bbox) == len(coor1) and (len(bbox) > 0 and len(coor1) > 0):
                        print("\n\n\n\nmyflag=True: bbox, coor1 before eucl_sort() \n", bbox, "\n", coor1)
                        coor1, bbox = eucl_sort(coor1, bbox)
                    elif counter_ > 1 and len(bbox_) < len(coor1):
                        print("\n\n\n\nmyflag=True: bbox, coor1 before eucl_sort_rem() \n", bbox, "\n", coor1)
                        coor1, bbox = eucl_sort_remove(coor1, bbox)
                else:  # when working for first camera
                    if counter_ > 1:  # when working for consecutive frames
                        if len(bbox) < len(coor1) and len(coor1) > 0:
                            myflag = False
                            coor1, bbox = eucl_sort_remove(coor1, bbox)
                            coor1, bbox = eucl_sort(coor1, bbox)
                            # print("bbox, coor1 after eucl_sort() \n", bbox,"\n", coor1)
                        elif len(bbox) == len(coor1) and (len(bbox) > 0 and len(coor1) > 0):
                            coor1, bbox = eucl_sort(coor1, bbox)
                            # print("bbox, coor1 after eucl_sort() \n", bbox, "\n", coor1)
                        else:
                            myflag = True
                            coor1, bbox = eucl_sort_add(coor1, bbox)
                            tmp_bbox = bbox
                    else:  # when working for first frame
                        if Cam2_flag == 0:  # if it is first camera
                            tmp_id = 0
                            print("Frames before renaming: bbox \n", bbox)
                            if len(bbox) > 0:
                                for l in range(len(bbox)):
                                    bbox[l][1] = tmp_id
                                    tmp_id += 1
                                print("Frames after renaming: bbox \n", bbox)
                            else:
                                pass
                    print("coor1, myflag after eucl before imshow(): \n", coor1, "\n", myflag)
                if len(bbox) > 0:
                    for i in range(len(bbox)):
                        myRect(img2, bbox, i)  # Generating bboxes on an image
                    cv2.imshow("Label Editor", img2)
                    if myflag:
                        try:
                            bbox = bbox[:len(coor1)]
                        except IndexError:
                            print("Index Error")
                break
        rm_labels = []
        return img2

    ############ Removing function  ################################
    def roiCUT(self, img1, img2):
        # img1 - input image containing bboxes for removal
        # img2 - image with no bboxes and to copy ROI from
        global buffer, bbox
        target = []
        img = img1
        buffer = []
        # rois = []
        while True:
            r = cv2.selectROI("Label Editor", img, fromCenter=False, showCrosshair=True)
            x1 = int(r[0])  # top
            y1 = int(r[1])  # left
            x2 = int(r[0] + r[2])  # bottom
            y2 = int(r[1] + r[3])  # right
            imgCrop = img2[y1:y2, x1:x2]
            # rois.append([y1, y2, x1, x2])
            target.append([x1, y1, x2, y2])
            print("Area selected for cutting(erasing) and removing the bboxes inside:\n", [x1, y1, x2, y2])
            buffer.append(imgCrop)
            img4 = img2.copy()
            img[y1:y2, x1:x2] = img2[y1:y2, x1:x2]
            bbox = euc_remove(bbox, target)
            centroid_xy = []
            for i in range(len(bbox)):
                img = myRect(img4, bbox, i)
                try:
                    centroid_xy.append(int((bbox[i][2][0] + bbox[i][2][2]) / 2))
                    centroid_xy.append(int((bbox[i][2][1] + bbox[i][2][3]) / 2))
                    cv2.circle(img, (centroid_xy[0], centroid_xy[1]), 3, (0, 255, 0), 2)
                except IndexError:
                    print("Index Error")
                centroid_xy = []
            dispay_info(img, 1, "Active", counter_, frames_total)
            if cv2.waitKey(0) & 0xff == 27:
                # bbox = euc_remove(bbox, target)
                break
        return img4


def detect_video(yolo, video_path, output_path="./OutputVid_.mp4"):
    global bbox, bboxPrev, counter_, interval, frames_total, bboxNxt, out, intrpltd_box, coordif, final_xy, frameNum, myflag, tmp_bbox, intervalCoord, out2, randomCounter_, cam2_flag, output_path3

    final_xy = []  # final array for bbox coordinates = ground truth data
    intervalVid = []  # array for interval frames (skipped frames interpolation)

    try:
        default_path = './2020_0109_140236_NOR_FILE.AVI'
        if Cam2_flag == 0:
            print("Default input file path: ", default_path)
            while True:
                fastTrack2ndCam = int(ez.enterbox("Enter 1 if you have camera 1 output, otherwise enter 0."))
                if fastTrack2ndCam == 1:
                    output_path3 = fileOpenClicked(default_path)
                    video_path = output_path3
                    Cam2_flag = 1
                    break
                elif fastTrack2ndCam == 0:
                    Cam2_flag = 0
                    video_path = fileOpenClicked(default_path)
                    break
                else:
                    print("You entered something wrong. Please, enter 1 or 0 to proceed.")
        if Cam2_flag == 1:
            if video_path[-8:-4] == "left":
                video_path2 = video_path[:-8] + "right" + video_path[-4:]
            else:
                video_path2 = video_path[:-9] + "left" + video_path[-4:]
            vid_new = cv2.VideoCapture(video_path2)
            if vid_new.isOpened():
                default_path = video_path2
                video_path = video_path2
                print("2nd camera input file: Input file path automatically selected: ", video_path)
            else:
                print(
                    "2nd camera input file: Input file path automatic selection failed. Openration is handed to user.\n")
                video_path = fileOpenClicked(default_path)
        else:
            raise IOError("Couldn't open webcam or video file")
    except IOError:
        print("Video format error.")
    except _tkinter.TclError:
        print("Video format error.")

    title_ = output_path[:-4] + dtime.datetime.today().strftime('%Y%m%d%H%M')
    type_ = output_path[-4:]
    output_path2 = title_ + type_  # for output video and displaying interpolation results
    vid = cv2.VideoCapture(video_path)
    vid2 = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    frame_count_total = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))  # total number of frames in a video file
    print("Video file duration: {0:<d} frames, {1:<.2f} seconds\n".format(frame_count_total, frame_count_total / 30.0))

    video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        # set the [video] output file write function
        # out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), 30.0, video_size)
        # out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, video_size)
        try:
            if video_path[-3:] == 'mp4' or video_path[-3:] == 'MP4':
                out2 = cv2.VideoWriter(output_path2, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, video_size)

            elif video_path[-3:] == 'avi' or video_path[-3:] == 'AVI':
                out2 = cv2.VideoWriter(output_path2, cv2.VideoWriter_fourcc(*'XVID'), 30.0, video_size)
            else:
                print("Please check file format", type(video_path), video_path[-4:], "?")
        except IOError:
            print("File format incorrect")

    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()

    while True:
        return_value, frame = vid.read()
        if not return_value:
            print("Output video file is generated at:", output_path2)
            print("---------------------------------------------------------------------------------------\n")
            if Cam2_flag == 1:
                print("2nd camera video labeling is finished")
                print("Program finished the job.\n")
            elif Cam2_flag == 0:
                print("2nd camera video labeling is started")
            else:
                print("Something else")
            break
        image = Image.fromarray(frame)
        imgcopy = np.asarray(image).copy()
        imgcopy3 = imgcopy.copy()
        image = yolo.detect_image(image)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
            coordif = []
            intervalCoord = []
        # Flag for image description-helpers:
        # disp_descriptions(image, flag_num, status, frame_counter, total_frame_count)
        # flag_num = 1-5 or else ("Flag number is empty"):
        # 1 => "Yolo's result is displayed. Editing (removing) bbox is enabled"
        # 2 => "Editing bbox is finished. Manual labeling (drawing) is enabled"
        # 3 => "Labeling for current frame is finished (final result)"
        # 4 => "Replay of Camera 1"
        # 5 => "Replay of Camera 2"
        # status => Active or Inactive
        ############ First interval ##################################

        if counter_ == 1:
            frame_counter = 0
            print("\n\n First_Frame\n\n")
            result_tmp = result.copy()
            result2 = result.copy()
            cv2.namedWindow("Label Editor", cv2.WINDOW_NORMAL)
            if Cam2_flag == 0:
                dispay_info(result2, 1, "Active", counter_,
                            frame_count_total)  # Adding text to result2. "Yolo's result. Editing is enabled..."
            elif Cam2_flag == 1:
                dispay_info(result_tmp, 1, "Inactive", counter_,
                            frame_count_total)  # Adding text to result2. "Yolo's result. Editing is enabled..."
                cv2.imshow("Label Editor", result_tmp)
                cv2.namedWindow("AutoTracking Cam1", cv2.WINDOW_NORMAL)
                camera1_tracker(output_path3, counter_, interval)
                vidtmp = cv2.VideoCapture(output_path3)
                vidtmp.set(cv2.CAP_PROP_POS_FRAMES, 0)
                _, frametmp = vidtmp.read()
                dispay_info(frametmp, 4, "Inactive", counter_,
                            frame_count_total)
                cv2.imshow("AutoTracking Cam1", frametmp)
                dispay_info(result2, 1, "Active", counter_,
                            frame_count_total)  # Adding text to result2. "Yolo's result. Editing is enabled..."
            imgcopy = yolo.roiCUT(result2, imgcopy)
            imgcopy2 = imgcopy.copy()
            dispay_info(imgcopy2, 2, "Active", counter_,
                        frame_count_total)  # Adding text to imgcopy to show updates. "Drawing is enabled..."
            cv2.imshow("Label Editor", imgcopy2)
            result = yolo.drawRect(imgcopy2, imgcopy3)
            result_tmp = result.copy()
            dispay_info(result_tmp, 3, "Active", counter_,
                        frame_count_total)  # Adding text to imgcopy to show updates. "Final result..."
            cv2.imshow("Label Editor", result_tmp)
            coor1 = bbox
            bbox = []
            cv2.waitKey(0) & 0xff

        ############ N-th interval  ##################################

        elif (counter_ - 1) % interval == 0:
            print("\n\n\n Interval_{} \n\n\n".format(int((counter_ - 1) / interval)))
            result2 = result.copy()
            result_tmp = result.copy()
            if Cam2_flag == 0:
                dispay_info(result2, 1, "Active", counter_,
                            frame_count_total)  # Adding text to result2. "Yolo's result. Editing is enabled..."
            if Cam2_flag == 1:
                dispay_info(result_tmp, 1, "Inactive", counter_,
                            frame_count_total)  # Adding text to result2. "Yolo's result. Editing is enabled..."
                cv2.imshow("Label Editor", result_tmp)
                print("camera1_tracker(output_path3, frame_count_total, counter_, interval)", output_path3,
                      frame_count_total, counter_, interval)
                camera1_tracker(output_path3, counter_, interval)
                dispay_info(result2, 1, "Active", counter_,
                            frame_count_total)  # Adding text to result2. "Yolo's result. Editing is enabled..."
            imgcopy = yolo.roiCUT(result2, imgcopy)
            imgcopy2 = imgcopy.copy()
            dispay_info(imgcopy2, 2, "Active", counter_,
                        frame_count_total)  # Adding text to imgcopy to show updates. "Drawing is enabled..."
            # Displaying edited Yolo's result and labeling (drawing) enabled ---------------------------------------
            cv2.imshow("Label Editor", imgcopy2)
            result = yolo.drawRect(imgcopy2, imgcopy3)
            result_tmp = result.copy()
            result_tmp2 = result.copy()
            dispay_info(result_tmp, 3, "Active", counter_,
                        frame_count_total)  # Adding text to imgcopy to show updates. "Final result..."
            cv2.imshow("Label Editor", result_tmp)
            print("After drawRect(), bbox, coor1\n", bbox, "\n", coor1)
            if counter_ == interval + 1:
                for i in range(interval):
                    vid2.set(cv2.CAP_PROP_POS_FRAMES, i)
                    _, frame2 = vid2.read()
                    intervalVid.append(frame2)
            elif counter_ > interval + 1:
                for i in range(interval):
                    vid2.set(cv2.CAP_PROP_POS_FRAMES, counter_ - interval - 1 + i)
                    _, frame2 = vid2.read()
                    intervalVid.append(frame2)
            if not myflag:
                try:
                    for i in range(len(bbox)):
                        dif = (np.subtract(bbox[i][2], coor1[i][2])) / interval
                        coordif.append(dif)
                        dif = []
                except IndexError:
                    print("Index Error at generating dif distance, frameNum={}, 1st condition".format(counter_))
            else:
                try:
                    for i in range(len(bbox)):
                        dif = (np.subtract(bbox[i][2], coor1[i][2])) / interval
                        coordif.append(dif)
                        dif = []
                except IndexError:
                    print("Index Error at generating dif distance, frameNum={}, 2nd condition".format(counter_))
            for i in range(interval):
                if len(bbox) > 0:
                    for j in range(len(bbox)):
                        try:
                            intrpltd_box[2] = np.add(coor1[j][2], coordif[j] * i).astype('int16')
                            intrpltd_box[1] = coor1[j][1]
                            intrpltd_box[0] = frame_counter
                            coorx.append(intrpltd_box)
                            final_xy.append(intrpltd_box)
                            intervalCoord.append(intrpltd_box)
                            intrpltd_box = [0, 0, (0, 0, 0, 0)]
                        except IndexError:
                            print("Index Error at enerating intrpltd_box, frameNum={}".format(counter_))
                    frame_counter += 1
                else:
                    pass
            randomCounter_ = counter_
            for l in range(interval):
                if len(bbox) > 0:
                    for k in range(len(bbox)):
                        # Generating bboxes on an image
                        intervalVid[l] = myRect(intervalVid[l], intervalCoord, (l * len(bbox) + k))
                if isOutput:
                    randomCounter_ += 1
                    out2.write(intervalVid[l])
            dispay_info(result_tmp2, 3, "Inactive", counter_,
                        frame_count_total)  # Adding text to imgcopy to show updates. "Final result..."
            for m in range(interval):
                intervalVid_tmp = intervalVid
                try:
                    if Cam2_flag == 1:
                        cv2.namedWindow("AutoTracking Cam2", cv2.WINDOW_NORMAL)
                        if m + 1 == interval:
                            dispay_info(intervalVid_tmp[m], 5, "Inactive", counter_ - interval + m + 1,
                                        frame_count_total)
                        else:
                            dispay_info(intervalVid_tmp[m], 5, "Active", counter_ - interval + m + 1,
                                        frame_count_total)
                        cv2.imshow("AutoTracking Cam2", intervalVid_tmp[m])
                    else:
                        cv2.namedWindow("AutoTracking Cam1", cv2.WINDOW_NORMAL)
                        if m + 1 == interval:
                            dispay_info(intervalVid_tmp[m], 4, "Inactive", counter_ - interval + m + 1,
                                        frame_count_total)
                        else:
                            dispay_info(intervalVid_tmp[m], 4, "Active", counter_ - interval + m + 1,
                                        frame_count_total)
                        cv2.imshow("AutoTracking Cam1", intervalVid_tmp[m])
                    cv2.imshow("Label Editor", result_tmp2)
                except IndexError:
                    print("Index Error at replay function frameNum={}".format(counter_))
                cv2.waitKey(0) & 0xff
            intervalVid = []
            intervalCoord = []
            coor1 = bbox
            if myflag:
                coor1 = tmp_bbox
                tmp_bbox = []
                myflag = False
            bbox = []
            coordif = []
            # cv2.waitKey(0) & 0xff

        ############# Last interval ##################################
        elif (frame_count_total - counter_) < interval and counter_ == frame_count_total:
            print("\n\n\n Last_Frame\n\n\n")
            result2 = result.copy()
            result_tmp = result.copy()

            if Cam2_flag == 0:
                dispay_info(result2, 1, "Active", counter_,
                            frame_count_total)  # Adding text to result2. "Yolo's result. Editing is enabled...")
            elif Cam2_flag == 1:
                dispay_info(result_tmp, 1, "Inactive", counter_,
                            frame_count_total)  # Adding text to result2. "Yolo's result. Editing is enabled..."
                cv2.imshow("Label Editor", result_tmp)
                print("camera1_tracker(output_path3, frame_count_total, counter_, interval)", output_path3,
                      frame_count_total, counter_, interval)
                camera1_tracker(output_path3, counter_, interval)
                dispay_info(result2, 1, "Active", counter_,
                            frame_count_total)  # Adding text to result2. "Yolo's result. Editing is enabled..."
            imgcopy1 = yolo.roiCUT(result2, imgcopy)
            dispay_info(imgcopy1, 2, "Active", counter_,
                        frame_count_total)  # Adding text to imgcopy to show updates. "Drawing is enabled..."
            result = yolo.drawRect(imgcopy1, imgcopy)
            result_tmp = result.copy()
            result_tmp2 = result.copy()
            dispay_info(result_tmp, 3, "Active", counter_,
                        frame_count_total)  # Adding text to imgcopy to show updates. "Final result"
            cv2.imshow("Label Editor", result_tmp)
            if frame_count_total % interval == 0:
                for i in range(interval):
                    vid2.set(cv2.CAP_PROP_POS_FRAMES, counter_ - interval + i)
                    _, frame2 = vid2.read()
                    intervalVid.append(frame2)
                if len(bbox) > 0:
                    for i in range(len(bbox)):
                        try:
                            dif = (np.subtract(bbox[i][2], coor1[i][2])) / interval
                            coordif.append(dif)
                        except IndexError:
                            print("Index Error at generating dif distance, frameNum={}, 1st condition".format(counter_))
                    for i in range(interval):
                        for j in range(len(bbox)):
                            try:
                                intrpltd_box[2] = np.add(coor1[j][2], coordif[j] * i).astype('int16')
                                intrpltd_box[1] = coor1[j][1]
                                intrpltd_box[0] = frame_counter
                                coorx.append(intrpltd_box)
                                final_xy.append(intrpltd_box)
                                intervalCoord.append(intrpltd_box)
                                intrpltd_box = [0, 0, (0, 0, 0, 0)]
                            except IndexError:
                                print("Index Error at generating intrpltd_box, frameNum={}, 1st condition".format(
                                    counter_))
                        frame_counter += 1
                else:
                    print("No bounding boxes. Skipping to next interval")
                randomCounter_ = counter_
                for l in range(interval):
                    if len(bbox) > 0:
                        for k in range(len(bbox)):
                            # Generating bboxes on an image
                            intervalVid[l] = myRect(intervalVid[l], intervalCoord, (l * len(bbox) + k))
                    if isOutput:
                        randomCounter_ += 1
                        out2.write(intervalVid[l])
                dispay_info(result_tmp2, 3, "Inactive", counter_,
                            frame_count_total)  # Adding text to imgcopy to show updates. "Final result"
                for m in range(interval):
                    intervalVid_tmp = intervalVid
                    try:
                        if Cam2_flag == 1:
                            dispay_info(intervalVid_tmp[m], 5, "Active", counter_ - interval + m + 1,
                                        frame_count_total)
                            cv2.imshow("AutoTracking Cam2", intervalVid_tmp[m])
                        else:
                            dispay_info(intervalVid_tmp[m], 4, "Active", counter_ - interval + m + 1,
                                        frame_count_total)
                            cv2.imshow("AutoTracking Cam1", intervalVid_tmp[m])
                        cv2.imshow("Label Editor", result_tmp2)  # Final result
                    except IndexError:
                        print("Index Error at replay function, frameNum={}, 1st condition".format(counter_))
                    cv2.waitKey(0) & 0xff
                intervalVid = []
                intervalCoord = []
            else:
                for i in range(frame_count_total % interval):
                    vid2.set(cv2.CAP_PROP_POS_FRAMES, counter_ - (frame_count_total % interval) + i)
                    _, frame2 = vid2.read()
                    intervalVid.append(frame2)
                if len(bbox) > 0:
                    for i in range(len(bbox)):
                        try:
                            dif = (np.subtract(bbox[i][2], coor1[i][2])) / (frame_count_total % interval)
                            coordif.append(dif)
                        except IndexError:
                            print("Index Error at generating dif distance, frameNum={}, 2nd condition".format(counter_))
                    for i in range(frame_count_total % interval):
                        for j in range(len(bbox)):
                            try:
                                intrpltd_box[2] = np.add(coor1[j][2], coordif[j] * i).astype('int16')
                                intrpltd_box[1] = coor1[j][1]
                                intrpltd_box[0] = frame_counter
                                coorx.append(intrpltd_box)
                                final_xy.append(intrpltd_box)
                                intervalCoord.append(intrpltd_box)
                                intrpltd_box = [0, 0, (0, 0, 0, 0)]
                            except IndexError:
                                print("Index Error at generating intrpltd_box, frameNum={}, 2nd condition".format(
                                    counter_))
                        frame_counter += 1
                    randomCounter_ = counter_
                else:
                    pass
                for l in range(frame_count_total % interval):
                    if len(bbox) > 0:
                        for k in range(len(bbox)):
                            # Generating bboxes on an image
                            intervalVid[l] = myRect(intervalVid[l], intervalCoord, (l * len(bbox) + k))
                        if isOutput:
                            randomCounter_ += 1
                            out2.write(intervalVid[l])
                    else:
                        pass
                dispay_info(result_tmp2, 3, "Inactive", counter_,
                            frame_count_total)  # Adding text to imgcopy to show updates. "Final result"
                for m in range(frame_count_total % interval):
                    intervalVid_tmp = intervalVid
                    try:
                        if Cam2_flag == 1:
                            dispay_info(intervalVid_tmp[m], 5, "Active", counter_ - interval + m + 1,
                                        frame_count_total)
                            cv2.imshow("AutoTracking Cam2", intervalVid_tmp[m])
                        else:
                            dispay_info(intervalVid_tmp[m], 4, "Active", counter_ - interval + m + 1,
                                        frame_count_total)
                            cv2.imshow("AutoTracking Cam1", intervalVid_tmp[m])
                        cv2.imshow("Label Editor", result_tmp2)
                    except IndexError:
                        print("Index Error at replay function, frameNum={}, 2nd condition".format(counter_))
                    cv2.waitKey(0) & 0xff
                intervalVid = []
                intervalCoord = []
            # Adding the new objects
            if myflag:
                a = len(tmp_bbox) - len(coor1)
                print("len(bbox), len(coor1), len(tmp_bbox) ? ", len(bbox), len(coor1), len(tmp_bbox))
                print("Number of new objects ? ", a)
                for x in range(a):
                    try:
                        print("Added: ", tmp_bbox[-(a - x)])
                        final_xy.append(tmp_bbox[-(a - x)])
                    except IndexError:
                        print("Index Error at adding new objects in last frame")
            cv2.waitKey(0) & 0xff

        print("frame_count_total, counter_", frame_count_total, counter_)
        if (counter_ == frame_count_total):
            print("\nLabeling is finished.\n")
            print("WRITING TO FILE")
            print("\n")
            write2File(final_xy)
        counter_ += 1
        # if isOutput:
        # out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    Cam2_flag += 1
    if Cam2_flag == 1:
        counter_ = 1
        interval = 10
        frame_count_total = 0
        final_coor_name = "./coordinates_.txt"
        final_coor_name = final_coor_name[:-4] + dtime.datetime.today().strftime('%Y%m%d%H%M') + final_coor_name[-4:]
        coorx = []
        coor1 = []  # previous frame data

        frame_counter = 0
        intrpltd_box = [frame_counter, 0, (0, 0, 0, 0)]  # temp variable for full data on one object
        output_path3 = output_path2
        detect_video(yolo, video_path, output_path="./OutputVid_.mp4")
    yolo.close_session()
