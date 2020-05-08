import sys
import numpy as np
import cv2
import os
from tkinter.filedialog import askopenfilename, Tk
import math

# Added functionalities:
    # Manually remove bboxes, and draw new ones
    # Auto identify the duration of the video input file in number of frames => frame_count_total variable
    # Handle missing objects and new objects by removal and addition functions
    # Generate output video
    # Display Two screens, an editor and replay of the interpolated frames



# A function for writing bbox coordinates into a file
def write2File(final_bbox, final_coor_name):
    for f in range(len(final_bbox)):
        print("Frame:", final_bbox[f])
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
    idn = 0

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
    video_path = os.path.split(video_path)[0] + "/" + os.path.split(video_path)[1]
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
            #print("mybbox, i", len(mybbox), "\n", i)
            cv2.rectangle(img, (int(mybbox[i][2][0]), int(mybbox[i][2][1])),
                          (int(mybbox[i][2][2]), int(mybbox[i][2][3])), (0, 255, 0), 2)
            cv2.rectangle(img, (int(mybbox[i][2][0]), int(mybbox[i][2][1]) - 20),
                          ((int(mybbox[i][2][0]) + 30), int(mybbox[i][2][1])), (0, 255, 0), -1)
            cv2.putText(img, str(mybbox[i][1]), (int(mybbox[i][2][0]) + 1, (int(mybbox[i][2][1]) - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    except Exception as e:
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
    return img


def sortFunc(data_0, data_x, sort_flag):
    print("sortFunc()\n")
    print("previous_data_0 ", data_0, '\n')
    print("current_data_x ", data_x, '\n')

    data_x = trunc_0s(data_x)
    data_0 = trunc_0s(data_0)

    data_xx = data_x[:]
    data_00 = data_0[:]

    if len(data_0) > 0 and len(data_x) > 0:
        # ----------------------------------------------------------------
        if len(data_0) == len(data_x):
            sort_flag = False
            print("if len()s are equal\n")
            for i in range(len(data_0)):
                for j in range(len(data_x)):
                    if data_0[i][1] == data_x[j][1]:
                        data_xx[i] = data_x[j].copy()
            print("final data_x", data_xx, "\n")
        # ----------------------------------------------------------------
        elif len(data_0) < len(data_x):
            sort_flag = True
            print("if len()s are not equal 0 < x \n")
            for i in range(len(data_0)):
                for j in range(len(data_x)):
                    if data_0[i][1] == data_x[j][1]:
                        data_00[i] = data_x[j].copy()
            arr = []
            for i in range(len(data_x)):
                index = False
                for j in range(len(data_0)):
                    try:
                        if data_x[i] == data_00[j]:
                            index = True
                    except IndexError:
                        print("IndexError")
                arr.append(index)
            for i in range(len(arr)):
                if arr[i] == False:
                    data_00.append(data_x[i])
            data_xx = data_00[:]
            data_00 = data_0[:]
            print("final data_x", data_xx, "\n")
            # ----------------------------------------------------------------
        elif len(data_0) > len(data_x):
            sort_flag = False
            print("if len()s are not equal 0 > x \n")
            for i in range(len(data_x)):
                for j in range(len(data_0)):
                    if data_x[i][1] == data_0[j][1]:
                        data_00[i] = data_0[j].copy()
            del data_00[-1]
            print("final data_0", data_00, "\n")
    else:
        print("One of arrays is empty: data_0, data_x: ", len(data_0), len(data_x))

    return data_00, data_xx, sort_flag


# Second camera related functions

frame_count_total1 = 0
# Replay previous results
def camera1_tracker(video_path, counter_, interval):
    global frame_count_total1
    intervalVid1 = []
    status = ""
    interval0 = interval
    vid1 = cv2.VideoCapture(video_path)
    frame_count_total1 = int(vid1.get(cv2.CAP_PROP_FRAME_COUNT))  # total number of frames in a video file
    if not vid1.isOpened():
        raise IOError("Couldn't open webcam or video")
    try:
        # Frame 1
        if counter_ == 1:
            vid1.set(cv2.CAP_PROP_POS_FRAMES, 0)
            _, frame1 = vid1.read()
            intervalVid1.append(frame1)
            interval0 = 1
        # n-th Frame
        elif (counter_ - 1) % interval == 0:
            print("camera1_tracker() Frame:{}".format(counter_))
            for i in range(interval):
                vid1.set(cv2.CAP_PROP_POS_FRAMES, counter_ - interval + i)
                _, frame1 = vid1.read()
                intervalVid1.append(frame1)
                interval0 = interval
        # Last frame
        else:
            print("camera1_tracker() Frame:{}".format(counter_))
            if frame_count_total1 % interval == 0:
                for i in range(interval):
                    vid1.set(cv2.CAP_PROP_POS_FRAMES, counter_ - interval + i)
                    _, frame1 = vid1.read()
                    intervalVid1.append(frame1)
                    interval0 = interval
            else:
                for i in range(frame_count_total1 % interval):
                    vid1.set(cv2.CAP_PROP_POS_FRAMES, counter_ - (frame_count_total1 % interval) + i)
                    _, frame1 = vid1.read()
                    intervalVid1.append(frame1)
                    interval0 = (frame_count_total1 % interval)
    except IndexError:
        print("Index Error")
    for i in range(interval0):
        try:
            if i + 1 == interval0:
                if counter_ == 1:
                    status = "Active"
                else:
                    status = "Inactive"
                dispay_info(intervalVid1[i], 4, status, counter_ - interval0 + i + 1, frame_count_total1)
            else:
                status = "Active"
                dispay_info(intervalVid1[i], 4, status, counter_ - interval0 + i + 1, frame_count_total1)
            cv2.imshow("AutoTracking Cam1", intervalVid1[i])
            cv2.waitKey(0) & 0xff
        except IndexError:
            print("Index Error")

# Display text to help user with labeling
def dispay_info(image, num, status, counter, frame_count):
    cv2.putText(image, text="Frame: {}".format(counter), org=(15, 50),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.80, color=(0, 255, 255), thickness=2)
    cv2.putText(image, text="Total frames: {}".format(frame_count), org=(15, 75),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.80, color=(0, 255, 255), thickness=2)
    cv2.putText(image, text="Window status: ", org=(15, 100),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.85, color=(0, 255, 255), thickness=2)
    if status == "Active":
        cv2.putText(image, text=status, org=(230, 100),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.80, color=(0, 255, 0), thickness=2)
    else:
        cv2.putText(image, text=status, org=(230, 100),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.80, color=(255, 0, 0), thickness=2)
    if num == 1:
        cv2.putText(image, text="Yolo's result. Editing bboxes is enabled. Press 'ESC' to continue. ", org=(15, 125),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.80, color=(0, 255, 255), thickness=2)
    elif num == 2:
        cv2.putText(image, text="Editing bboxes finished. Manual labeling is enabled. Press 'ESC' to continue.",
                    org=(15, 150),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.80, color=(0, 255, 255), thickness=2)
    elif num == 3:
        cv2.putText(image, text="Labeling for current frame is finished. Press 'ESC' to continue.",
                    org=(15, 125),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.80, color=(0, 255, 255), thickness=2)
    elif num == 4:
        cv2.putText(image, text="Replay of Camera 1 interval frames.",
                org=(15, 125),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.80, color=(0, 255, 255), thickness=2)
    elif num == 5:
        cv2.putText(image, text="Replay of Camera 2 interval frames.",
                    org=(15, 125),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.80, color=(0, 255, 255), thickness=2)
    else:
        print("Number flag is empty.")

    return image


'''
for i in range(len(bbox)):
    if (y2 - y1) <= 20 and (x2 - x1) <= 20:
        eucl_find(bbox, [1, 1, (x1, y1, x2, y2)] )
print("id area?, ", y1,y2, x1,x2)

'''
