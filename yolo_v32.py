# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video

"""

import colorsys
from timeit import default_timer as timer
import datetime as dtime
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
from keras.utils import multi_gpu_model
import easygui as ez
from myFuncs import *
import _tkinter

# Global variables
counter_ = 1
interval = 10
frames_total = 0
final_coor_name = "/home/jsr1611/PycharmProjects/just_yolo3/output_files/coordinates_.txt"
final_coor_name = final_coor_name[:-4] + dtime.datetime.today().strftime('%Y%m%d%H%M') + final_coor_name[-4:]
bboxNxt = []
bboxPrev = []  # previous frame data

frameNum = 0
intrpltd_box = [frameNum, 0, (0, 0, 0, 0), ""]  # temp variable for full data on one object
cam2_flag = 0
output_path3 = ""
id_max = 0
status = ""  # Window status indicator, ex: Active or Inactive

############ Drawing function ##################################
def drawRect(img1, img2):
    # img1 - input image with bboxes for displaying and drawing on
    # img2 - orignal image to be output with bboxes only
    global bbox, bboxPrev, label, id_n, my_label, rm_labels, numOfObj, tmp_bbox, sort_flag, cam2_flag, id_max, eucfind_data0
    id_n = 1
    label = len(bboxPrev) + id_n
    tmp_bbox = []
    bbox_ = []
    sort_flag = False
    newObjCounter = 1
    # print("coor1 before eucfind_data0 gets it: \n", coor1)
    eucfind_data0 = bboxPrev[:]
    eucfind_data1 = bboxPrev[:]  # checked

    def label_me(label, eucfind_data0, eucfind_data1):
        msg = "Selected ID is  {}. Is this object ID correct?".format(label)
        title = "Object ID selection process"
        choices = ["Yes", "Modify", "Cancel"]
        key = ez.buttonbox(msg, title, choices)
        print(msg)
        if key == "Yes":
            eucfind_data1 = eucfind_data0[:]
            print("selected Yes, label=", label)
            pass
        elif key == "Modify":
            eucfind_data0 = eucfind_data1[:]
            label = ez.enterbox("Please enter the ID of the object")
            print("selected Modify, label=", label)
        else:
            label = '00'
            print("selected Cancel, label=", label)
            pass
        return label, eucfind_data0, eucfind_data1

    while True:
        r = cv2.selectROI("Label Editor", img1, fromCenter=False, showCrosshair=True)
        x1 = int(r[0])
        y1 = int(r[1])
        x2 = int(r[0] + r[2])
        y2 = int(r[1] + r[3])

        if cam2_flag == 1:  # Label tag (ID) for second camera
            print("Coor1 before labeling for second cam: \n", bboxPrev)
            while True:
                print("\nlen(eucfind_data0) = {}, len(eucfind_data1) = {}\n".format(len(eucfind_data0), len(eucfind_data1)))
                try:
                    print("Label(before):", label)
                    if counter_ == 1:
                        label = ez.enterbox("Please enter the id of the object")
                        print("Label: ", label)
                    elif counter_ > 1:
                        data_x = [frameNum, 0, (x1, y1, x2, y2)]
                        if data_x[2][0] == 0 and data_x[2][1] == 0 and data_x[2][2] == 0:
                            # print("jumped?")
                            break
                        else:
                            if len(eucfind_data0) > 0:
                                eucfind_data0, bbox_ = eucl_find(eucfind_data0, data_x)
                                label = bbox_[1]
                                label, eucfind_data0, eucfind_data1 = label_me(label, eucfind_data0, eucfind_data1)
                            else:
                                for i in range(len(bboxPrev)):
                                    if id_max < int(bboxPrev[i][1]):
                                        id_max = int(bboxPrev[i][1])
                                for j in range(len(bbox)):
                                    if id_max < int(bbox[j][1]):
                                        id_max = int(bbox[j][1])
                                id_max += 1
                                print("Increment ID by one?", label)
                                label = id_max
                                label, eucfind_data0, eucfind_data1 = label_me(label, eucfind_data0, eucfind_data1)

                    else:
                        label = '00'
                        print("selected else, label=", label)
                    print("Label(after1):", label)
                    if type(int(label)) is int:
                        print("label given: {}, my_flag3={}, counter_={}".format(label, cam2_flag, counter_))
                        break
                    else:
                        print("Please enter the object id correctly. The id should be an integer number!", label)
                except TypeError:
                    print("Type Error at bbox appending frameNum={}".format(counter_))
                except ValueError:
                    print("Value Error")

        else:  # Label tag (ID) for first camera
            if counter_ > 1:  # when working for consecutive frames
                if len(rm_labels) == 1 and newObjCounter == 1:
                    label = rm_labels[0]
                    print("label:{},myflg3={}, rm_lbls#={}".format(label, cam2_flag, len(rm_labels)))
                elif len(rm_labels) > 1 or newObjCounter > 1:
                    for i in range(len(bboxPrev)):
                        if id_max < bboxPrev[i][1]:
                            id_max = bboxPrev[i][1]
                    label = id_max + id_n
                    id_n += 1
                    print("label:{},myflg3={},countr={},rm_lbl# = {}".format(label, cam2_flag, counter_,
                                                                             len(rm_labels)))
            else:  # when working for first frame
                label = id_n - 1
                id_n += 1
        # Bounding box filtration

        try:
            print("Label(after try):", label)
            bbox_ = [frameNum, int(label), (x1, y1, x2, y2)]
            bbox_2 = trunc_0s(bbox_)
            # print("bbox_2, bbox_ (check jumped or not)", bbox_2, bbox_)
            if bbox_ != bbox_2:
                pass
                print("jumped?")
            else:
                # print("not jumped?")
                bbox.append([frameNum, int(label), (x1, y1, x2, y2)])
                print("not jumped? label=", label)
                myRect(img1, bbox_, 'one')  # Generating bboxes on an image
                print("Drawn bbox:", [frameNum, label, (x1, y1, x2, y2)])
        except ValueError:
            print("Value Error")
        newObjCounter += 1
        # Bounding box additin
        if cv2.waitKey(0) & 0xff == 27:
            # print("Coor1 after ESC key is pressed: \n", coor1)
            if cam2_flag == 1:  # adding bbox for second camera
                print("added bbox for second camera?")
                if counter_ > 1:
                    bboxPrev, bbox, sort_flag = sortFunc(bboxPrev, bbox, sort_flag)
                    if sort_flag == False:
                        tmp_bbox = bbox
                        #bbox = tmp_bbox[:len(coor1)]
                        print("tmp_bbox=\n", tmp_bbox, "\n", bbox)

            else:  # adding bbox for first camera
                print("added bbox for first camera?")
                if counter_ > 1:  # adding for consecutive frames first camera
                    if len(bbox) < len(bboxPrev) and len(bboxPrev) > 0:
                        sort_flag = False
                        bboxPrev, bbox = eucl_sort_remove(bboxPrev, bbox)
                        bboxPrev, bbox = eucl_sort(bboxPrev, bbox)
                        # print("bbox, coor1 after eucl_sort() \n", bbox,"\n", coor1)
                    elif len(bbox) == len(bboxPrev) and (len(bbox) > 0 and len(bboxPrev) > 0):
                        bboxPrev, bbox = eucl_sort(bboxPrev, bbox)
                        # print("bbox, coor1 after eucl_sort() \n", bbox, "\n", coor1)
                    else:
                        sort_flag = True
                        bboxPrev, bbox = eucl_sort_add(bboxPrev, bbox)
                        tmp_bbox = bbox

                        print("tmp_bbox=\n", tmp_bbox)
                else:  # adding for first frame first camera
                    if cam2_flag == 0:  # if it is first camera
                        tmp_id = 0
                        print("Frames before renaming: bbox \n", bbox)
                        if len(bbox) > 0:
                            for l in range(len(bbox)):
                                bbox[l][1] = tmp_id
                                tmp_id += 1
                            print("Frames after renaming: bbox \n", bbox)
                        else:
                            pass
                # print("coor1, myflag after eucl before imshow(): \n", coor1, "\n", myflag)
            print("added bbox for second(1) or first camera(0) = ", cam2_flag)
            if len(bbox) > 0:
                for i in range(len(bbox)):
                    # print("really added bbox\n")
                    myRect(img2, bbox, i)  # Generating bboxes on an image
                cv2.imshow("Label Editor", img2)
                '''if sort_flag and cam2_flag == 0:
                    try:
                        bbox = bbox[:len(bboxPrev)]
                        print("what is happening here?", bbox)
                    except Exception as e:
                        print(e)
                '''
            break
    rm_labels = []
    return img2, bbox

############ Removing function  ################################
def roiCUT(img1, img2):
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
        y1 = int(r[1])  # left                  left top right bottom
        x2 = int(r[0] + r[2])  # bottom
        y2 = int(r[1] + r[3])  # right
        imgCrop = img2[y1:y2, x1:x2]
        # rois.append([y1, y2, x1, x2])
        target.append([x1, y1, x2, y2])
        print("Area selected for cutting(erasing) and removing the bboxes inside:\n", [x1, y1, x2, y2])
        buffer.append(imgCrop)
        img4 = img2.copy()
        img[y1:y2, x1:x2] = img2[y1:y2, x1:x2]
        '''
        for i in range(len(bbox)):
            if (y2 - y1) <= 20 and (x2 - x1) <= 20:
                eucl_find(bbox, [1, 1, (x1, y1, x2, y2)] )
        print("id area?, ", y1,y2, x1,x2)
        '''


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
        global bbox, frameNum
        start = timer()
        img = np.asarray(image)
        bbox = []
        id_num = 0
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
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            bbox.append([frameNum, id_num, (left, top, right, bottom), "{}".format(predicted_class)])
            centroid_xy.append(int((left + right) / 2))
            centroid_xy.append(int((top + bottom) / 2))
            print(frameNum, id_num, (left, top, right, bottom), "{}".format(predicted_class))
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

        end = timer()
        # print(end - start)
        return img

    def close_session(self):
        self.sess.close()


def detect_video(yolo, video_path, output_path="/home/jsr1611/PycharmProjects/just_yolo3/output_vid/OutputVid_.mp4"):
    global bbox, bboxPrev, counter_, interval, frames_total, bboxNxt, intrpltd_box, coordif, final_xy, \
        frameNum, sort_flag, tmp_bbox, intervalCoord, out2, randomCounter_, cam2_flag, output_path3, final_coor_name

    final_xy = []  # final array for bbox coordinates = ground truth data
    intervalVid = []  # array for interval frames (skipped frames interpolation)

    try:
        default_path = './2020_0109_140236_NOR_FILE.AVI'
        if cam2_flag == 0:
            print("Default input file path: ", default_path)
            while True:
                fastTrack2ndCam = int(ez.enterbox("Enter 1 if you have camera 1 output, otherwise enter 0."))
                if fastTrack2ndCam == 1:
                    output_path3 = fileOpenClicked(default_path)
                    video_path = output_path3
                    cam2_flag = 1
                    break
                elif fastTrack2ndCam == 0:
                    cam2_flag = 0
                    video_path = fileOpenClicked(default_path)
                    break
                else:
                    print("You entered something wrong. Please, enter 1 or 0 to proceed.")
        if cam2_flag == 1:
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
    frames_total = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))  # total number of frames in a video file
    print("Video file duration: {0:<d} frames, {1:<.2f} seconds\n".format(frames_total, frames_total / 30.0))

    video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        # set the [video] output file write function
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
            if cam2_flag == 1:
                print("2nd camera video labeling is finished")
                print("Program finished the job.\n")
            elif cam2_flag == 0:
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
            frameNum = 0
            print("\n\n First_Frame\n\n")
            result_tmp = result.copy()
            result2 = result.copy()
            cv2.namedWindow("Label Editor", cv2.WINDOW_NORMAL)
            if cam2_flag == 0:
                dispay_info(result2, 1, "Active", counter_,
                            frames_total)  # Adding text to result2. "Yolo's result. Editing is enabled..."
            elif cam2_flag == 1:
                dispay_info(result_tmp, 1, "Inactive", counter_,
                            frames_total)  # Adding text to result2. "Yolo's result. Editing is enabled..."
                cv2.imshow("Label Editor", result_tmp)
                cv2.namedWindow("AutoTracking Cam1", cv2.WINDOW_NORMAL)
                camera1_tracker(output_path3, counter_, interval)
                vidtmp = cv2.VideoCapture(output_path3)
                vidtmp.set(cv2.CAP_PROP_POS_FRAMES, 0)
                _, frametmp = vidtmp.read()
                dispay_info(frametmp, 4, "Inactive", counter_,
                            frames_total)
                cv2.imshow("AutoTracking Cam1", frametmp)
                dispay_info(result2, 1, "Active", counter_,
                            frames_total)  # Adding text to result2. "Yolo's result. Editing is enabled..."
            imgcopy = roiCUT(result2, imgcopy)
            imgcopy2 = imgcopy.copy()
            dispay_info(imgcopy2, 2, "Active", counter_,
                        frames_total)  # Adding text to imgcopy to show updates. "Drawing is enabled..."
            cv2.imshow("Label Editor", imgcopy2)
            result, bbox = drawRect(imgcopy2, imgcopy3)
            result_tmp = result.copy()
            dispay_info(result_tmp, 3, "Active", counter_,
                        frames_total)  # Adding text to imgcopy to show updates. "Final result..."
            cv2.imshow("Label Editor", result_tmp)
            bboxPrev = bbox[:]
            # print("Frame 1 finished. Coor1 = \n", coor1)
            bbox = []
            cv2.waitKey(0) & 0xff

        ############ N-th interval  ##################################

        elif (counter_ - 1) % interval == 0:
            print("\n\n\n Interval_{} \n\n\n".format(int((counter_ - 1) / interval)))
            result2 = result.copy()
            result_tmp = result.copy()
            # print("Frame 2+. Coor1 = \n", coor1)
            if cam2_flag == 0:
                dispay_info(result2, 1, "Active", counter_,
                            frames_total)  # Adding text to result2. "Yolo's result. Editing is enabled..."
            if cam2_flag == 1:
                dispay_info(result_tmp, 1, "Inactive", counter_,
                            frames_total)  # Adding text to result2. "Yolo's result. Editing is enabled..."
                cv2.imshow("Label Editor", result_tmp)
                print("camera1_tracker(output_path3, frame_count_total, counter_, interval)", output_path3,
                      frames_total, counter_, interval)
                camera1_tracker(output_path3, counter_, interval)
                dispay_info(result2, 1, "Active", counter_,
                            frames_total)  # Adding text to result2. "Yolo's result. Editing is enabled..."
            imgcopy = roiCUT(result2, imgcopy)
            imgcopy2 = imgcopy.copy()
            dispay_info(imgcopy2, 2, "Active", counter_,
                        frames_total)  # Adding text to imgcopy to show updates. "Drawing is enabled..."
            # Displaying edited Yolo's result and labeling (drawing) enabled ---------------------------------------
            cv2.imshow("Label Editor", imgcopy2)
            result, bbox = drawRect(imgcopy2, imgcopy3)     # result=newly drawn image, bbox=cur frame bbox
            result_tmp = result.copy()
            result_tmp2 = result.copy()
            dispay_info(result_tmp, 3, "Active", counter_,
                        frames_total)  # Adding text to imgcopy to show updates. "Final result..."
            cv2.imshow("Label Editor", result_tmp)
            print("After drawRect(), bbox, coor1\n", bbox, "\n", bboxPrev)
            # if cur frame(bbox) has new objects which prev frame(coor1) didn't have.
            if sort_flag == True:
                tmp_bbox = bbox
                bbox = tmp_bbox[:len(bboxPrev)]
            if counter_ == interval + 1:
                print("First frame intervalVid\n len(intrVid)", len(intervalVid))
                for i in range(interval):
                    vid2.set(cv2.CAP_PROP_POS_FRAMES, i)
                    _, frame2 = vid2.read()
                    intervalVid.append(frame2)
                    print("added: (counter) (fr num): ", counter_, i)
            elif counter_ > interval + 1:
                print("Consecutive frames intervalVid\n len(intrVid)", len(intervalVid))
                for i in range(interval):
                    vid2.set(cv2.CAP_PROP_POS_FRAMES, counter_ - interval - 1 + i)
                    _, frame2 = vid2.read()
                    intervalVid.append(frame2)
                    print("added: (counter) (fr num): ", counter_, counter_ - interval - 1 + i)
            try:
                print("Any error?\n bbox:\n", bbox, "\ncoor1:\n", bboxPrev)
                for i in range(len(bbox)):
                    dif = (np.subtract(bbox[i][2], bboxPrev[i][2])) / interval
                    coordif.append(dif)
                    dif = []
            except IndexError:
                print("Index Error at generating dif distance, frameNum={}".format(counter_))
            for i in range(interval):
                if len(bbox) > 0:
                    for j in range(len(bbox)):
                        try:
                            intrpltd_box[2] = np.add(bboxPrev[j][2], coordif[j] * i).astype('int16')
                            intrpltd_box[1] = bboxPrev[j][1]
                            intrpltd_box[0] = frameNum
                            bboxNxt.append(intrpltd_box)
                            final_xy.append(intrpltd_box)
                            intervalCoord.append(intrpltd_box)
                            intrpltd_box = [0, 0, (0, 0, 0, 0)]
                        except IndexError:
                            print("Index Error at generating intrpltd_box, frameNum={}".format(counter_))
                    frameNum += 1
                else:
                    pass
            #randomCounter_ = counter_
            for l in range(interval):
                if len(bbox) > 0:
                    for k in range(len(bbox)):
                        # Generating bboxes on an image
                        intervalVid[l] = myRect(intervalVid[l], intervalCoord, (l * len(bbox) + k))
                if isOutput:
                    #randomCounter_ += 1
                    out2.write(intervalVid[l])
            dispay_info(result_tmp2, 3, "Inactive", counter_,
                        frames_total)  # Adding text to imgcopy to show updates. "Final result..."
            # Replay of interpolated frames
            for m in range(interval):
                intervalVid_tmp = intervalVid[:]
                try:
                    if cam2_flag == 1:
                        cv2.namedWindow("AutoTracking Cam2", cv2.WINDOW_NORMAL)
                        if m + 1 == interval:
                            dispay_info(intervalVid_tmp[m], 5, "Inactive", counter_ - interval + m,
                                        frames_total)
                        else:
                            dispay_info(intervalVid_tmp[m], 5, "Active", counter_ - interval + m,
                                        frames_total)
                        cv2.imshow("AutoTracking Cam2", intervalVid_tmp[m])
                    else:
                        cv2.namedWindow("AutoTracking Cam1", cv2.WINDOW_NORMAL)
                        if m + 1 == interval:
                            dispay_info(intervalVid_tmp[m], 4, "Inactive", counter_ - interval + m,
                                        frames_total)
                        else:
                            dispay_info(intervalVid_tmp[m], 4, "Active", counter_ - interval + m,
                                        frames_total)
                        cv2.imshow("AutoTracking Cam1", intervalVid_tmp[m])
                    cv2.imshow("Label Editor", result_tmp2)
                except IndexError:
                    print("Index Error at replay function frameNum={}".format(counter_))
                cv2.waitKey(0) & 0xff
            intervalVid = []
            intervalCoord = []
            print("coor1, bbox after interval is done\n", bboxPrev, "\n", bbox)
            bboxPrev = bbox[:]
            if sort_flag:
                print("coor1, bbox when sort_flag: {}\n".format(sort_flag), bboxPrev, "\n", bbox)
                bboxPrev = tmp_bbox[:]
                tmp_bbox = []
                sort_flag = False
                print("coor1, bbox when sort_flag: {}\n".format(sort_flag), bboxPrev, "\n", bbox)

            bbox = []
            coordif = []
            # cv2.waitKey(0) & 0xff

        ############# Last interval ##################################
        elif (frames_total - counter_) < interval and counter_ == frames_total:
            print("\n\n\n Last_Frame\n\n\n")
            result2 = result.copy()
            result_tmp = result.copy()

            if cam2_flag == 0:
                dispay_info(result2, 1, "Active", counter_,
                            frames_total)  # Adding text to result2. "Yolo's result. Editing is enabled...")
            elif cam2_flag == 1:
                dispay_info(result_tmp, 1, "Inactive", counter_,
                            frames_total)  # Adding text to result2. "Yolo's result. Editing is enabled..."
                cv2.imshow("Label Editor", result_tmp)
                print("camera1_tracker(output_path3, frame_count_total, counter_, interval)", output_path3,
                      frames_total, counter_, interval)
                camera1_tracker(output_path3, counter_, interval)
                dispay_info(result2, 1, "Active", counter_,
                            frames_total)  # Adding text to result2. "Yolo's result. Editing is enabled..."
            imgcopy1 = roiCUT(result2, imgcopy)
            dispay_info(imgcopy1, 2, "Active", counter_,
                        frames_total)  # Adding text to imgcopy to show updates. "Drawing is enabled..."
            result, bbox = drawRect(imgcopy1, imgcopy)
            result_tmp = result.copy()
            result_tmp2 = result.copy()
            dispay_info(result_tmp, 3, "Active", counter_,
                        frames_total)  # Adding text to imgcopy to show updates. "Final result"
            cv2.imshow("Label Editor", result_tmp)
            # if cur frame(bbox) has new objects which prev frame(coor1) didn't have.
            if sort_flag:
                tmp_bbox = bbox[:]
                bbox = tmp_bbox[:len(bboxPrev)]

            # Scenario 1
            if frames_total % interval == 0:
                print("Last frame. Scenario 1: frames_total % interval == 0")
                for i in range(interval):
                    vid2.set(cv2.CAP_PROP_POS_FRAMES, counter_ - interval + i)
                    _, frame2 = vid2.read()
                    intervalVid.append(frame2)
                if len(bbox) > 0:
                    for i in range(len(bbox)):
                        try:
                            dif = (np.subtract(bbox[i][2], bboxPrev[i][2])) / interval
                            coordif.append(dif)
                        except IndexError:
                            print("Index Error at generating dif distance, frameNum={}, 1st condition".format(counter_))
                    for i in range(interval):
                        if sort_flag and i == interval - 1:
                            for j in range(len(tmp_bbox)):
                                try:
                                    intrpltd_box[2] = tmp_bbox[j][2]
                                    intrpltd_box[1] = tmp_bbox[j][1]
                                    intrpltd_box[0] = frameNum
                                    bboxNxt.append(intrpltd_box)
                                    final_xy.append(intrpltd_box)
                                    intervalCoord.append(intrpltd_box)
                                    #print("intrpltd_box: ", intrpltd_box)
                                    intrpltd_box = [0, 0, (0, 0, 0, 0)]
                                except Exception as e:
                                    print(e)
                            frameNum += 1
                        else:
                            for j in range(len(bbox)):
                                try:
                                    intrpltd_box[2] = np.add(bboxPrev[j][2], coordif[j] * i).astype('int16')
                                    intrpltd_box[1] = bboxPrev[j][1]
                                    intrpltd_box[0] = frameNum
                                    bboxNxt.append(intrpltd_box)
                                    final_xy.append(intrpltd_box)
                                    intervalCoord.append(intrpltd_box)
                                    #print("intrpltd_box ", intrpltd_box)
                                    intrpltd_box = [0, 0, (0, 0, 0, 0)]
                                except Exception as e:
                                    print(e)
                            frameNum += 1
                else:
                    print("No bounding boxes. Skipping to next interval")
                randomCounter_ = counter_
                for l in range(interval):
                    if len(bbox) > 0:
                        if sort_flag and l == interval - 1:
                            for k in range(len(tmp_bbox)):
                                #print("Line: 759 if T, myRect() call i val=", (l * len(bbox) + k))
                                intervalVid[l] = myRect(intervalVid[l], intervalCoord, (l * len(bbox) + k))
                        else:
                            for k2 in range(len(bbox)):
                                #print("Line: 763 else, myRect() call i value=",(l * len(bbox) + k2))
                                # Generating bboxes on an image
                                intervalVid[l] = myRect(intervalVid[l], intervalCoord, (l * len(bbox) + k2))
                    if isOutput:
                        randomCounter_ += 1
                        out2.write(intervalVid[l])
                dispay_info(result_tmp2, 3, "Inactive", counter_, frames_total)
                # Adding text to imgcopy to show updates. "Final result"
                for m in range(interval):
                    intervalVid_tmp = intervalVid
                    try:
                        if cam2_flag == 1:
                            dispay_info(intervalVid_tmp[m], 5, "Active", counter_ - interval + m + 1,
                                        frames_total)
                            cv2.imshow("AutoTracking Cam2", intervalVid_tmp[m])
                        else:
                            dispay_info(intervalVid_tmp[m], 4, "Active", counter_ - interval + m + 1,
                                        frames_total)
                            cv2.imshow("AutoTracking Cam1", intervalVid_tmp[m])
                        cv2.imshow("Label Editor", result_tmp2)  # Final result
                    except Exception as e:
                        print(e)
                    cv2.waitKey(0) & 0xff
                intervalVid = []
                intervalCoord = []
            # Scenario 2
            else:
                print("Last frame. Scenario 2: frames_total % interval != 0")
                for i in range(frames_total % interval):
                    vid2.set(cv2.CAP_PROP_POS_FRAMES, counter_ - (frames_total % interval) + i)
                    _, frame2 = vid2.read()
                    intervalVid.append(frame2)
                if len(bbox) > 0:
                    print("len(bbox) > 0 at Scenario 2", len(bbox))
                    for i in range(len(bbox)):
                        try:
                            print("diff calc")
                            dif = (np.subtract(bbox[i][2], bboxPrev[i][2])) / (frames_total % interval)
                            coordif.append(dif)
                        except Exception as e:
                            print(e)
                    for i in range(frames_total % interval):
                        if sort_flag and i == (frames_total % interval) - 1:
                            for j in range(len(tmp_bbox)):
                                try:
                                    intrpltd_box[2] = tmp_bbox[j][2]
                                    intrpltd_box[1] = tmp_bbox[j][1]
                                    intrpltd_box[0] = frameNum
                                    bboxNxt.append(intrpltd_box)
                                    final_xy.append(intrpltd_box)
                                    intervalCoord.append(intrpltd_box)
                                    intrpltd_box = [0, 0, (0, 0, 0, 0)]
                                except Exception as e:
                                    print(e)
                            frameNum += 1
                        else:
                            for j in range(len(bbox)):
                                try:
                                    intrpltd_box[2] = np.add(bboxPrev[j][2], coordif[j] * i).astype('int16')
                                    intrpltd_box[1] = bboxPrev[j][1]
                                    intrpltd_box[0] = frameNum
                                    bboxNxt.append(intrpltd_box)
                                    final_xy.append(intrpltd_box)
                                    intervalCoord.append(intrpltd_box)
                                    intrpltd_box = [0, 0, (0, 0, 0, 0)]
                                except Exception as e:
                                    print(e)
                            frameNum += 1
                    randomCounter_ = counter_
                    for l in range(frames_total % interval):

                        if sort_flag and l == (frames_total % interval) - 1:
                            for k in range(len(tmp_bbox)):
                                # Generating bboxes on an image
                                intervalVid[l] = myRect(intervalVid[l], intervalCoord, (l * len(bbox) + k))
                        else:
                            for k in range(len(bbox)):
                                # Generating bboxes on an image
                                intervalVid[l] = myRect(intervalVid[l], intervalCoord, (l * len(bbox) + k))
                        if isOutput:
                            randomCounter_ += 1
                            out2.write(intervalVid[l])
                else:
                    print("len(bbox) < 0 at Scenario 2", len(bbox))
                    pass

                dispay_info(result_tmp2, 3, "Inactive", counter_,
                            frames_total)  # Adding text to imgcopy to show updates. "Final result"
                for m in range(frames_total % interval):
                    intervalVid_tmp = intervalVid
                    try:
                        if cam2_flag == 1:
                            dispay_info(intervalVid_tmp[m], 5, "Active", counter_ - interval + m + 1,
                                        frames_total)
                            cv2.imshow("AutoTracking Cam2", intervalVid_tmp[m])
                        else:
                            dispay_info(intervalVid_tmp[m], 4, "Active", counter_ - interval + m + 1,
                                        frames_total)
                            cv2.imshow("AutoTracking Cam1", intervalVid_tmp[m])
                        cv2.imshow("Label Editor", result_tmp2)
                    except Exception as e:
                        print(e)
                    cv2.waitKey(0) & 0xff
                intervalVid = []
                intervalCoord = []
            # Adding the new objects
            '''if sort_flag:
                new_bbox = tmp_bbox[len(bboxPrev):]
                print("len(bbox), len(coor1), len(tmp_bbox) ? ", len(bbox), len(bboxPrev), len(tmp_bbox))
                print("Number of new objects ? ", new_bbox)
                for x in range(len(new_bbox)):
                    try:
                        new_bbox[x][0] = frameNum-1
                        print("Added: ", new_bbox[x])
                        final_xy.append(new_bbox[x])
                    except Exception as e:
                        print(e)
            cv2.waitKey(0) & 0xff
            '''

        print("frame_count_total, counter_", frames_total, counter_)
        if (counter_ == frames_total):
            print("\nLabeling is finished.\n")
            print("WRITING TO FILE")
            print("\n")
            write2File(final_xy, final_coor_name)
        counter_ += 1
        # if isOutput:
        # out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam2_flag += 1
    if cam2_flag == 1:
        counter_ = 1
        interval = 10
        frames_total = 0
        final_coor_name = "/home/jsr1611/PycharmProjects/just_yolo3/output_files/coordinates_.txt"
        final_coor_name = final_coor_name[:-4] + dtime.datetime.today().strftime('%Y%m%d%H%M') + final_coor_name[-4:]
        bboxNxt = []
        bboxPrev = []  # previous frame data

        frameNum = 0
        intrpltd_box = [frameNum, 0, (0, 0, 0, 0)]  # temp variable for full data on one object
        output_path3 = output_path2
        detect_video(yolo, video_path, output_path="/home/jsr1611/PycharmProjects/just_yolo3/output_vid/OutputVid_.mp4")
    yolo.close_session()
