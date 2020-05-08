import cv2
import numpy as np
import os
from os.path import isfile, join

pathIn = './frames/'
pathOut = './output_2020march04.mp4'
final_data = "./20200227_02.txt"
num = 3


def write2vid(dir, pathIn, pathOut, num):
    data = (open(dir, 'r')).read().split()
    data = np.array(data).reshape(int(len(data) / 5), 5)

    # check if your input data at this stage is correct
    # print(data[0])
    # expected output:
    # ['veh0' '902' '535' '1055' '652']

    fps = 30.0

    font = cv2.FONT_HERSHEY_SIMPLEX

    def convert_frames_to_video(pathIn, pathOut, fps):
        frame_array = []
        files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
        files.sort()
        for i in range(len(files)):
            filename = pathIn + files[i]
            # reading each files
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width, height)
            # inserting the frames into an image array
            for j in range(num):
                label, left, top, right, bottom = data[i * num + j, :]
                #print((int(left), int(top)), (int(right), int(bottom)))  # , (0, 255, 0), 3)  # Main boundry
                img = cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0),
                                    2)  # Main boundry
                img = cv2.rectangle(img, (int(left), int(top) - 30), ((int(left) + 60), int(top)), (0, 255, 0),
                                    2)  # Title
                cv2.putText(img, label, (int(left) + 2, int(top) - 5), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

            frame_array.append(img)
        out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        for i in range(len(frame_array)):
            # writing to a image array
            out.write(frame_array[i])
        out.release()

    fps = 30.0
    convert_frames_to_video(pathIn, pathOut, fps)
    print("Output video file is generated at: {}".format(pathOut))
    print("Done")


write2vid(final_data, pathIn, pathOut, num)