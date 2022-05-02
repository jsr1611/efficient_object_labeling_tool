import sys, os
import time
import argparse
from yolo import YOLO, detect_video
from PIL import Image

outputvid_path = "/home/jsr1611/PycharmProjects/just_yolo3/output_vid/OutputVid_.mp4"

#outputvid_path = outputvid_path[:2] + ez.enterbox("Please enter a name for output video:  ") + outputvid_path[-4:]
print("output video path:", outputvid_path)
def detect_img(yolo):
    while True:
        try:
            img = input('Input image filename:')
            beginTime = time.time()
            image = Image.open(img)
            r_image = yolo.detect_image(image)
            r_image.show()
            procTime = time.time() - beginTime
            print("Fps {}".format(1/procTime))
        except KeyboardInterrupt:
            print(os.linesep + "Exit")
            break
        except:
            print('Open Error! Try again!')
            continue
    yolo.close_session()

FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model_path', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors_path', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes_path', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=True, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default=outputvid_path,
        help = "[Optional] Video output path"
    )

    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = False  # to log device placement (on which device the operation ran)
    # (nothing gets printed in Jupyter, only if you run it standalone)
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras

    FLAGS = parser.parse_args()

    if FLAGS.image:
        print("Video detection mode")
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)

    elif "input" in FLAGS:
        """
                Image detection mode, disregard any remaining command line arguments
                """
        print("Image detection mode")
        print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_img(YOLO(**vars(FLAGS)))
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
