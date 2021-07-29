import sys
import argparse
import os
from yolo import YOLO, detect_video
from PIL import Image


def detect_img(yolo):
    datasets_folder_path = '/home/HeadPoseEstimation-WHENet/datasets/'

    # BIWI part
    biwi_path = datasets_folder_path + 'BIWI/'

    biwi_annotations = open(biwi_path + 'BIWI_annotations.txt', 'r', encoding='utf8')
    annotations = biwi_annotations.read().split('\n')
    biwi_annotations.close()

    print('Buscando los rostros en las', len(annotations), 'imagenes de BIWI')
    """
    biwi_bboxes = ''
    for line in annotations:
        image_path = biwi_path + line.split(' ')[0]
        a, b, c, d = yolo.detect_image(Image.open(image_path))
        if a == -1:
            continue
        biwi_bboxes += image_path + ',' + str(a) + ' ' + str(b) + ' ' + str(c) + ' ' + str(d) + '\n'

    print('Guardando en ', biwi_path + 'BIWI_bboxes.txt')
    biwi_file = open(biwi_path + 'BIWI_bboxes.txt', "w")
    biwi_file.write(biwi_bboxes)
    biwi_file.close()
    """

    # AFLW2000 part
    aflw_path = datasets_folder_path + 'AFLW2000/'
    print('Buscando rostros en las imagenes de AFLW2000')

    aflw_bboxes = ''
    for file in os.listdir(aflw_path):
        if file.endswith('jpg'):
            image_path = aflw_path + file
            print(image_path)
            a, b, c, d = yolo.detect_image(Image.open(image_path))
            if a == -1:
                continue
            aflw_bboxes += image_path + ',' + str(a) + ' ' + str(b) + ' ' + str(c) + ' ' + str(d) + '\n'

    print('Guardando en ', aflw_path + 'AFLW2000_bboxes.txt')
    aflw_file = open(aflw_path + 'AFLW2000_bboxes.txt', "w")
    aflw_file.write(aflw_path)
    aflw_file.close()

    yolo.close_session()


FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str, required=False, default='./path2your_video',
        help="Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help="[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    detect_img(YOLO(**vars(FLAGS)))
