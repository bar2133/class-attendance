import argparse
import copy
import os
import uuid
from typing import List

import cv2
import numpy as np
from facenet_pytorch.models.mtcnn import MTCNN

from classes.camera_handler import CameraHandler
from definitions import ROOT_DIR

# config file save location
ds_images_path = os.path.join(ROOT_DIR, 'data', 'dataset', 'images')
ds_labels_path = os.path.join(ROOT_DIR, 'data', 'dataset', 'labels')
ts_images_path = os.path.join(ROOT_DIR, 'data', 'testset', 'images')
ts_labels_path = os.path.join(ROOT_DIR, 'data', 'testset', 'labels')


def config_parser():
    parser = argparse.ArgumentParser(
        prog='Face collector',
        description='', )
    parser.add_argument('-p', '--pics', required=True, type=int, help='number of picture to capture')
    parser.add_argument('-i', '--id', required=True, type=int, help='class id')
    return parser


def resize_frame(img):
    scale_percent = 60  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


def main():
    # init parser and get args
    parser = config_parser()
    args = vars(parser.parse_args())
    num_of_pics = args.get("pics", 500)
    class_id = args.get("id", -1)

    init_needed_folders_if_not_exist([ds_images_path, ds_labels_path, ts_images_path, ts_labels_path])

    counter = 0
    camera = CameraHandler()
    mtcnn = MTCNN(keep_all=True)
    camera.start()

    save_labels_path = ds_labels_path
    save_images_path = ds_images_path
    while counter < num_of_pics:
        # change saving location from the dataset to testing set after half of the num_of_pics
        if counter == int(num_of_pics * 0.7):
            save_labels_path = ts_labels_path
            save_images_path = ts_images_path
        frame = copy.deepcopy(camera.frame)
        frame = resize_frame(frame)
        # find faces to collect.
        boxes, prob = mtcnn.detect(frame)
        if boxes is None:
            continue
        # convert boxes from float to int
        boxes = np.array(boxes).astype(int)
        for box in boxes:
            save_img_with_box_label(box, class_id, frame, save_images_path, save_labels_path)
        counter += 1
        print(counter)
    camera.stop()


def save_img_with_box_label(box, class_id, frame, save_images_path, save_labels_path):
    file_name = uuid.uuid4()
    # save pic to file
    cv2.imwrite(f"{save_images_path}/{file_name}.jpg", frame)
    # save config file to yolo
    with open(f"{save_labels_path}/{file_name}.txt", 'w+') as file:
        x = ((box[0] + box[2]) / 2) / frame.shape[1]
        y = ((box[1] + box[3]) / 2) / frame.shape[0]
        width = (max(box[2], box[0]) - min(box[2], box[0])) / frame.shape[1]
        height = (max(box[1], box[3]) - min(box[1], box[3])) / frame.shape[0]

        file.write(f"{class_id} {x:.6f} {y:.6f} {width:.6f} {height:.6f}")


def init_needed_folders_if_not_exist(paths: List[str]):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


if __name__ == '__main__':
    main()
