import argparse
import copy
import os
import uuid
import cv2
from facenet_pytorch.models.mtcnn import MTCNN
from classes.camera_handler import CameraHandler
from definitions import ROOT_DIR


def config_parser():
    parser = argparse.ArgumentParser(
        prog='Face collector',
        description='',)
    parser.add_argument('-pics', '--pics', required=True, type=int, help='number of picture to capture')
    parser.add_argument('-id', '--id', required=True, type=int, help='class id')


    return parser

def main():
    parser = config_parser()
    args = vars(parser.parse_args())
    num_of_pics = args.get("pics", 500)
    class_id = args.get("id", -1)


    images_path = os.path.join(ROOT_DIR, 'data', 'images')
    labels_path = os.path.join(ROOT_DIR, 'data', 'labels')

    if not os.path.exists(images_path):
        os.makedirs(images_path)
    if not os.path.exists(labels_path):
        os.makedirs(labels_path)

    counter = 0
    camera = CameraHandler()
    mtcnn = MTCNN(keep_all=True)
    camera.start()

    while counter < num_of_pics:
        frame = copy.deepcopy(camera.frame)
        boxes, prob = mtcnn.detect(frame)
        if boxes is None:
            continue
        # boxes = np.array(boxes).astype(int)
        for box in boxes:
            file_name = uuid.uuid4()
            # save pic to file
            cv2.imwrite(f"{images_path}/{file_name}.jpg", frame)
            # save config file to yolo
            with open(f"{labels_path}/{file_name}.txt", 'w+') as file:
                x = ((box[0]+box[2])/2)/frame.shape[1]
                y = ((box[1]+box[3])/2)/frame.shape[0]
                width = (max(box[2], box[0]) - min(box[2], box[0]))/frame.shape[1]
                height = (max(box[1], box[3]) - min(box[1], box[3]))/frame.shape[0]

                file.write(f"{class_id} {x:.6f} {y:.6f} {width:.6f} {height:.6f}")
        counter += 1
    camera.stop()

if __name__ == '__main__':
    # print(ROOT_DIR)
    main()