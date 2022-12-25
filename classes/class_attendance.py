import copy
import logging
import random
from typing import Tuple

import numpy as np
from datetime import datetime
import uuid
import cv2
import torch as torch
from facenet_pytorch.models.mtcnn import MTCNN

from classes.camera_handler import CameraHandler


class ClassAttendance:
    RELOAD_TIME = 0.5
    THICKNESS = 2
    MARGIN = -5
    def __init__(self, gpu=True, data_model=None, fps=False, logger_level=logging.INFO):
        """

        :param gpu: use gpu or not
        :param data_model: custom trained data_model
        """
        self.__set_logger_level(logger_level)
        self.__logger = logging.getLogger(self.__class__.__name__)
        self.__data_model = data_model
        self.__gpu: bool = gpu
        self.__gpu_device = None
        self.__camera = CameraHandler()
        if self.__gpu:
            self.__connect_to_gpu()
        self.mtcnn = MTCNN(keep_all=True) if not self.__gpu_device else MTCNN(device=self.__gpu_device, keep_all=True)
        self.__last_boxes = None
        self.__timer = 0
        self.__class_color = {'unknown': self.__random_color()}
        self.__recognized_classes: set = set()

    def __connect_to_gpu(self) -> None:
        """ Getting GPU Device

        :return: None
        """
        # connect to cuda device
        if torch.cuda.is_available():
            self.__logger.debug('Connecting to cuda GPU !')
            self.__gpu_device = torch.device('cuda')
            return
        else:
            self.__logger.debug('cuda GPU is not available')
        # TODO - Apple m1 GPU doesnt have implemented support yet. required code below
        # # connect to Apple m1 GPU device
        # if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        #     self.logger.debug('Apple m1 GPU is available !')
        #     self.gpu_device = torch.device("mps")
        # else:
        #     self.logger.debug('Apple GPU is not available')

    @staticmethod
    def __set_logger_level(logger_level):
        logging.basicConfig(level=logger_level)

    def start(self):
        """ Starts the camera frames gattering in a new thread
        and every RELOAD_TIME sending the frame to the MTCNN pre-trained network for generally face detection.
        the cv2 draw the returned boxes on the frames.

        :return: None
        """
        self.__camera.start()
        self.__timer = datetime.now()
        while 1:
            frame = copy.deepcopy(self.__camera.frame)
            new_time = datetime.now()
            if (new_time - self.__timer).seconds > ClassAttendance.RELOAD_TIME:
                self.__timer = new_time
                self.__get_new_boxes(frame)

            self.__draw_boxes(frame)
            cv2.imshow('', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        self.__camera.stop()

    def __get_new_boxes(self, frame) -> None:
        """ sending the frame to the mtcnn and gets the new boxes arrays.

        :param frame: frame to work on
        :return: None
        """
        boxes, prob = self.mtcnn.detect(frame)
        if boxes is not None:
            boxes = np.array(boxes).astype(int)

        self.__last_boxes = boxes

    def __draw_boxes(self, frame) -> None:
        if self.__last_boxes is None:
            return
        for box in self.__last_boxes:
            # cv2.imwrite('pics/'+uuid.uuid4().__str__()+'.jpg', frame[box[1]:box[0], box[3]:box[2]])
            classification: str = self.__classification(box)
            self._add_class_to_set(classification)
            self.__set_box_label(box, classification, frame)
            self.__draw_rectangle(box, classification, frame)

    def __set_box_label(self, box, classification, frame):
        cv2.putText(frame, classification, (box[0], box[1] + self.MARGIN), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    self.__get_cls_color(classification), self.THICKNESS, cv2.LINE_AA)

    def __draw_rectangle(self, box, classification, frame) -> None:
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]),
                      self.__get_cls_color(classification), self.THICKNESS)

    def __get_cls_color(self, classification: str):
        return self.__class_color.get(classification, self.__random_color())

    def __random_color(self) -> Tuple[int, int, int]:
        return random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)

    def __classification(self, box) -> str:
        """ recognize the face in the picture and returns the name

        :return: string -> face name.
        """
        cls = 'unknown'
        if cls != 'unknown' and cls not in self.__class_color:
            self.__class_color[cls] = self.__random_color()
        return cls

    def _add_class_to_set(self, classification: str) -> None:
        self.__recognized_classes.add(classification)


if __name__ == '__main__':
    # attendance = ClassAttendance(logger_level=logging.DEBUG)
    clas = ClassAttendance(logger_level=logging.DEBUG)
    clas.start()

