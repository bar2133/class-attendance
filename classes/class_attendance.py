import copy
import logging
import os.path

import cv2
import numpy as np
import torch as torch
from torch import device

from classes.camera_handler import CameraHandler
from definitions import ROOT_DIR


class ClassAttendance:
    DEFAULT_MODEL = os.path.join(ROOT_DIR, 'models', 'faces.pt')

    def __init__(self, custom_data_model: str = None, force_reload: bool = False, logger_level=logging.INFO):
        """

        :param custom_data_model: a path to the custom data model.
        :param force_reload: force reload the data model.
        :param logger_level: the logging level of the program.
        """
        # logger
        self.__set_logger_level(logger_level)
        self.__logger = logging.getLogger(self.__class__.__name__)
        # processing device (gpu device if available)
        self.__device = self.__get_device()
        # data model
        self.__custom_data_model_path = custom_data_model
        self.__data_model = self.__init_data_model(force_reload)
        # camera
        self.__camera = CameraHandler()
        # store
        self.__timer = 0
        self.__frame = None

    def __get_device(self) -> device:
        """ Getting GPU Device if available else CPU device

        :return: device
        """
        # connect to cuda device
        if torch.cuda.is_available():
            self.__logger.debug('cuda GPU is available !')
            return torch.device('cuda')
        # TODO1 - connect to Apple m1 GPU device doesn't fully implemented yet.
        # elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        #     self.__logger.debug('Apple silicon GPU is available !')
        #     return torch.device("mps")
        else:
            self.__logger.warning("could not connect to GPU device, works on CPU")
            return torch.device("cpu")

    def __init_data_model(self, force_reload: bool = False):
        return torch.hub.load(
            model='custom',
            source="local",
            repo_or_dir=os.path.join(ROOT_DIR, "yolov5"),
            path=self.__custom_data_model_path if self.__custom_data_model_path else self.DEFAULT_MODEL,
            force_reload=force_reload,
            device=self.__device)

    @staticmethod
    def __set_logger_level(logger_level) -> None:
        logging.basicConfig(level=logger_level)

    def start(self) -> None:
        """ Starts the camera frames gattering in a new thread
        and every RELOAD_TIME sending the frame to the model for face detection.

        :return: None
        """
        self.__camera.start()
        while 1:
            self.__frame = copy.deepcopy(self.__camera.frame)
            try:
                self.__classification(self.__frame)
            except Exception as e:
                print(e)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        self.__camera.stop()

    def __classification(self, frame) -> None:
        """ recognize the face in the picture, and draw the classification on the frame useing cv2. """
        result = self.__data_model(frame)
        new_img = result.render()
        if new_img:
            self.__last_frame = new_img
        frame = np.squeeze(self.__last_frame)
        self.__add_detections_on_frame(frame, result)
        cv2.imshow("", frame)

    @staticmethod
    def __add_detections_on_frame(frame, result):
        cv2.putText(frame, result.print(classes=True),
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    1,
                    2)
        # adding quit instruction.
        cv2.putText(frame, "Press Q to quit",
                    (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    1,
                    2)


if __name__ == '__main__':
    custom_model_path = os.path.join(ROOT_DIR, 'yolov5', 'runs', 'train', 'best.pt')
    clas = ClassAttendance(logger_level=logging.DEBUG,
                           custom_data_model=custom_model_path,
                           force_reload=True)
    clas.start()
