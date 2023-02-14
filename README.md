# class-attendance

This project simulate online class attendance software via webcam.

<h2>usage:</h2>
by default the project use MTCNN for generic recognition of faces on frames form the camera.<br>
there is an option to use custom data model with program by passing a path for .pt file of yolov5  custom data model.<br> 

<h2>faces_collector.py: </h2>
This script collects pictures of faces with MTCNN for train custom model of yolov5. <br>
the script saves the pictures and the labels under data folder in root directory.<br>

this script receives two arguments:<br>
-i: the id of the class for the labeling under the dataset <br>
-p: the amount of pictures to collect

<h2>class_attendance.py: </h2>
for using the custom model of yolov5, change the value of the variable custom_model_path under the main section.
after that you can run the program.<br>
for using the program with the default model of MTCNN pass None under custom_model_path.

<h3>Training after collecting pictures</h3>
After using the tool, open yolov5/data/coco128.yml and edit the classes names according to the ids that collected before<br>
Example to taing command:<br>
python train.py --img 640 --batch 16 --epochs 500 --data coco128.yaml --weights yolov5s.pt --cache --workers=4<br>
find more information about this command in yolo repository on github <br>


created by: Dana Atias and Bar Nachlieli

<h3>Resources</h3>
<