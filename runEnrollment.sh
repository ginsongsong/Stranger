#!/bin/bash
#openblas set default thread


#compiler with AgeEstimation
ER_FILE="STRANGER/Enrollment.cpp"
U_FILE="library/Utility/float2str.cpp library/Utility/int2str.cpp "
FD_FILE="library/FaceDetection/FaceDetection.cpp "


#Compiler library work place
WORK_DIR="/home/gin/WORK"
CV_INC="$WORK_DIR/cv3/include"
CV_LIB="$WORK_DIR/cv3/lib"
CV_PKG="-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_calib3d -lopencv_imgcodecs -lopencv_contrib -lopencv_features2d -lopencv_flann -lopencv_gpu -lopencv_legacy -lopencv_ml -lopencv_objdetect -lopencv_photo -lopencv_stitching -lopencv_superres -lopencv_video -lopencv_videostab -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_videoio" 


OUTPUT="ERM.out"

g++  -w -O2 -O3 -O1 -Os -o $OUTPUT -export-dynamic $ER_FILE $U_FILE $FD_FILE -I$CV_INC -L$CV_LIB $CV_PKG


