
#!/bin/bash
#openblas set default thread



#compiler with AgeEstimation
ED_FILE="STRANGER/EnrollmentDatabase.cpp"
U_FILE="library/Utility/float2str.cpp library/Utility/int2str.cpp "
FD_FILE="library/FaceDetection/FaceDetection.cpp "
FR_FILE="library/FaceRecognition/CFaceRec.cpp "
FILE="$ED_FILE $U_FILE $FD_FILE $FR_FILE"


#Compiler library work place
WORK_DIR="/home/gin/WORK"
#arm library option
#if environment is TX1
ARM_LIB="" #"-L/usr/lib/aarch64-linux-gnu/"
#is x86
#ARM_LIB=" "
#caffe include lib
CAFFE_INCLUDE="$WORK_DIR/caffe/include"
CAFFE_LIB="$WORK_DIR/caffe/build/lib"
CAFFE_3PARTY="$WORK_DIR/caffe/3rdparty"
#opencv include
CV_INCLUDE="$WORK_DIR/cv3/include"
CV_LIB="$WORK_DIR/cv3/lib"
#CV3 /usr/lib/pkgconfig/opencv3.pc
CV_PKG="-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_calib3d -lopencv_imgcodecs -lopencv_contrib -lopencv_features2d -lopencv_flann -lopencv_gpu -lopencv_legacy -lopencv_ml -lopencv_objdetect -lopencv_photo -lopencv_stitching -lopencv_superres -lopencv_video -lopencv_videostab -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_videoio" 
#"`pkg-config opencv3 --cflags --libs`"
#default opencv configuration file is opencv.pc
#CV_PKG="`pkg-config opencv --cflags --libs`"
CUDA_PATH="/usr/local/cuda"

CUDA_INC=$CUDA_PATH/include
CUDA_LIB=$CUDA_PATH/lib64

CUDA="-I $CUDA_INC -L $CUDA_LIB -lcudart"

LIB_INCLUDE="-I/usr/local/include  -I $CAFFE_3PARTY  -L/usr/local/lib/ $ARM_LIB  -I$CAFFE_INCLUDE -I$CV_INCLUDE -L$CV_LIB $CV_PKG  -L$CAFFE_LIB -lboost_system  $CUDA  -lcaffe -lglog -lprotobuf"


OUTPUT="ERD.out"

g++  -w -O2 -O3 -O1 -Os -o $OUTPUT -export-dynamic $FILE  $LIB_INCLUDE
