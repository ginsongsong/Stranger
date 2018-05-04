#ifndef FACEDETECTION_H
#define FACEDETECTION_H

// OpenCV 3
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/objdetect/objdetect_c.h>

namespace DV {

void faceDetection( const cv::Mat &Src, std::vector<cv::Rect> &faceRegions );

} // end namespace DV

#endif //FACEDETECTION_H
