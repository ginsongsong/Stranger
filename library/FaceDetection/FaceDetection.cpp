#include "FaceDetection.h"

namespace DV {

//--------------------------------------------------------------------------
void faceDetection( const cv::Mat &Src, std::vector<cv::Rect> &faceRegions )
{
   cv::CascadeClassifier face_cascade;

   // Alternative use
   const std::string face_cascade_name = "./Models/FaceDetection/haarcascades/haarcascade_frontalface_alt2.xml";
   //const std::string face_cascade_name = "../Models/FaceDetection/haarcascades/haarcascade_frontalface_default.xml";
   //const std::string face_cascade_name = "../Models/FaceDetection/lbpcascades/lbpcascade_frontalface.xml";

   if( !face_cascade.load( face_cascade_name ) )
   {
      std::cout << "[Error] Couldn't load face cascade!!\n";
      return;
   }

   //std::vector<cv::Rect> faces;
   cv::Mat SrcImg_gray;

   cvtColor( Src, SrcImg_gray, CV_BGR2GRAY );
   //equalizeHist( SrcImg_gray, SrcImg_gray );

   //Detect faces
   face_cascade.detectMultiScale( SrcImg_gray, faceRegions, 1.1, 3, 0 | CV_HAAR_SCALE_IMAGE, cv::Size( 50, 50 ) );

   // Center face ( Close to center on Horizontal axis. )
   //int max = std::numeric_limits<int>::max();
   //int index = -1;
   //for( size_t i = 0; i < faceRegions.size(); i++ )
   //{
   //   int tmp = ( faces[ i ].x + faces[ i ].width / 2 ) - ( Src.cols / 2 );
   //   if( abs( tmp ) < max )
   //   {
   //      index = i;
   //      max = tmp;
   //   }
   //}//end for i
   //if( Nface !=0 )
   //    faceRegion = Rect( faces[ index ].x, faces[ index ].y, faces[ index ].width, faces[ index ].height );
} //end void faceDetection( const Mat SrcImg, Rect &faceRegion )
//-------------------------------------------------------------------------

} //end namespace DV {
