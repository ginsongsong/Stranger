#ifndef CFACEREC_H
#define CFACEREC_H

// STL
#include <string>
#include <vector>

// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// DeepVision
#include "../config.h"

#ifdef DV_USE_CAFFE
// caffe
#include "caffe/caffe.hpp" 
#endif

namespace DV {

class CFaceRec
{
public:

    CFaceRec(
        std::string f_model_def_path,   // model definition
        std::string f_model_wei_path ); // model parameters(weights)
    virtual ~CFaceRec();
    
    int getFeatureLength();           // get the length of feature vector

    float getUnknowThd(                                     // Note: if the class is initialized or registered images have changed, please update the threshold using this function as the input argument of function exeFaceRec().
        const std::vector<int> &IDList,                     // [Input] ID list for each feature (e.g. 1 1 2 2 2 3 3 4 5) 
        const std::vector<std::vector<float> > &feaMatrix,  // [Input] feature matrix, size: length of IDList * length of feature
        int level = 2 );                                    // [Input] get a threshold for determining unknown face. The higher level could accept more unknown faces.

    // execution
    void exeFeatureExt(                                     // feature extraction
        const cv::Mat &srcFace,                             // [Input] face image
        std::vector<float> &features );                     // [Output] feature vector

    void exeFaceRec(                                        // Face recognition based on k-nearest neighbors
        const cv::Mat &srcFace,                             // [Input] query face image
        const std::vector<int> &IDList,                     // [Input] ID list for each feature (e.g. 1 1 2 2 2 3 3 4 5) 
        const std::vector<std::vector<float> > &feaMatrix,  // [Input] feature matrix, size: length of IDList * length of feature
        const int k_knn,                                    // [Input] k of k-nearest neighbors algorithm
        const float thd,                                    // [Input] a threshold for determining unknown face. Note: it can be obtained by function getUnknowThd. The higher value could accept more unknown faces.
        int &Id,                                            // [Output] identity. If Id is -1, the face is unknown(stranger). 
        float &confidence );                                // [Output] confidence ~[0,1].

private:
    
    std::string m_fea_layer;                      // name of feature layer
    int m_fea_layer_ID;                           // feature layer ID

    std::string m_model_def;                      // definition path
    std::string m_model_wei;                      // parameters path

    // set parameters
    void setModel_def( std::string model_def );
    void setModel_wei( std::string model_wei );

    void init(                                    // initialize model
        std::string model_def,
        std::string model_wei );    

    void img2model(                               // image to input of model
        const cv::Mat &srcImg,                    // [Input] source image
        std::vector<cv::Mat> *input_channels );   // [Output] Input vector              
         

    void subtactMeans( cv::Mat &img );            // subtract RGB means from original image 

    void imgResize(                               // image size is based on the definition of input layer
        const cv::Mat &src,
        cv::Mat &dst,
        int img_W,
        int img_H );

#ifdef DV_USE_CAFFE
    caffe::Net<float> *m_caffe_FRNet;
#endif

    cv::Mat m_faceImg;                            // a face image
    int m_Nfeatures;                              // # features

    std::vector<cv::Mat> m_input_channels;        // input vector for model

    int m_img_H;        // image height
    int m_img_W;        // image width

    void quickSort(
        double arr[],   //Input & Output
        int idx[],      //Input & Output, 0-index
        int left,       //Input
        int right );    //Input

    float wei_knn( const float distance );        // get a weight of sample

}; // class CFaceRec

} // namespace DV

#endif // CFACEREC_H
