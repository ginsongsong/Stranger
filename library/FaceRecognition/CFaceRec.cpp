#include "CFaceRec.h"

#ifdef DV_USE_CAFFE
// caffe
#include "caffe/caffe.hpp" 
#endif

// C++ STL
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>

namespace DV {

//--------------------------------------------------------------------------
CFaceRec::CFaceRec(
    std::string f_model_def_path,
    std::string f_model_wei_path )
{
    m_model_def = f_model_def_path;
    m_model_wei = f_model_wei_path;
    m_fea_layer = "fc7";
    init( m_model_def, m_model_wei );
} // CFaceRec( std::string f_model_def, std::string f_model_wei )
//--------------------------------------------------------------------------

//--------------------------------------------------------------------------
CFaceRec::~CFaceRec()
{
    //dtor
#ifdef DV_USE_CAFFE
    free( m_caffe_FRNet );
    m_caffe_FRNet = NULL;
#endif
} // ~CFaceRec()
//--------------------------------------------------------------------------

//--------------------------------------------------------------------------
void CFaceRec::setModel_def( std::string model_def )
{
    m_model_def = model_def;
} // CFaceRec::setModel_def( std::string f_model_def )
//--------------------------------------------------------------------------

//--------------------------------------------------------------------------
void CFaceRec::setModel_wei( std::string model_wei )
{
    m_model_wei = model_wei;
} // CFaceRec::setModel_wei( std::string f_model_wei )
//--------------------------------------------------------------------------

//-------------------------------------------------------------------------
void CFaceRec::subtactMeans( cv::Mat &img )
{
    cv::MatIterator_<cv::Vec3b> it, end;
    const float avgImg[] = { 129.1863, 104.7624, 93.5940 }; // RGB
    for( it = img.begin<cv::Vec3b>(), end = img.end<cv::Vec3b>(); it != end; ++it )
    {
        ( *it )[ 0 ] -= avgImg[ 2 ];    // B
        ( *it )[ 1 ] -= avgImg[ 1 ];    // G
        ( *it )[ 2 ] -= avgImg[ 0 ];    // R
    } // end for it
} // CFaceRec::subtactMeans( cv::Mat &Face )
//-------------------------------------------------------------------------

//-------------------------------------------------------------------------
void CFaceRec::imgResize(
    const cv::Mat &src,
    cv::Mat &dst,
    int img_W,
    int img_H )
{
    resize( src, dst, cv::Size( img_W, img_H ), cv::INTER_CUBIC );
} // CFaceRec::normalization( ... )
//------------------------------------------------------------------------

//------------------------------------------------------------------------
void CFaceRec::init(
    std::string model_def,
    std::string model_wei )
{
#ifdef DV_USE_CAFFE
   /// Load model
    CHECK_GT( model_def.size(), 0 ) << "Need a model definition to face recognition.";
    CHECK_GT( model_wei.size(), 0 ) << "Need model weights to face recognition.";

    // set mode
#ifdef CPU_ONLY
    caffe::Caffe::set_mode( caffe::Caffe::CPU );
#else
    caffe::Caffe::set_mode( caffe::Caffe::GPU );
#endif

    // instantiate the Caffe net
    m_caffe_FRNet = new caffe::Net<float>( model_def, caffe::TEST );
    m_caffe_FRNet->CopyTrainedLayersFrom( model_wei );
    CHECK_EQ( m_caffe_FRNet->num_inputs(), 1 ) << "Network should have exactly one input.";
    CHECK_EQ( m_caffe_FRNet->num_outputs(), 1 ) << "Network should have exactly one output.";

    // get image size from model definition
    int Nchannels = m_caffe_FRNet->input_blobs()[ 0 ]->channels();
    CHECK( Nchannels == 3 || Nchannels == 1 ) << "Input layer should have 1 or 3 channels.";
    m_img_W = m_caffe_FRNet->input_blobs()[ 0 ]->width();
    m_img_H = m_caffe_FRNet->input_blobs()[ 0 ]->height();
 
    // reshape model
    m_caffe_FRNet->input_blobs()[ 0 ]->Reshape( 1, Nchannels, m_img_H, m_img_W );
    m_caffe_FRNet->Reshape();

    // initialize input vector
    float *input_data = m_caffe_FRNet->input_blobs()[ 0 ]->mutable_cpu_data();
    for( int i = 0; i < m_caffe_FRNet->input_blobs()[ 0 ]->channels(); ++i ) 
    {
        cv::Mat channel( m_img_H, m_img_W, CV_32FC1, input_data );
        m_input_channels.push_back( channel );
        input_data += m_img_W * m_img_H;
    } // for i

    // feature ID
    int fea_layer_ID = 0;
    for( int i = 1; i < m_caffe_FRNet->layers().size(); i++ )
    {
        if( m_fea_layer.compare( m_caffe_FRNet->layer_names()[ i ] ) == 0 )
            fea_layer_ID = i;
    } // for i
    
    m_Nfeatures = m_caffe_FRNet->top_vecs()[ fea_layer_ID ][ 0 ]->count() / m_caffe_FRNet->top_vecs()[ fea_layer_ID ][ 0 ]->num();
#endif
} // void init(...)
//------------------------------------------------------------------------

//------------------------------------------------------------------------
int CFaceRec::getFeatureLength()
{
#ifdef DV_USE_CAFFE
   if( m_caffe_FRNet != NULL )
        return m_Nfeatures;
#else
   return 0;
#endif
} // int CFaceRec::getFeatureLength()
//------------------------------------------------------------------------

//------------------------------------------------------------------------
void CFaceRec::img2model( 
    const cv::Mat &srcImg,                      // [Input] source image
    std::vector<cv::Mat> *input_channels )      // [Output] Input vector 
{
	  //printf("ok777\n");
    // convert the input image to the input image format of the network. 
    cv::Mat sample;
    if( srcImg.channels() == 4 )
        cv::cvtColor( srcImg, sample, CV_BGRA2BGR );
    else if( srcImg.channels() == 1 )
        cv::cvtColor( srcImg, sample, CV_GRAY2BGR );
    else
        sample = srcImg;
	  //printf("ok7777\n");
	  
    cv::Mat sample_resized;
    cv::Size input_size( m_img_W, m_img_H );
    if( sample.size() != input_size )
        cv::resize( sample, sample_resized, cv::Size(m_img_W, m_img_H) );
    else
        sample_resized = sample;
	//printf("ok77777\n");
    cv::Mat sample_float;
    sample_resized.convertTo( sample_float, CV_32FC3 );

    cv::Mat sample_normalized = sample_float;
    subtactMeans( sample_normalized );

    cv::split( sample_float, *input_channels );

#ifdef DV_USE_CAFFE
    CHECK( reinterpret_cast<float*>( input_channels->at( 0 ).data )
        == m_caffe_FRNet->input_blobs()[ 0 ]->cpu_data() )
        << "Input channels are not wrapping the input layer of the network.";
#endif
} // void CFaceRec::img2model(...)
//------------------------------------------------------------------------

//------------------------------------------------------------------------
void CFaceRec::exeFeatureExt(               // feature extraction
    const cv::Mat &srcFace,                 // [Input] face image
    std::vector<float> &features )          // [Output] feature vector
{
    // input vector
    //printf("ok77\n");
    img2model( srcFace, &m_input_channels );
	//printf("ok8\n");
     
#ifdef DV_USE_CAFFE
    //m_caffe_FRNet->ForwardPrefilled();
    //
 // initialize input vector
    float *input_data = m_caffe_FRNet->input_blobs()[ 0 ]->mutable_cpu_data();
    for( int i = 0; i < m_caffe_FRNet->input_blobs()[ 0 ]->channels(); ++i ) 
    {
        cv::Mat channel( m_img_H, m_img_W, CV_32FC1, input_data );
        m_input_channels.push_back( channel );
        input_data += m_img_W * m_img_H;
    } // for i
    //  
    //printf("ok9\n");
       
    m_caffe_FRNet->ForwardFromTo( 0, m_caffe_FRNet->layers().size() -1);

    // Features
    //const boost::shared_ptr<caffe::Blob<float> > feature_blob = m_caffe_FRNet->blob_by_name( m_fea_layer ); // optional: it can extract features from a specific layer.
    long double squre_sum = 0.0;
    const caffe::Blob<float>* feature_blob = m_caffe_FRNet->output_blobs()[ 0 ];
    const float *blob_data = feature_blob->cpu_data();
    int fea_idx = 0;
    for( int i_n = 0; i_n < feature_blob->num(); i_n++ )
    {
        for( int i_c = 0; i_c < feature_blob->channels(); i_c++ )
        {
            for( int i_h = 0; i_h < feature_blob->height(); i_h++ )
            {
                for( int i_w = 0; i_w < feature_blob->width(); i_w++ )
                {
                    features[ fea_idx ] = *( blob_data + feature_blob->offset( i_n, i_c, i_h, i_w ) );
                    squre_sum += ( features[ fea_idx ] * features[ fea_idx ]);
                    fea_idx++;
                } // for i_w
            } // for i_h
        } // for i_c
    }// for i_n
    // L2-norm normalization
    squre_sum = std::sqrt( squre_sum );
    for( int i = 0; i < features.size(); i++ )
        features[ i ] /= squre_sum;
#endif
} // void exeFeatureExt(...)
//------------------------------------------------------------------------

//------------------------------------------------------------------------
void CFaceRec::exeFaceRec(                              // Face recognition based on k-nearest neighbors
    const cv::Mat &srcFace,                             // [Input] query face image
    const std::vector<int> &IDList,                     // [Input] ID list for each feature (e.g. 1 1 2 2 2 3 3 4 5) 
    const std::vector<std::vector<float> > &feaMatrix,  // [Input] feature matrix, size: length of IDList * length of feature
    const int k_knn,                                    // [Input] k of k-nearest neighbors algorithm
    const float thd,                                    // [Input] a threshold for determining unknown face. Note: it can be obtained by function getUnknowThd. The higher value could accept more unknown faces.
    int &Id,                                            // [Output] identity. If Id is -1, the face is unknown(stranger). 
    float &confidence )                                 // [Output] confidence ~[0,1]. 
{
    // extract features
    std::vector<float> fea_query( m_Nfeatures );
    exeFeatureExt( srcFace, fea_query );

    // compute distances, range~[0,feature size]
    double *distance = new double[ IDList.size() ];
    long double tmp_sum;
    int *idx_sorted = new int[ IDList.size() ];
    for( int i = 0; i < IDList.size(); i++ )
    {
        tmp_sum = 0.0;
        for( int j = 0; j < m_Nfeatures; j++ )
        {
            tmp_sum += ( fea_query[ j ] - feaMatrix[ i ][ j ] ) * ( fea_query[ j ] - feaMatrix[ i ][ j ] );
        } // for j
        distance[ i ] = std::sqrt( tmp_sum );
        idx_sorted[ i ] = i;
    } // for i

    // sorting
    quickSort( distance, idx_sorted, 0, IDList.size() - 1 );

    // unknown face
    if( distance[ 0 ] > thd )
    {
        Id = -1;
        delete[] distance;
        delete[] idx_sorted;
        return;
    } // if

    // kNN (optional)
    //std::vector<int> member_Topk;
    //std::vector<int> counting;
    //member_Topk.push_back( IDList[ idx_sorted[ 0 ] ] );
    //counting.push_back( 1 );
    //for( int i = 1; i < k_knn; i++ )
    //{
    //    bool isNew = true;
    //    for( int j = 0; j < member_Topk.size(); j++ )
    //    {
    //        if( member_Topk[ j ] == IDList[ idx_sorted[ i ] ] )
    //        {
    //            isNew = false;
    //            counting[ j ]++;
    //            break;
    //        } // if
    //    } // for j
    //    if( isNew )
    //    {
    //        member_Topk.push_back( IDList[ idx_sorted[ i ] ] );
    //        counting.push_back( 1 );
    //    } // if     
    //} // for i

    // weighted kNN
    std::vector<int> member_Topk;
    std::vector<double> counting;
    member_Topk.push_back( IDList[ idx_sorted[ 0 ] ] );
    counting.push_back( wei_knn( distance[ 0 ] ) );

    for( int i = 1; i < k_knn; i++ )
    {
        bool isNew = true;
        for( int j = 0; j < member_Topk.size(); j++ )
        {
            if( member_Topk[ j ] == IDList[ idx_sorted[ i ] ] )
            {
                isNew = false;
                counting[ j ] += wei_knn( distance[ i ] );
                break;
            } // if
        } // for j
        if( isNew )
        {
            member_Topk.push_back( IDList[ idx_sorted[ i ] ] );
            counting.push_back( wei_knn( distance[ i ] ) );
        } // if     
    } // for i

    int idx_max = 0;
    double max = counting[ 0 ];
    double sum = counting[ 0 ];
    for( int i = 1; i < counting.size(); i++ )
    {
        if( counting[ i ] > max )
        {
            idx_max = i;
            max = counting[ i ];
        } // if
        sum += counting[ i ];
    } // for i

    Id = member_Topk[ idx_max ];
    confidence = counting[ idx_max ] / sum;

    delete[] distance;
    delete[] idx_sorted;
} // void CFaceRec::exeFaceRec(...)
//--------------------------------------------------------------------------

//--------------------------------------------------------------------------
void CFaceRec::quickSort(
    double *arr,  //Input & Output
    int *idx,     //Input & Output, 0-index
    int left,     //Input
    int right )   //Input
{
    int i = left, j = right;
    double tmp;
    int tmp_idx;
    double pivot = arr[ ( left + right ) / 2 ];

    // partition
    while( i <= j )
    {
        while( arr[ i ] < pivot )
            i++;
        while( arr[ j ] > pivot )
            j--;
        if( i <= j )
        {
            //data
            tmp = arr[ i ];
            arr[ i ] = arr[ j ];
            arr[ j ] = tmp;
            //index
            tmp_idx = idx[ i ];
            idx[ i ] = idx[ j ];
            idx[ j ] = tmp_idx;

            i++;
            j--;
        } // end if
    } // end while

    // recursion
    if( left < j )
        quickSort( arr, idx, left, j );
    if( i < right )
        quickSort( arr, idx, i, right );
} // void CFaceRec::quickSort(...)
//------------------------------------------------------------------------

//------------------------------------------------------------------------
float CFaceRec::getUnknowThd( 
        const std::vector<int> &IDList,                     // [Input] ID list for each feature (e.g. 1 1 2 2 2 3 3 4 5) 
        const std::vector<std::vector<float> > &feaMatrix,  // [Input] feature matrix, size: length of IDList * length of feature
        int level )                                         // [Input] get a threshold for determining unknown face. The higher level could accept more unknown faces.
{    
    // compute distances
    std::vector<std::vector<float> > disMatrix( IDList.size(), std::vector<float>( IDList.size(), 0.0 ) ); // Upper-triangular matrix
    for( int i = 0; i < IDList.size(); i++ )
    {
        for( int j = 0; j < IDList.size(); j++ )
        {
            if( j <= i )
                continue;
            long double distance = 0.0;
            for( int k = 0; k < m_Nfeatures; k++ )
            {
                distance += ( feaMatrix[ i ][ k ] - feaMatrix[ j ][ k ] ) * ( feaMatrix[ i ][ k ] - feaMatrix[ j ][ k ] );
            } //end for k
            disMatrix[ i ][ j ] = std::sqrt( distance );
        } // for j
    } // for i

    ///  variances
    std::vector<int> member;
    std::vector<int> NregistImg; // the number of registered images, length: Nmember
    member.push_back( IDList[ 0 ] );
    NregistImg.push_back( 1 );
    for( int i = 1; i < IDList.size(); i++ )
    {
        bool isNew = true;
        for( int j = 0; j < member.size(); j++ )
        {
            if( member[ j ] == IDList[ i ] )
            {
                isNew = false;
                NregistImg[ j ]++;
                break;
            } // if
        } // for j
        if( isNew )
        {
            member.push_back( IDList[ i ] );
            NregistImg.push_back( 1 );
        } // if     
    } // for i

    std::vector<float> mean( member.size() );
    for( int i = 0; i < member.size(); i++ )
    {
        long double sum = 0.0;
        for( int j = 0; j < IDList.size(); j++ )
        {
            if( IDList[ j ] != member[ i ] )
                continue;
            for( int k = j + 1; k < IDList.size(); k++ )
            {
                if( IDList[ k ] == member[ i ] )
                    sum += disMatrix[ j ][ k ];
            } // for k
        } // for j
        mean[ i ] = sum / ( ( NregistImg[ i ] * NregistImg[ i ] - NregistImg[ i ] ) / 2.0 );
    } // for i

    std::vector<float> var( member.size() );
    for( int i = 0; i < member.size(); i++ )
    {
        long double sum = 0.0;
        for( int j = 0; j < IDList.size(); j++ )
        {
            if( IDList[ j ] != member[ i ] )
                continue;
            for( int k = j + 1; k < IDList.size(); k++ )
            {
                if( IDList[ k ] == member[ i ] )
                    sum += ( disMatrix[ j ][ k ] - mean[ i ] ) * ( disMatrix[ j ][ k ] - mean[ i ] );
            } // for k
        } // for j
        var[ i ] = std::sqrt( sum / ( ( NregistImg[ i ] * NregistImg[ i ] - NregistImg[ i ] ) / 2.0 ) );
    } // for i

    long double sum_avg = 0.0;
    for( int i = 0; i < member.size(); i++ )
        sum_avg += mean[ i ];
    sum_avg /= member.size();

    long double sum_std = 0.0;
    for( int i = 0; i < member.size(); i++ )
        sum_std += var[ i ];
    sum_std /= member.size();

    return sum_avg + level * sum_std;
} // float CFaceRec::getUnknowThd( int level )
//------------------------------------------------------------------------

//-----------------------------------------------------------------------
float CFaceRec::wei_knn( const float distance )
{
    const float tolerance = 0.00001;
    float dis_square = distance * distance;
    if( dis_square < tolerance )
        return ( 1.0f / tolerance );
    else
        return ( 1.0f / dis_square );
} // float CFaceRec::wei_knn( float distance )
//-----------------------------------------------------------------------

} // namespace DV
