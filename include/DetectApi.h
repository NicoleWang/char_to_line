/*
 * Brief: Detect Api  for text line detection
 * Author: wangyuzhuo@bytedance.com
 * Date: 2016/1/08 15:28
 */

#ifndef _DETECTAPI_H_
#define _DETECTAPI_H_

 #include <opencv2/opencv.hpp>
//#include "char_detector.hpp"
namespace caffe {
    class Detector;
}
//using namespace caffe;
namespace text{
struct DetectParams{
    DetectParams() {
        m_device_id = 0;
    }
    DetectParams(int device_id) {
        m_device_id = device_id;
    }
    int m_device_id;
};

caffe::Detector* Init(const std::string& proto_path, const std::string& model_path, const DetectParams& detParams);
void Release(caffe::Detector*& detector);
int GetTextLine(const cv::Mat& img,
                caffe::Detector* detector,
                std::vector<cv::Mat>& LineImgs, 
                std::vector<cv::Mat>& LineBins,
                std::vector<cv::Point>& offsetPts, 
                std::vector< std::vector<cv::Rect> >& char_pos);
int GetTextLine(const cv::Mat& img,
                caffe::Detector* detector,
                std::vector<cv::Mat>& LineImgs, 
                std::vector<cv::Mat>& LineBins,
                std::vector<cv::Point>& offsetPts,
                const std::string& save_dir,
                const std::string& image_name);
int GetTextLine(const cv::Mat& img,
                std::vector<cv::Mat>& LineImgs, 
                std::vector<cv::Mat>& LineBins,
                std::vector<cv::Point>& offsetPts,
                const std::string& chars_dir,
                const std::string& save_dir,
                const std::string& image_name);


}//end of namespace

#endif
