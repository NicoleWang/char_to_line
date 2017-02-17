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
 /*
struct DetectModel{
    DetectModel(const std::string& proto_path, const std::string& model_path) {
        m_proto_path = proto_path;
        m_model_path = model_path;
    }
    std::string m_proto_path;
    std::string m_model_path;
};
*/

caffe::Detector* Init(const std::string& proto_path, const std::string& model_path);
void Release(caffe::Detector*& detector);
int GetTextLine(const cv::Mat& img,
                caffe::Detector* detector,
                std::vector<cv::Mat>& LineImgs, 
                std::vector<cv::Mat>& LineBins,
                std::vector<cv::Point>& offsetPts);
int GetTextLine(const cv::Mat& img,
                caffe::Detector* detector,
                std::vector<cv::Mat>& LineImgs, 
                std::vector<cv::Mat>& LineBins,
                std::vector<cv::Point>& offsetPts,
                const std::string& save_dir,
                const std::string& image_name);
}//end of namespace

#endif
