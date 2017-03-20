#ifndef _CHAR_DETECTOR_
#define _CHAR_DETECTOR_
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <utility>
#include <string>
#include <vector>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include "TextCore.h"

namespace caffe {
struct Box{
    float x;
    float y;
    float w;
    float h;
    float score;
    Box(){
        x = 0.0f;
        y = 0.0f;
        w = 0.0f;
        h = 0.0f;
        score = 0.0f;
    }
};

float iou(Box& b1, Box& b2);
void nms(std::vector<Box>& boxes, std::vector<Box>& out, float thresh);
void TransBox2TextChar(const std::vector<Box>& in, std::vector<text::TextChar>& out);

class Detector {
 public:
  //copy weights from trained model
  Detector(const string& model_file,
           const string& weights_file,
           const int gpu_id);
  //copy weights from other inited net variable
  Detector(const string& model_file,
           const caffe::Net<float>* other_net,
           const int gpu_id);
  ~Detector() {
      if (NULL != net_) {
          delete net_;
          net_ = NULL;
      }
  }
  //std::vector<vector<float> > Detect(const cv::Mat& img);
  void Detect(const cv::Mat& img, std::vector<Box>& final_dets);
  caffe::Net<float>* get_net() {
      return net_;
  }

 private:
//  void SetMean();
  void WrapInputLayer(std::vector<cv::Mat>* input_channels, const cv::Mat& img);
  void Preprocess(const cv::Mat& img, cv::Mat& out_img);
  void retrieve_bboxes(const shared_ptr<Blob<float> >& rois_blob,
                       const Blob<float>* deltas_blob,
                       const Blob<float>* scores_blob,
                       std::vector<float>& out_boxes,
                       std::vector<float>& out_scores);

 private:
  caffe::Net<float>* net_;
  cv::Size input_geometry_;
  int num_channels_;
  unsigned int target_size_;
  unsigned int max_size_;
  cv::Scalar mean_;
  float image_scale_;
};

}//end of namespace caffe
#endif
