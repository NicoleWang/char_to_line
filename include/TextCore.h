/*
 * Brief: Basic data structure for text line generation
 * Author: wangyuzhuo@bytedance.com
 * Date: 2016/12/08 20:35
 */
#ifndef _TEXTCORE_H_
#define _TEXTCORE_H_

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

namespace text {

template<class Dtype>
struct Box{
    Dtype left;
    Dtype top;
    Dtype width;
    Dtype height;
    Dtype right;
    Dtype bottom;
};

class TextChar {
    TextChar();
    TextChar(const std::string& line, bool has_score);
    ~TextChar();
    public:
    float get_area();
    cv::Point2f get_center();
    float get_intersection(const Box<float>& other);
    float get_union(const Box<float>& other);

    Box<int> m_ibox;
    Box<float> m_fbox;
    cv::Point2f m_center;
    float score;
};
}//end of namespace text
#endif



