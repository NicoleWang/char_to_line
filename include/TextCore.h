/*
 * Brief: Basic data structure for text line generation
 * Author: wangyuzhuo@bytedance.com
 * Date: 2016/12/08 20:35
 */
#ifndef _TEXTCORE_H_
#define _TEXTCORE_H_

//#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

namespace text {

class TextChar {
    public:
    TextChar();
    TextChar(const std::string& line, bool has_score);
    TextChar(const cv::Rect& rect, const float score);
    ~TextChar();
    inline void get_center();
    inline float get_area();
    inline float get_inter(const TextChar& other);
    inline float get_union(const TextChar& other);
    inline float get_iou(const TextChar& other);

    cv::Rect m_box;
    cv::Point2f m_center;
    float m_score;
};
}//end of namespace text
#endif



