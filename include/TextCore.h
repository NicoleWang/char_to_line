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

#define EPSILON 0.0001
#define KNN_NUM 5
namespace text {

class TextChar {
    public:
    TextChar();
    TextChar(const std::string& line, bool has_score);
    TextChar(const cv::Rect& rect, const float score);
    ~TextChar();
    inline void get_center();
    inline float get_area() const;
    inline float get_inter(const TextChar& other);
    inline float get_union(const TextChar& other);
    inline float get_iou(const TextChar& other);

    cv::Rect m_box;
    cv::Point2f m_center;
    float m_score;
};

class TextPair {
    public:
    //TextPair();
    //TextPair(const unsigned int idx);
    unsigned int m_idx;
    std::vector<unsigned int> m_pair_idx;

    void eliminate_unvalid_pair();
};

class TextLine {
    public:
    TextLine(const cv::Mat& img, const std::vector<TextChar>& boxes);
    void gen_text_pairs();
    void vis_pairs();

    cv::Mat m_im;
    std::string m_imname;
    std::vector<TextChar> m_boxes;
    std::vector<TextPair> m_pairs;
};
}//end of namespace text
#endif



