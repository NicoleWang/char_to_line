/*
 * Brief: Basic data structure for text line generation
 * Author: wangyuzhuo@bytedance.com
 * Date: 2016/12/08 20:35
 */
#ifndef _TEXTCORE_H_
#define _TEXTCORE_H_

//#include <vector>
#include <string>
#include <utility>
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
    void get_center();
    float get_area() const;
    float get_inter(const TextChar& other) const;
    float get_union(const TextChar& other) const;
    float get_iou(const TextChar& other) const;
    void print() const;

    cv::Rect m_box;
    cv::Point2f m_center;
    float m_score;
};

class TextPair {
    public:
    //TextPair(const unsigned int idx);
    void sort_pairs_idx();
    bool is_box_included(const TextPair& p);
    //void set_start_end();
    void print();
    unsigned int m_idx;
    unsigned int m_start;
    unsigned int m_end;
    bool m_isolate;
    std::vector<std::pair<int, TextChar> > m_pair_idx;
    std::vector<float> m_angles;

};

class TextLine {
    public:
    TextLine(const cv::Mat& img, const std::vector<TextChar>& boxes);
    void gen_text_pairs();
    void eliminate_unvalid_pair(const cv::Mat& centers,
                                          const cv::Mat& dists,
                                          const cv::Mat& indices);
    void merge_text_pairs();
    void gen_initial_lines();
    void merge_initial_lines();
    void gen_final_lines();

    void vis_pairs(const std::vector<TextPair>& pairs); 
    void vis_lines(const std::vector<cv::Rect>& lines); 
    cv::Mat m_im;
    std::string m_image_name;
    std::string m_save_dir;;
    std::vector<TextChar> m_boxes;
    std::vector<TextPair> m_pairs;
    std::vector<TextPair> m_final_pairs;
    std::vector<cv::Rect> m_initial_lines;
    std::vector<cv::Rect> m_final_lines;
};
}//end of namespace text
#endif



