/*
 * Brief: Utils for text line generation
 * Author: wangyuzhuo@bytedance.com
 * Date: 2016/12/08 21:02
 */
#ifndef _TEXTUTIL_H_
#define _TEXTUTIL_H_

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "TextCore.h"


namespace text {
#define PI 3.1415
std::vector<std::string> get_filelist(const std::string& dir_name);
std::string get_name_prefix(const std::string& name);
std::vector<TextChar>  load_boxes_from_file(const std::string& filepath);
void check_chars_locs(const cv::Mat& img, std::vector<TextChar>& boxes);
void vis_boxes(cv::Mat& im, const std::vector<TextChar>& boxes);
std::vector<TextChar> nms_boxes(std::vector<TextChar>& boxes, float thresh);
cv::Mat char_centers_to_mat(std::vector<TextChar>& boxes);
float compute_pts_dist(const cv::Point2f& pt1, const cv::Point2f& pt2);
float compute_pts_angle(const cv::Point2f& pt1, const cv::Point2f& pt2);
bool compare_box_x(const std::pair<int, TextChar>& p1, const std::pair<int, TextChar>& p2);
bool compare_box_y(const std::pair<int, TextChar>& p1, const std::pair<int, TextChar>& p2);
bool compare_box(const std::pair<int, TextChar>& p1, const std::pair<int, TextChar>& p2);
bool is_two_boxes_close(const TextChar& b1, const TextChar& b2);
bool is_two_pairs_same_angle(const TextPair& p1, const TextPair& p2);
bool merge_two_line_rect(cv::Rect& r1, cv::Rect& r2);
}//end of namespace text
#endif



