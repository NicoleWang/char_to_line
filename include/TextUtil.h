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
std::vector<std::string> get_filelist(const std::string& dir_name);
std::vector<TextChar>  load_boxes_from_file(const std::string& filepath);
void vis_boxes(cv::Mat& im, const std::vector<TextChar>& boxes);
cv::Mat char_centers_to_mat(std::vector<TextChar>& boxes);

}//end of namespace text
#endif



