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

}//end of namespace text
#endif



