#include <iostream>
#include <fstream>
#include <sstream>
#include <dirent.h>
#include "TextUtil.h"

namespace text{

std::vector<std::string> get_filelist(const std::string& dir_name) {
    std::vector<std::string> filename_v;
    DIR* dir;
    struct dirent* entry;
    if ((dir = opendir(dir_name.c_str())) != NULL) {
        while((entry = readdir(dir)) != NULL){
            std::string name = entry->d_name;
            if (name == std::string(".") || (name == std::string(".."))) {
                continue;
            }
            filename_v.push_back(name);
        }
    }
    closedir(dir);
    return filename_v;
}

std::vector<TextChar>  load_boxes_from_file(const std::string& filepath) {
    std::vector<TextChar> boxes;
    std::ifstream infile;
    infile.open(filepath.c_str(), std::ios::in);
    if(!infile.is_open()) {
        std::cerr << "Open file: " << filepath << " failed! " << std::endl;
    } else {
        std::string line;
        while(std::getline(infile, line)) {
            std::stringstream ss(line);
            cv::Rect rect;
            float score = 0.0f;
            ss >> rect.x >> rect.y >> rect.width >> rect.height >> score;
            rect.width = rect.width - rect.x + 1;
            rect.height = rect.height - rect.y + 1;
            boxes.push_back(TextChar(rect, score));
        }
    }
    infile.close();
    return boxes;
}

void vis_boxes(cv::Mat& im, const std::vector<TextChar>& boxes) {
    for (unsigned int i = 0; i < boxes.size(); ++i) {
        cv::Point lt(boxes[i].m_box.x, boxes[i].m_box.y);
        cv::Point rb(boxes[i].m_box.x + boxes[i].m_box.width - 1, 
                     boxes[i].m_box.y + boxes[i].m_box.height - 1);
        cv::rectangle(im, boxes[i].m_box, cv::Scalar(255, 0, 0));
        cv::Rect rect = boxes[i].m_box;
        //std::cout << rect.x << " " << rect.y << " " << rect.width << " " << rect.height << std::endl;
    }
}

cv::Mat char_centers_to_mat(std::vector<TextChar>& boxes) {
    cv::Mat box_mat(boxes.size(), 2, CV_32FC1);
    for (int i = 0; i < boxes.size(); ++i) {
        box_mat.at<float>(i, 0) = boxes[i].m_center.x;
        box_mat.at<float>(i, 1) = boxes[i].m_center.y;
    }
    return box_mat;
}

float compute_pts_dist(const cv::Point2f& pt1, const cv::Point2f& pt2) {
    return std::sqrt((pt1.x - pt2.x) * (pt1.x - pt2.x) 
                     + (pt1.y - pt2.y) * (pt1.y - pt2.y));
}

}
