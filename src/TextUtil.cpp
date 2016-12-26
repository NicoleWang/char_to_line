#include <iostream>
#include <fstream>
#include <sstream>
#include <dirent.h>
#include "TextUtil.h"

namespace text{


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

inline bool compare_char_score(const TextChar& c1, const TextChar& c2) {
    return (c1.m_score > c2.m_score)?true:false;
}

std::vector<TextChar> nms_boxes(std::vector<TextChar>& boxes, float thresh) {
    std::vector<bool> is_filtered(boxes.size(), false);
    sort(boxes.begin(), boxes.end(), compare_char_score);
    //eliminate char boxes with small confidence
    for (unsigned int i = 0; i < boxes.size(); ++i) {
        if (boxes[i].m_score < 0.1f) {
            is_filtered[i] = true;
        }
    }
    //std::vector<TextChar> nms_boxes;
    int cnt = 0;
    for(unsigned int i = 0; i < boxes.size(); ++i){
        if (is_filtered[i]) {
            continue;
        }
        cnt++;

        for (unsigned int j = i + 1; j < boxes.size(); ++j) {
            if (is_filtered[j]) {
                continue;
            }
            float overlap = boxes[i].get_iou(boxes[j]);
  //          std::cout << overlap << std::endl;
            if (overlap >= thresh) {
                is_filtered[j] = true;
            }
        }
    }
    std::vector<TextChar> out(cnt);
    int idx = 0;
    for(int i = 0; i < boxes.size(); ++i) {
        if(!is_filtered[i]) {
            out[idx] = boxes[i];
            idx++;
        }
    }
    return out;
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

float compute_pts_angle(const cv::Point2f& pt1, const cv::Point2f& pt2) {
    float temp = (1.0f * pt2.y - pt1.y) / (1.0f * pt2.x - pt1.x);
    float angle = std::atan(temp) * 180 / PI;
    return angle;
}

bool compare_box_x(const std::pair<int, TextChar>& p1, const std::pair<int, TextChar>& p2) {
    //bool res = (p1.second.m_box.x <= p2.second.m_box.x)?true:false;
    return (p1.second.m_box.x < p2.second.m_box.x)?true:false;
}

bool compare_box_y(const std::pair<int, TextChar>& p1, const std::pair<int, TextChar>& p2) {
    //bool res = (p1.second.m_box.x <= p2.second.m_box.x)?true:false;
    return (p1.second.m_box.y < p2.second.m_box.y)?true:false;
}
bool is_two_boxes_close(const TextChar& left, const TextChar& right) {
    float dist = compute_pts_dist(left.m_center, right.m_center);
//  float dist = right.m_box.x - left.m_box.x - left.m_box.width + 1;
    float thresh = 1.0 * std::min(left.m_box.width, right.m_box.width);
    return dist<=thresh?true:false;
}

bool is_two_pairs_same_angle(const TextPair& p1, const TextPair& p2) {
    float delta1 = 0;
    float meany1 = 0;
    cv::Point2f p1_start_center = p1.m_pair_idx[0].second.m_center;
    for (int i = 0; i < p1.m_pair_idx.size(); ++i) {
        cv::Point2f p1_center = p1.m_pair_idx[1].second.m_center;
        meany1 += p1_center.y;
        //delta1 += std::fabs(p1_center.y - p1_start_center.y); 
    }
    meany1 = meany1 / p1.m_pair_idx.size();

    float delta2 = std::fabs(p2.m_pair_idx[0].second.m_center.y - p1_start_center.y);
    float meany2 = 0;
    for (int i = 0; i < p2.m_pair_idx.size(); ++i) {
        cv::Point2f p2_center = p2.m_pair_idx[i].second.m_center;
        float temp = std::fabs(p2_center.y - p1_start_center.y);
        if (delta2 > temp) {
            delta2 = temp;
        }
        meany2 += p2_center.y;
    }
    meany2 = meany2 / p2.m_pair_idx.size();
    if (std::fabs(meany2 - meany1) >= p1.m_pair_idx[0].second.m_box.height * 0.5) {
        return false;
    } else {
        return true;
    }
}

bool merge_two_line_rect(cv::Rect& r1, cv::Rect& r2) {
    bool res = false;
    float h_dist = std::min(std::fabs(r1.x - r2.x - r2.width + 1), std::fabs(r2.x - r1.x - r1.width + 1));
    float base_height = 0.5 * std::min(r1.height, r2.height);
    float min_top = std::min(r1.y, r2.y);
    float max_bot = std::max(r1.y + r1.height - 1, r2.y + r2.height - 1);
    float size_ratio = 1.0 * (max_bot - min_top) / std::min(r1.height, r2.height);
    if( size_ratio <= 1.5 && h_dist / base_height <= 0.5) {
        res = true;
        float left = std::min(r1.x, r2.x);
        float top = std::min(r1.y, r2.y);
        r1.width =  std::max(r1.x + r1.width, r2.x + r2.width) - left;
        r1.height = std::max(r1.height, r2.height);
        r1.x = left;
        r1.y = top;
    }
    return res;
}

}
