#include <sstream>
#include <algorithm>
#include "CommonUtil.h"
#include "TextCore.h"
#include "TextUtil.h"

namespace text {
TextChar::TextChar():m_box(cv::Rect(0,0,0,0)),
                     m_center(cv::Point2f(0.0f, 0.0f)),
                     m_score(0.0f){
}

TextChar::~TextChar(){
}

TextChar::TextChar(const std::string& line, bool has_score) {
    std::stringstream ss(line);
    if (has_score) {
        ss >> m_box.x >> m_box.y 
           >> m_box.width >> m_box.height
           >> m_score;
    } else {
        ss >> m_box.x >> m_box.y 
           >> m_box.width >> m_box.height;
        m_score = 0.0f;
    }
    get_center();
}

TextChar::TextChar(const cv::Rect& rect, const float score) {
    m_box = rect;
    m_score = score;
    get_center();
}

inline void TextChar::get_center() {
    m_center.x = m_box.x + 0.5f * m_box.width;
    m_center.y = m_box.y + 0.5f * m_box.height;
}

inline float TextChar::get_area() const{
    return 1.0 * m_box.width * m_box.height;
}

inline float TextChar::get_inter(const TextChar& other) const{
    float xstart = std::max(m_box.x, other.m_box.x);
    float xend = std::min(m_box.x + m_box.width - 1,
                          other.m_box.x + other.m_box.width - 1);
    float ystart = std::max(m_box.y, other.m_box.y);
    float yend = std::min(m_box.y + m_box.height - 1, 
                          other.m_box.y + other.m_box.height - 1);
    float delta_x = xend - xstart;
    float delta_y = yend - ystart;
    if ((delta_x < EPSILON) || (delta_y < EPSILON)) {
        return 0.0f;
    }
    return (delta_x * delta_y);
}

float TextChar::get_iou(const TextChar& other) const{
    float area_self = get_area();
    float area_other = other.get_area();
    float area_inter = get_inter(other);
    float base_area = (m_score > other.m_score)?area_self:area_other;

    return (1.0f * area_inter / area_self);
    //return (1.0f * area_inter / base_area);
    //return (1.0f * area_inter / (area_self + area_other - area_inter));
}
 
inline void TextChar::print() const {
    std::cout << m_box.x << " " << m_box.y << " "
              << m_box.width << " " << m_box.height << std::endl;
}

bool TextPair::is_box_included(const TextPair& p) {
    bool is_include = false;
    for(int i = 0; i < m_pair_idx.size(); ++i) {
        for (int j = 0; j < p.m_pair_idx.size(); ++j) {
            if(p.m_pair_idx[j].first == m_pair_idx[i].first){
                is_include = true;
            }
        }
    }
    return is_include;
}

void TextPair::sort_pairs_idx(const TextDirection& textdir) {
    if(m_pair_idx.size() > 1){
        //eliminate duplicate indexs first before sorting
        std::vector<std::pair<int, TextChar> > temp = m_pair_idx;
        std::vector<bool> flags(m_pair_idx.size(), true);
        for (int i = 0; i < temp.size(); ++i) {
            for (int j = i + 1; j < temp.size(); ++j) {
                if (temp[j].first == temp[i].first) {
                    flags[j] = false;
                }
            }
        }
        m_pair_idx.clear();
        for (int i = 0; i < flags.size(); ++i){
            if(flags[i]) {
                m_pair_idx.push_back(temp[i]);
            }
        }
        if (VER_ONLY == textdir) {
            std::sort(m_pair_idx.begin(), m_pair_idx.end(), compare_box_y);
        } else if (HOR_ONLY == textdir) {
            std::sort(m_pair_idx.begin(), m_pair_idx.end(), compare_box_x);
        }
    }
   m_start = m_pair_idx[0].first;
   m_end = m_pair_idx[m_pair_idx.size() - 1].first;

   if (m_start == m_end) {
       m_isolate = true;
   } else {
       m_isolate = false;
   }
}

void TextPair::print() {
    for (unsigned int i = 0; i < m_pair_idx.size(); ++i) {
        std::cout << m_pair_idx[i].first << " ";
    }
    std::cout << std::endl;
    std::cout << "start id: " << m_start << " end id: " << m_end << std::endl;
}

TextLine::TextLine(const cv::Mat& img, const std::vector<TextChar>& boxes) {
    m_im = img.clone();
    m_boxes = boxes;
}
void TextLine::vis_pairs(const std::vector<TextPair>& pairs){ 
    for (int i = 0; i < pairs.size(); ++i) {
        cv::Mat vis_im = m_im.clone();
        //int id = m_final_pairs[i].m_idx;
        int id = pairs[i].m_start;
        int end_id = pairs[i].m_end;
        cv::Scalar color(0, 0, 255);
        for (int j = 0; j < pairs[i].m_pair_idx.size(); ++j) {
            int kid = pairs[i].m_pair_idx[j].first;
            cv::rectangle(vis_im, m_boxes[kid].m_box, color);
        }
        cv::rectangle(vis_im, m_boxes[id].m_box, cv::Scalar(255, 0, 0));
        cv::rectangle(vis_im, m_boxes[end_id].m_box, cv::Scalar(255, 0, 0));
        //cv::rectangle(vis_im, m_initial_lines[i], cv::Scalar(0, 255, 0));
        std::string prefix = common::get_name_prefix(m_image_name);
        char savename[128];
        sprintf(savename,"%s/%s_%d.jpg", m_save_dir.c_str(), prefix.c_str(), i);
        cv::imwrite(savename, vis_im);
    }
}

void TextLine::vis_lines(const std::vector<cv::Rect>& lines) {
    cv::Mat vis_im = m_im.clone();
    cv::Scalar color(0, 0, 255);
    for (unsigned int i = 0; i < m_initial_lines.size(); ++i) {
        cv::rectangle(vis_im, m_initial_lines[i], color);
    }
    char savename[128];
    sprintf(savename,"%s/%s", m_save_dir.c_str(), m_image_name.c_str());
    cv::imwrite(savename, vis_im);
}

void TextLine::gen_text_pairs() {
    cv::Mat centers = char_centers_to_mat(m_boxes);//each row of centers represents a center point of a char box
    int knn_num = std::min(5, centers.rows);// nearest neightbour num for each char box
    cv::flann::KDTreeIndexParams indexParams(knn_num);
    cv::flann::Index kdtree(centers, indexParams); //kdtree is fast, but it initiated with random seeds, take care.
    cv::Mat indices;
    cv::Mat dists;
    kdtree.knnSearch(centers, indices, dists, knn_num, cv::flann::SearchParams(64));
    //std::cout << indices << std::endl;
    eliminate_unvalid_pair(centers, dists, indices);
    //std::cout << "eliminate done " << centers.rows << std::endl;
    //std::cout << dists << std::endl;
}

void TextLine::eliminate_unvalid_pair(const cv::Mat& centers,
                                      const cv::Mat& dists,
                                      const cv::Mat& indices) {
   //eliminate left / top neighbors
    // std::cout << "boxes num: " << centers.rows << std::endl;
     for (int i = 0; i < centers.rows; ++i) {
        TextPair tp;
        std::vector<unsigned int> indexs;
        std::vector<float> angles;
        tp.m_idx = i;
        for (int j = 1; j < indices.cols; ++j) {
            float pt_dist = std::sqrt(dists.at<float>(i, j));
            int kid = indices.at<int>(i, j);
            float center_dist_ratio = pt_dist / (0.5 * m_boxes[i].m_box.width + 0.5 *  m_boxes[kid].m_box.width);
            if (center_dist_ratio <= 1.5f){
                cv::Point2f pt1(centers.at<float>(i, 0), centers.at<float>(i, 1));
                cv::Point2f pt2(centers.at<float>(kid, 0), centers.at<float>(kid, 1));
                float angle = compute_pts_angle(pt1, pt2);
                if(VER_ONLY == m_direction) {
                    if(-30 > angle || angle > 30) {
                        angles.push_back(angle);
                        indexs.push_back(kid);
                    }
                } else {
                    if (-30 <= angle && angle <= 30){
                        angles.push_back(angle);
                        indexs.push_back(kid);
                    }
                }
            }
        }
        tp.m_pair_idx.push_back(std::pair<int, TextChar>(i, m_boxes[i]));
        if (indexs.size() > 0) {
            tp.m_pair_idx.push_back(std::pair<int, TextChar>(indexs[0], m_boxes[indexs[0]]));
            for (int j = 1; j < indexs.size(); ++j) {
                if (std::fabs(angles[j] - angles[0]) <= 15.0f) {
                    tp.m_pair_idx.push_back(std::pair<int, TextChar>(indexs[j], m_boxes[indexs[j]]));
                }
            }
        }
        tp.sort_pairs_idx(m_direction);
        m_pairs.push_back(tp);
    }
}

void TextLine::merge_text_pairs(){
    std::vector<bool> flags(m_pairs.size(), true);
    for (unsigned int i = 0; i < m_pairs.size(); ++i){
        if(!flags[i]) {
            continue;
        }

        TextPair tp = m_pairs[i];
        bool merge_done = false;
        while(!merge_done) {
            int merge_cnt = 0;
            for (unsigned int j = i + 1; j < m_pairs.size(); ++j) {
                bool is_merge = false;
                if(!flags[j]) {
                    continue;
                }
                /*
                if (!is_two_pairs_same_angle(tp, m_pairs[j])) {
                    continue;
                }
                */
                if (is_two_boxes_close(m_boxes[tp.m_end], m_boxes[m_pairs[j].m_start])
                    && is_two_pairs_same_angle(tp, m_pairs[j])) {
                    tp.m_pair_idx.insert(tp.m_pair_idx.end(), m_pairs[j].m_pair_idx.begin(), m_pairs[j].m_pair_idx.end());
                    flags[j] = false;
                    is_merge = true;
                    merge_cnt++;
                } else if (is_two_boxes_close(m_boxes[tp.m_start], m_boxes[m_pairs[j].m_end])) {
                    tp.m_pair_idx.insert(tp.m_pair_idx.begin(), m_pairs[j].m_pair_idx.begin(), m_pairs[j].m_pair_idx.end());
                    flags[j] = false;
                    is_merge = true;
                    merge_cnt++;
                } else if (tp.is_box_included(m_pairs[j])) {
                    tp.m_pair_idx.insert(tp.m_pair_idx.begin(), m_pairs[j].m_pair_idx.begin(), m_pairs[j].m_pair_idx.end());
                    flags[j] = false;
                    is_merge = true;
                    merge_cnt++;
                }
                if (is_merge) {
                    tp.sort_pairs_idx(m_direction);
                }
            }
            if (merge_cnt == 0) {
                merge_done = true;
            }
        }
        m_final_pairs.push_back(tp);
    }
    std::cout << "final path size: " << m_final_pairs.size() << std::endl;
}

void TextLine::gen_initial_lines() {
    for (unsigned int i = 0; i < m_final_pairs.size(); ++i) {
        cv::Rect line;
        int start_id = m_final_pairs[i].m_start;
        int end_id = m_final_pairs[i].m_end;
        line.x = m_boxes[start_id].m_box.x;
        line.width = m_boxes[end_id].m_box.x + m_boxes[end_id].m_box.width - line.x;
        float liney = m_final_pairs[i].m_pair_idx[0].second.m_box.y;
        float lineb = 0;
        for (unsigned int j = 0; j < m_final_pairs[i].m_pair_idx.size(); ++j) {
            float boxy = m_final_pairs[i].m_pair_idx[j].second.m_box.y;
            float boxb = m_final_pairs[i].m_pair_idx[j].second.m_box.y + m_final_pairs[i].m_pair_idx[j].second.m_box.height - 1;
            liney = (liney<=boxy)?liney:boxy;
            lineb = (lineb>=boxb)?lineb:boxb;
        }
        line.y = static_cast<int>(liney);
        line.height = static_cast<int>(lineb - liney + 1);
        m_initial_lines.push_back(line);
    }
}

void TextLine::merge_initial_lines() {
    std::vector<bool> is_merged(m_initial_lines.size(), false);
    std::vector<cv::Rect> temp;
    int line_num = 0;
    for(unsigned int i = 0; i < m_initial_lines.size(); ++i) {
        if(is_merged[i]) {
            continue;
        }
        cv::Rect cur_rect = m_initial_lines[i];
        bool is_continue = true;
        while(is_continue) {
            int cnt = 0;
            for (unsigned int j =  0; j < m_initial_lines.size(); ++j) {
                if (is_merged[j]) {
                    continue;
                }
                if(merge_two_line_rect(cur_rect, m_initial_lines[j])) {
                    is_merged[j] = true;
                    cnt++;
                }
            }
            if (cnt == 0) {
                is_continue = false;
            }
        }
        line_num++;
        temp.push_back(cur_rect);
    }
    /*
    std::vector<cv::Rect> temp(line_num);
    int idx = 0;
    for(unsigned int i = 0; i < m_initial_lines.size(); ++i){
        if(!is_merged[i]) {
            temp[idx] = m_initial_lines[i];
                idx++;
        }
    }*/

    m_initial_lines = temp;
}

}//end of namespace
