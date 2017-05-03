#include <sstream>
#include <algorithm>
#include <stdio.h>
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
    //float area_other = other.get_area();
    float area_inter = get_inter(other);
    //float base_area = (m_score > other.m_score)?area_self:area_other;

    return (1.0f * area_inter / area_self);
    //return (1.0f * area_inter / base_area);
    //return (1.0f * area_inter / (area_self + area_other - area_inter));
}
 
inline void TextChar::print() const {
    std::cout << m_box.x << " " << m_box.y << " "
              << m_box.width << " " << m_box.height << std::endl;
}

bool TextPair::do_ols() {
    //std::vector<cv::Point2f> pts;
    m_ols.m_pts.clear();
    m_ols.m_pts.resize(m_pair_idx.size());
    for (unsigned int i = 0; i < m_pair_idx.size(); ++i) {
        m_ols.m_pts[i] = m_pair_idx[i].second.m_center;
    }
    if (!m_ols.do_OLS_estimation()) {
        return false;
    } else {
        float sum = 0;
        for (unsigned int i = 0; i < m_pair_idx.size(); ++i) {
            sum += m_ols.pt2line(m_ols.m_pts[i]);
        }
        //std::cout << "sum dist: " << sum << std::endl;
        m_ave_dist = sum / m_ols.m_pts.size();
        //std::cout << "ave dist: " << m_ave_dist << std::endl;
        return true;
    }
}

bool TextPair::is_box_included(const TextPair& p) {
    bool is_include = false;
    for(unsigned int i = 0; i < m_pair_idx.size(); ++i) {
        for (unsigned int j = 0; j < p.m_pair_idx.size(); ++j) {
            if(p.m_pair_idx[j].first == m_pair_idx[i].first){
                is_include = true;
                break;
            }
        }
    }
    return is_include;
}

void TextPair::sort_pairs_idx(const TextDirection& textdir) {
    if(m_pair_idx.size() > 1) {
        //eliminate duplicate indexs first before sorting
        std::vector<std::pair<int, TextChar> > temp = m_pair_idx;
        std::vector<bool> flags(m_pair_idx.size(), true);
        for (unsigned int i = 0; i < temp.size(); ++i) {
            for (unsigned int j = i + 1; j < temp.size(); ++j) {
                if (temp[j].first == temp[i].first) {
                    flags[j] = false;
                }
            }
        }

        m_pair_idx.clear();
        for (unsigned int i = 0; i < flags.size(); ++i){
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
    for (unsigned int i = 0; i < m_boxes.size(); ++i) {
        m_boxes[i].get_center();
    }
}
void TextLine::vis_pairs(const std::vector<TextPair>& pairs){ 
    for (unsigned int i = 0; i < pairs.size(); ++i) {
        cv::Mat vis_im = m_im.clone();
        //int id = m_final_pairs[i].m_idx;
        int id = pairs[i].m_start;
        int end_id = pairs[i].m_end;
        cv::Scalar color(0, 0, 255);
        for (unsigned int j = 0; j < pairs[i].m_pair_idx.size(); ++j) {
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

void  TextLine::vis_rotated_lines(const std::vector<cv::RotatedRect>& lines) {
    cv::Mat vis_im = m_im.clone();
    cv::Scalar color(0, 0, 255);
    for (unsigned int  i = 0; i < m_rotated_lines.size(); ++i) {
        cv::Point2f rect_points[4]; 
        m_rotated_lines[i].points(rect_points);
        for(int j = 0; j < 4; j++) {
            cv::line( vis_im, rect_points[j], rect_points[(j+1)%4], color, 1, 8 );
        }
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
            for (unsigned int j = 1; j < indexs.size(); ++j) {
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
        //m_final_pairs.clear();
        TextPair tp = m_pairs[i];
        //tp.do_ols();
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
                /*
                cv::Point2f s_pt = m_boxes[m_pairs[j].m_start].m_center;
                cv::Point2f e_pt = m_boxes[m_pairs[j].m_end].m_center;
                float s_y = tp.m_ols.compute_y(s_pt);
                float e_y = tp.m_ols.compute_y(e_pt);
                TextChar s_box = m_boxes[m_pairs[j].m_start];
                TextChar e_box = m_boxes[m_pairs[j].m_end];
                float s_thresh =  std::min(s_box.m_box.width, s_box.m_box.height);
                float e_thresh =  std::min(e_box.m_box.width, s_box.m_box.height);
                if(std::fabs(s_pt.y - s_y) <= s_thresh && std::fabs(e_pt.y - e_y) <= e_thresh) {
                    tp.m_pair_idx.insert(tp.m_pair_idx.end(), m_pairs[j].m_pair_idx.begin(), m_pairs[j].m_pair_idx.end());
                    flags[j] = false;
                    is_merge = true;
                    merge_cnt++;
                } else */
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
//    std::cout << "final path size: " << m_final_pairs.size() << std::endl;
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
    std::vector<TextPair> merged_pairs;
    std::vector<bool> is_merged(m_initial_lines.size(), false);
    std::vector<cv::Rect> temp;
    int line_num = 0;
    for (unsigned int i = 0; i < m_initial_lines.size(); ++i) {
        if(is_merged[i]) {
            continue;
        }
        TextPair tp = m_final_pairs[i];
        cv::Rect cur_rect = m_initial_lines[i];
        bool is_continue = true;
        while(is_continue) {
            int cnt = 0;
            for (unsigned int j =  0; j < m_initial_lines.size(); ++j) {
                if (is_merged[j]) {
                    continue;
                }
                if (j == i) {
                    continue;
                }
                if(merge_two_line_rect(cur_rect, m_initial_lines[j])) {
                    tp.m_pair_idx.insert(tp.m_pair_idx.end(), m_final_pairs[j].m_pair_idx.begin(),  m_final_pairs[j].m_pair_idx.end());
                    tp.sort_pairs_idx(m_direction);
                    is_merged[j] = true;
                    cnt++;
                }
            }
            if (cnt == 0) {
                is_continue = false;
            }
        }
        cur_rect.x = std::max(0, cur_rect.x);
        cur_rect.y = std::max(0, cur_rect.y);
        if (cur_rect.x + cur_rect.width> m_im.cols) {
            cur_rect.width = m_im.cols - cur_rect.x - 1;
        }
        if (cur_rect.y + cur_rect.height > m_im.rows) {
            cur_rect.height = m_im.rows - cur_rect.y - 1;
        }
        line_num++;
        temp.push_back(cur_rect);
        merged_pairs.push_back(tp);
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
    m_final_pairs = merged_pairs;
}

void TextLine::get_rotated_bounding_box(const TextPair& line) {
    std::vector<cv::Point> contour;
    cv::Rect s_rect = m_boxes[line.m_start].m_box;
    cv::Rect e_rect = m_boxes[line.m_end].m_box;

   contour.push_back(cv::Point(s_rect.x, s_rect.y)); 
   contour.push_back(cv::Point(s_rect.x + s_rect.width - 1, s_rect.y)); 
   for (unsigned int i = 1; i < line.m_pair_idx.size() - 1; ++i) {
       cv::Rect crect = line.m_pair_idx[i].second.m_box;
       contour.push_back(cv::Point(crect.x, crect.y));
       contour.push_back(cv::Point(crect.x + crect.width - 1, crect.y));
   }
   contour.push_back(cv::Point(e_rect.x, e_rect.y));
   contour.push_back(cv::Point(e_rect.x + e_rect.width - 1, e_rect.y));
   contour.push_back(cv::Point(e_rect.x + e_rect.width - 1, e_rect.y + e_rect.height - 1));
   contour.push_back(cv::Point(e_rect.x, e_rect.y + e_rect.height - 1));
   for (unsigned int i = line.m_pair_idx.size() - 1; i < 0; i--) {
       cv::Rect crect = line.m_pair_idx[i].second.m_box;
       contour.push_back(cv::Point(crect.x + crect.width - 1, crect.y + crect.height - 1));
       contour.push_back(cv::Point(crect.x, crect.y + crect.height - 1));
   }
   contour.push_back(cv::Point(s_rect.x + s_rect.width - 1, s_rect.y + s_rect.height - 1));
   contour.push_back(cv::Point(s_rect.x, s_rect.y + s_rect.height - 1)); 
   cv::RotatedRect bbox = minAreaRect(cv::Mat(contour));

   m_rotated_lines.push_back(bbox);
   m_final_lines.push_back(bbox.boundingRect());
}

void TextLine::get_all_rotated_lines() {
    for (unsigned int i = 0; i < m_final_pairs.size(); ++i) {
        get_rotated_bounding_box(m_final_pairs[i]);
    }
}

void TextLine::crop_and_rotate_lines(std::vector<cv::Mat>& outs, std::vector< std::vector<cv::Rect> >& char_pos) {
    outs.clear();
    char_pos.clear();
    char_pos.resize(m_final_pairs.size());
    for (unsigned int i = 0; i < m_final_pairs.size(); ++i) {
        m_final_pairs[i].do_ols();
        //std::cout << m_im.cols << " " << m_im.rows << std::endl;
        //std::cout << m_initial_lines[i].x << " " << m_initial_lines[i].y << " " << m_initial_lines[i].width << " " << m_initial_lines[i].height << std::endl;
        cv::Mat crop_im = m_im(m_initial_lines[i]);
        cv::Mat rotated;
        std::vector<cv::Rect> line_chars;

        cv::Point2f src_pts[4];
        cv::Point2f dst_pts[4];
        cv::Point2f tmp_pts[4];
        m_rotated_lines[i].points(tmp_pts);
        /*
        std::cout  << tmp_pts[0] << std::endl
            << tmp_pts[1] << std::endl
            << tmp_pts[2] << std::endl
            << tmp_pts[3] << std::endl << std::endl;
        */
        common::find_four_pts_clockwise(tmp_pts);
        src_pts[0] = tmp_pts[0];
        src_pts[1] = tmp_pts[1];
        src_pts[2] = tmp_pts[2];
        src_pts[3] = tmp_pts[3];
        /*
        std::cout  << src_pts[0] << std::endl
            << src_pts[1] << std::endl
            << src_pts[2] << std::endl
            << src_pts[3] << std::endl << std::endl;
        */
        float angle = std::fabs(compute_pts_angle(src_pts[1], src_pts[0]));
//      float angle = std::fabs(std::atan(m_final_pairs[i].m_ols.m_b) * 180 / 3.1415);
        if (angle <= 5) {
            rotated = crop_im;
            TextPair tp = m_final_pairs[i];
            cv::Rect trect = m_initial_lines[i];
            line_chars.resize(tp.m_pair_idx.size());
            //int hei = m_initial_lines[i].height;
            for (unsigned int j = 0; j < tp.m_pair_idx.size(); ++j) {
                cv::Rect tbox = tp.m_pair_idx[j].second.m_box;
                tbox.x = tbox.x - trect.x;
                tbox.y = tbox.y - trect.y;
                line_chars[j] = tbox;
            }
            char_pos.push_back(line_chars);
        } else {
            int width = static_cast<int>(compute_pts_dist(src_pts[0], src_pts[1]));
            int height = static_cast<int>(compute_pts_dist(src_pts[1], src_pts[2]));

            dst_pts[0] = cv::Point2f(0, 0); //tl
            dst_pts[1] = cv::Point2f( width - 1, 0); //tr
            dst_pts[2] = cv::Point2f( width - 1, height - 1); //br
            dst_pts[3] = cv::Point2f(0, height - 1); //bl
            cv::Size size(width, height);

            cv::Mat warpMatrix = cv::getPerspectiveTransform(src_pts, dst_pts);
            cv::warpPerspective(m_im, rotated, warpMatrix, size, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
            //cv::warpPerspective(m_im, rotated, warpMatrix, rotated.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT);

            TextPair tp = m_final_pairs[i];
            //cv::Rect trect = m_initial_lines[i];
            line_chars.resize(tp.m_pair_idx.size());
            //int left = 0;
            //int hei = m_initial_lines[i].height;
            for (unsigned int j = 0; j < tp.m_pair_idx.size(); ++j) {
                cv::Rect crect = tp.m_pair_idx[j].second.m_box;

                std::vector<cv::Point2f> before_trans, after_trans;
                before_trans.resize(4);
                before_trans[0] = cv::Point2f(crect.x, crect.y); //tl
                before_trans[1] = cv::Point2f(crect.x + crect.width - 1, crect.y); //tr
                before_trans[2] = cv::Point2f(crect.x + crect.width - 1, crect.y + crect.height - 1); //br
                before_trans[3] = cv::Point2f(crect.x, crect.y + crect.height - 1);  //bl
                perspectiveTransform(before_trans, after_trans, warpMatrix);

//                cv::Point p1(static_cast<int>(after_trans[0].x), static_cast<int>(after_trans[0].y));
//                cv::Point p2(static_cast<int>(after_trans[1].x), static_cast<int>(after_trans[1].y));
                cv::Point p1(static_cast<int>(tp.m_pair_idx[0].second.m_box.x), static_cast<int>(tp.m_pair_idx[0].second.m_box.y));
                cv::Point p2(static_cast<int>(crect.x), static_cast<int>(crect.y));
                int dist = compute_pts_dist(p1, p2);
                //line_chars[j] = cv::Rect(left, 0, dist, height);
                //left += dist;
                line_chars[j] = cv::Rect(dist, 0, crect.width, height);
            }
            char_pos.push_back(line_chars);
        }
        outs.push_back(rotated.clone());
#if 1
        std::string prefix = common::get_name_prefix(m_image_name);
        char savename[128];
        /*
        for (unsigned int j = 0; j < line_chars.size(); ++j) {
            cv::rectangle(rotated, line_chars[j], cv::Scalar(255, 0, 0));
        }
        */
        cv::Rect tt_rect = m_initial_lines[i];
        sprintf(savename,"%s/%s_%d_%d_%d_%d_%d.jpg", m_save_dir.c_str(), prefix.c_str(), i, tt_rect.x, tt_rect.y, tt_rect.x + tt_rect.width - 1, tt_rect.y + tt_rect.height -1);
        cv::imwrite(savename, rotated);
#endif
        //m_final_pairs[i].do_ols();
        //float angle = std::atan(m_final_pairs[i].m_ols.m_b) * 180 / 3.1415;
        //std::cout << "angle " << i << angle << std::endl;

    }
}

}//end of namespace
