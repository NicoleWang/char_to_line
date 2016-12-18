#include <sstream>
#include <algorithm>
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

void TextChar::get_center() {
    m_center.x = m_box.x + 0.5f * m_box.width;
    m_center.y = m_box.y + 0.5f * m_box.height;
}

float TextChar::get_area() const{
    return 1.0 * m_box.width * m_box.height;
}

float TextChar::get_inter(const TextChar& other) {
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

float TextChar::get_iou(const TextChar& other) {
    float area_self = get_area();
    float area_other = other.get_area();
    float area_inter = get_inter(other);

    return (1.0f * area_inter / (area_self + area_other - area_inter));
}

TextLine::TextLine(const cv::Mat& img, const std::vector<TextChar>& boxes) {
    m_im = img.clone();
    m_boxes = boxes;
}
void TextLine::vis_pairs(){ 
    for (int i = 0; i < m_pairs.size(); ++i) {
        cv::Mat vis_im = m_im.clone();
        int id = m_pairs[i].m_idx;
        cv::Scalar color(0, 0, 255);
        cv::rectangle(vis_im, m_boxes[id].m_box, color);
        for (int j = 0; j < m_pairs[i].m_pair_idx.size(); ++j) {
            int kid = m_pairs[i].m_pair_idx[j];
            cv::rectangle(vis_im, m_boxes[kid].m_box, color);
        }
        char savename[128];
        sprintf(savename,"result_%d.jpg", i);
        cv::imwrite(savename, vis_im);
    }
}


void TextLine::gen_text_pairs() {
    cv::Mat centers = char_centers_to_mat(m_boxes);
    int knn_num = 5;// nearest neightbour num for each char box
    cv::flann::KDTreeIndexParams indexParams(5);
    cv::flann::Index kdtree(centers, indexParams); //kdtree is fast, but it initiated with random seeds, take care.
    cv::Mat indices;
    cv::Mat dists;
    kdtree.knnSearch(centers, indices, dists, knn_num, cv::flann::SearchParams(64));
    //std::cout << indices << std::endl;
    //std::cout << dists << std::endl;

    for (int i = 0; i < centers.rows; ++i) {
        TextPair tp;
        tp.m_idx = i;
        for (int j = 1; j < knn_num; ++j) {
            float pt_dist = std::sqrt(dists.at<float>(i, j));
            int kid = indices.at<int>(i, j);
            float center_dist_ratio = pt_dist / (0.5 * m_boxes[i].m_box.width + 0.5 *  m_boxes[kid].m_box.width);
            if (center_dist_ratio <= 1.5f 
                && (m_boxes[i].m_box.x <= m_boxes[kid].m_box.x 
                    || m_boxes[i].m_box.y <= m_boxes[kid].m_box.y)){
                tp.m_pair_idx.push_back(kid); //push back valid neighbor
            }
        }
        m_pairs.push_back(tp);
    }
}
}//end of namespace
