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

void TextLine::gen_text_pairs(std::vector<TextChar>& boxes) {
    cv::Mat centers = char_centers_to_mat(boxes);
    int knn_num = 5;
    cv::flann::KDTreeIndexParams indexParams(5);
    cv::flann::Index kdtree(centers, indexParams);
    cv::Mat indices;
    cv::Mat dists;
    kdtree.knnSearch(centers, indices, dists, knn_num, cv::flann::SearchParams(64));
    std::cout << indices << std::endl;

}
}
