/*
 * Brief: Utils for most commonly used functions
 * Author: wangyuzhuo@bytedance.com
 * Date: 2016/12/21 15:12
 */

#ifndef _COMMONUTIL_H_
#define _COMMONUTIL_H_

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

namespace common {

//Ordinary Least Square line estimation
template<class CType>
class OLS{
    public:
        OLS();
        OLS(const std::vector<CType>& pts_in);
        bool do_OLS_estimation();
        inline bool compute_y(float x, float& res);
        float compute_y(const CType& in);
        inline float pt2line(const CType& pt_in);
        void print_pts();
       
        bool m_estimate_done;
        float m_a;
        float m_b;
        std::vector<CType> m_pts;
};

//Ordinary Least Square Estimation
template <class CType>
OLS<CType>::OLS() {
    m_a = 0.0;
    m_b = 0.0;
    m_pts.clear();
    m_estimate_done = false;
}

template <class CType>
OLS<CType>::OLS(const std::vector<CType>& pts_in) {
    m_a = 0.0f;
    m_b = 0.0f;
    m_pts = pts_in;
    m_estimate_done = false;
}


template <class CType>
void OLS<CType>::print_pts() {
    for (unsigned int i = 0; i < m_pts.size(); ++i) {
        std::cout << m_pts[i].x << " " << m_pts[i].y << std::endl;
    }
}

template <class CType>
bool OLS<CType>::do_OLS_estimation() {
    if (2 > m_pts.size()) {
        //Need 2 points at least
        return false;
    }

    float xi_xi_sum = 0.0f;
    float xi_sum = 0.0f;
    float xi_yi_sum = 0.0f;
    float yi_sum = 0.0f;
    for (unsigned int i = 0; i < m_pts.size(); ++i) {
        xi_xi_sum += (m_pts[i].x * m_pts[i].x);
        xi_sum += m_pts[i].x;
        xi_yi_sum += (m_pts[i].x * m_pts[i].y);
        yi_sum += m_pts[i].y;
    }
    m_a = 1.0f * (xi_xi_sum * yi_sum - xi_sum * xi_yi_sum) / (m_pts.size() * xi_xi_sum + xi_sum * xi_sum);
    m_b = 1.0f * (m_pts.size() * xi_yi_sum - xi_sum * yi_sum) / (m_pts.size() * xi_xi_sum - xi_sum * xi_sum);
    m_estimate_done = true;
    return true;
}

template <class CType>
bool OLS<CType>::compute_y(float x, float& y) {
    if (!m_estimate_done) {
        if(!do_OLS_estimation()) {
            return false;
        }
    }
    y = m_a + m_b * x; 
    // m_b * x - y + m_a = 0;
    return true;
}

template <class CType>
float OLS<CType>::compute_y(const CType& pt) {
   float  y = m_a + m_b * pt.x; 
    // m_b * x - y + m_a = 0;
    return y;
}

template <class CType>
float OLS<CType>::pt2line(const CType& pt_in) {
    float  dist = std::fabs(m_b * pt_in.x - pt_in.y + m_a) /std::sqrt(m_b * m_b + 1);
    return dist;
}


std::vector<std::string> get_filelist(const std::string& dir_name);
std::string get_name_prefix(const std::string& name);
//std::vector<cv::Point2f> find_four_pts_clockwise(const cv::Point2f* pts);
void find_four_pts_clockwise(cv::Point2f* pts);
}//end of namespace
#endif
