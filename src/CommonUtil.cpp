#include "CommonUtil.h"
#include <dirent.h>

namespace common {
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

std::string get_name_prefix(const std::string& name) {
    size_t dot_pos = name.rfind(".");
    if (dot_pos == std::string::npos) {
        dot_pos = name.length();
    }
    size_t slash_pos = name.rfind("/");
    if (slash_pos == std::string::npos) {
        slash_pos = 0;
    }
    std::string prefix = name.substr(slash_pos, dot_pos - slash_pos);
    return prefix;
}

//Ordinary Least Square Estimation
template <class CType>
OLS<CType>::OLS() {
    m_a = 0.0;
    m_b = 0.0;
    m_pts.clear();
}

template <class CType>
OLS<CType>::OLS(const std::vector<CType>& pts_in) {
    m_a = 0.0f;
    m_b = 0.0f;
    m_pts = pts_in;
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
}

}//end of name space
