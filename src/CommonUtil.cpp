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

bool compare_2fpts(const cv::Point2f& pt1, const cv::Point2f& pt2) {
    return (pt1.x < pt2.x)?true:false;
}
void  find_four_pts_clockwise(cv::Point2f* pts) {
    std::vector<cv::Point2f> out;
    out.push_back(pts[0]);
    out.push_back(pts[1]);
    out.push_back(pts[2]);
    out.push_back(pts[3]);
    std::sort(out.begin(), out.end(), compare_2fpts);
    std::vector<cv::Point2f> tmp = out;
    if(out[0].y > out[1].y) {
        pts[0] = tmp[1];
        pts[3] = tmp[0];
    } else {
        pts[0] = tmp[0];
        pts[3] = tmp[1];
    }
    if(out[2].y > out[3].y) {
        pts[1] = tmp[3];
        pts[2] = tmp[2];
    } else {
        pts[1] = tmp[2];
        pts[2] = tmp[3];
    }
}

}//end of name space
