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
            boxes.push_back(TextChar(rect, score));
        }
    }
    infile.close();
    return boxes;
}
}
