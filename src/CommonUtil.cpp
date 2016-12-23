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

}//end of name space
