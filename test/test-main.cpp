#include "TextUtil.h"

int main(int argc, char** argv) {
    std::string filepath = std::string(argv[1]);
    std::vector<std::string> filelist = text::get_filelist(filepath);
    for(unsigned int i = 0; i < filelist.size(); ++i) {
        std::cout << "file " << i << ": ";
        std::cout << filelist[i] << std::endl;
    }
    //bool success = text::load_boxes_from_file(filepath);
    return 0;
}
