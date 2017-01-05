#include <stdlib.h>
#include "CommonUtil.h"
#include "TextCore.h"
#include "TextUtil.h"

int main(int argc, char** argv) {
    std::string imgdir = std::string(argv[1]);
    std::string dirpath = std::string(argv[2]);
    std::string savedir= std::string(argv[3]);
    int image_num = atoi(argv[4]);

    std::vector<std::string> filelist = common::get_filelist(dirpath);
    std::vector<std::string> imglist = common::get_filelist(imgdir);

    std::cout << "Processing " << image_num << "th image: " << " " << imglist[image_num] << std::endl;
    int i = image_num;
    std::string imagepath = imgdir + "/" + imglist[i];
    std::string filepath = dirpath + "/" + imglist[i] + ".txt";
    cv::Mat img = cv::imread(imagepath.c_str(), 1);
    std::vector<text::TextChar> boxes = text::load_boxes_from_file(filepath);
    std::vector<text::TextChar> nms_boxes = text::nms_boxes(boxes, 0.7);
    text::TextLine line(img, nms_boxes);
    line.m_image_name = imglist[i];
    line.m_save_dir = savedir;
    line.m_direction = text::HOR_ONLY;
    line.gen_text_pairs();
    std::cout << "Gen text pairs done" << std::endl;
    line.merge_text_pairs();
    line.gen_initial_lines();
    //line.merge_initial_lines();
    line.vis_lines(line.m_initial_lines);
    //line.vis_pairs(line.m_final_pairs);
    return 0;
}
