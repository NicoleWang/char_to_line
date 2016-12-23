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

    for(unsigned int i = 0; i < filelist.size(); ++i) {
        if (image_num != i) {
            //continue;
        }
        std::cout << "Processing " << i << "th image: " << " " << imglist[i] << std::endl;
        std::string imagepath = imgdir + "/" + imglist[i];
        std::string filepath = dirpath + "/" + imglist[i] + ".txt";
        cv::Mat img = cv::imread(imagepath.c_str(), 1);
        std::vector<text::TextChar> boxes = text::load_boxes_from_file(filepath);
        std::vector<text::TextChar> nms_boxes = text::nms_boxes(boxes, 0.7);
        cv::Mat centers = text::char_centers_to_mat(nms_boxes);
        text::TextLine line(img, nms_boxes);
        line.m_image_name = imglist[i];
        line.m_save_dir = savedir;
        line.gen_text_pairs();

        line.merge_text_pairs();
        line.gen_initial_lines();
        line.merge_initial_lines();
        line.vis_lines(line.m_initial_lines);
        //line.vis_pairs(line.m_final_pairs);
        //line.vis_pairs(line.m_pairs);
        //std::cout << centers << std::endl;
        //text::vis_boxes(img, nms_boxes);
        //std::string savepath = savedir + "/" + imglist[i];
        //cv::imwrite(savepath.c_str(), img);
        //for (unsigned int j = 0; j < boxes.size(); ++j) {
        //    std::cout << "rect: " << boxes[j].m_box.x << " " 
        //              << boxes[j].m_box.y << " " 
        //              << boxes[j].m_box.width << " "
        //              << boxes[j].m_box.height << " "
        //              << boxes[j].m_score << std::endl;
        //}
    }
    //bool success = text::load_boxes_from_file(filepath);
    return 0;
}
