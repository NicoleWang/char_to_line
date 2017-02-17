#include "DetectApi.h"
#include "CommonUtil.h"
using std::string;

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0]
              << " image_dir chars_dir"
              << " result_dir" << std::endl;
    return 1;
  }

  //::google::InitGoogleLogging(argv[0]);

  string imgdir  = argv[1];
  string charsdir  = argv[2];
  string savedir= argv[3];

  std::vector<std::string> imglist = common::get_filelist(imgdir);
  //process image one by one
  //std::ifstream infile(argv[3]);
  //std::string imagepath;
  for(unsigned int i = 0; i < imglist.size(); ++i) {
      std::cout << "Processing " << i << "th image: " << " " << imglist[i] << std::endl;
      std::string imagepath = imgdir + "/" + imglist[i];
      cv::Mat img = cv::imread(imagepath);
      //CHECK(!img.empty()) << "Unable to decode image" << imagepath;
      std::vector<cv::Mat> LineImgs;
      std::vector<cv::Mat> LineBins;
      std::vector<cv::Point> offsets;

      text::GetTextLine(img, LineImgs, LineBins, offsets, charsdir, savedir, imglist[i]);
     // std::cout<< "Line image num: " << LineImgs.size() << " Line bins size: " << LineBins.size() << std::endl;

#if 0
      std::vector<Box> dets;
      detector.Detect(img, dets);
      std::vector<text::TextChar> chars;
      //line generation
      caffe::TransBox2TextChar(dets, chars);
      std::cout << "Detect " << chars.size() << " chars " << std::endl;
      std::vector<text::TextChar> nms_boxes = text::nms_boxes(chars, 0.3);
      text::TextLine line(img, nms_boxes);
      line.m_direction = text::HOR_ONLY;
      line.m_image_name = imglist[i];
      line.m_save_dir = savedir;
      line.gen_text_pairs();
      //std::cout << "Gen text pairs done " << line.m_pairs.size() << " pairs" << std::endl;
      //line.merge_text_pairs_v2();
      line.merge_text_pairs();
      //std::cout << "Merge text pairs done " << line.m_pairs.size() << " pairs" << std::endl;
      line.gen_initial_lines();
      //std::cout << "Gen text lines done  " << line.m_initial_lines.size()  << " lines" << std::endl;
      line.merge_initial_lines();
      std::vector<cv::Mat> temp;
      line.crop_and_rotate_lines(temp);
      //std::cout << "Merge text lines done  " << line.m_final_lines.size()  << " lines" << std::endl;
      //line.vis_lines(line.m_final_lines);
      //line.vis_pairs(line.m_pairs);
      line.vis_pairs(line.m_final_pairs);
#endif
  }
}
