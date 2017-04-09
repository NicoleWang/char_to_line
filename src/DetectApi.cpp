#include "DetectApi.h"
#include "char_detector.hpp"
#include "TextUtil.h"
#include <fstream>
namespace text {
pthread_mutex_t det_mutex = PTHREAD_MUTEX_INITIALIZER;
caffe::Detector* Init(const std::string& proto_path, const std::string& model_path, const DetectParams& detParams) {
    caffe::Detector* detector = new caffe::Detector(proto_path, model_path, detParams.m_device_id);
    return detector;
}

caffe::Detector* Init(const std::string& proto_path, caffe::Detector& other, const DetectParams& detParams) {
    caffe::Net<float>* other_net = other.get_net();
    caffe::Detector* detector = new caffe::Detector(proto_path, other_net, detParams.m_device_id);
    return detector;
}
void Release(caffe::Detector*& detector) {
    if (NULL != detector) {
        delete detector;
        detector = NULL;
    }
}

bool reverse_bw(const cv::Mat& img) {
    int bg_num = 0;
    int fg_num = 0;
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            char px = img.at<char>(i, j);
            if(255 == px) {
                bg_num++;
            } else {
                fg_num++;
            }
        }
    }
    if (bg_num <= fg_num){
        return true;
    } else {
        return false;
    }
}

int GetTextLine(const cv::Mat& img,
                std::vector<cv::Mat>& LineImgs, 
                std::vector<cv::Mat>& LineBins,
                std::vector<cv::Point>& offsetPts,
                const std::string& chars_dir,
                const std::string& save_dir,
                const std::string& image_name) {
      std::string char_path = chars_dir + "/" + image_name +".txt";
      std::vector<text::TextChar> chars;
      chars = text::load_boxes_from_file(char_path);
      std::cout << "Load " << chars.size() << " chars " << std::endl;
      std::vector<text::TextChar> nms_boxes = chars;
      /*
      for (unsigned int i = 0; i < nms_boxes.size(); ++i) {
          std::cout << nms_boxes[i].m_box.x << " " 
                    << nms_boxes[i].m_box.y << " "
                    << nms_boxes[i].m_box.width << " "
                    << nms_boxes[i].m_box.height << std::endl;
      }
      */
      //std::vector<text::TextChar> nms_boxes = text::nms_boxes(chars, 0.3);
      text::TextLine line(img, nms_boxes);
#if 0
      cv::Mat vis_im = img.clone();
      vis_boxes(vis_im, nms_boxes);
      char save_name[256];
      sprintf(save_name, "%s/char_%s", save_dir.c_str(), image_name.c_str());
      imwrite(save_name, vis_im);
#endif
      line.m_direction = text::HOR_ONLY;
      line.m_image_name = image_name;
      line.m_save_dir = save_dir;
      line.gen_text_pairs();
      //std::cout << "Gen text pairs done " << line.m_pairs.size() << " pairs" << std::endl;
      line.merge_text_pairs();
      line.gen_initial_lines();
      line.merge_initial_lines();
      std::vector< std::vector<cv::Rect> > temp_char_pos;
      line.get_all_rotated_lines();
      
      line.crop_and_rotate_lines(LineImgs, temp_char_pos);

      for(unsigned int i = 0; i < LineImgs.size(); ++i) {
          cv::Mat im_gray;
          cv::cvtColor(LineImgs[i], im_gray, CV_RGB2GRAY);
          cv::Mat img_bw;
          cv::threshold(im_gray, img_bw, 128.0, 255.0, cv::THRESH_BINARY);
          LineBins.push_back(img_bw);
          char save_path[128];
          sprintf(save_path, "binary/%d_%s",i,image_name.c_str());
          cv::imwrite(save_path, img_bw);
          cv::Point pt(line.m_initial_lines[i].x, line.m_initial_lines[i].y);
          offsetPts.push_back(pt);
      }
      
      //line.vis_rotated_lines(line.m_rotated_lines);
      //line.vis_lines(line.m_initial_lines);
      std::fstream fn;
      std::string file_path;
      file_path = save_dir + "/" + image_name + ".txt";
      fn.open(file_path.c_str(), std::ios::out);
      for (unsigned int i = 0; i < line.m_initial_lines.size(); ++i) {
          fn << line.m_initial_lines[i].x << "\t" 
             << line.m_initial_lines[i].y << "\t"
             << line.m_initial_lines[i].x + line.m_initial_lines[i].width - 1<< "\t"
             << line.m_initial_lines[i].y + line.m_initial_lines[i].height - 1<< std::endl;
      }
      fn.close();
      //line.vis_pairs(line.m_pairs);
      //line.vis_pairs(line.m_final_pairs);
      return 0;
}

int GetTextLine(const cv::Mat& img,
                caffe::Detector* detector,
                std::vector<cv::Mat>& LineImgs, 
                std::vector<cv::Mat>& LineBins,
                std::vector<cv::Point>& offsetPts, 
                const std::string& save_dir,
                const std::string& image_name) {
      std::vector<caffe::Box> dets;
      detector->Detect(img, dets);
      std::vector<text::TextChar> chars;
      //line generation
      caffe::TransBox2TextChar(dets, chars);
      std::cout << "Detect " << chars.size() << " chars " << std::endl;
      std::vector<text::TextChar> nms_boxes = text::nms_boxes(chars, 0.5);
      text::TextLine line(img, nms_boxes);
#if 1
      cv::Mat vis_im = img.clone();
      vis_boxes(vis_im, nms_boxes);
      char save_name[256];
      sprintf(save_name, "%s/char_%s", save_dir.c_str(), image_name.c_str());
      imwrite(save_name, vis_im);
      std::cout << "Come here" << std::endl;
#endif
      line.m_direction = text::HOR_ONLY;
      line.m_image_name = image_name;
      line.m_save_dir = save_dir;
      line.gen_text_pairs();
      //std::cout << "Gen text pairs done " << line.m_pairs.size() << " pairs" << std::endl;
      line.merge_text_pairs();
      line.gen_initial_lines();
      line.merge_initial_lines();
      std::vector< std::vector<cv::Rect> > temp_char_pos;
      line.get_all_rotated_lines();
      /*
      line.crop_and_rotate_lines(LineImgs, temp_char_pos);

      for(unsigned int i = 0; i < LineImgs.size(); ++i) {
          cv::Mat im_gray;
          cv::cvtColor(LineImgs[i], im_gray, CV_RGB2GRAY);
          cv::Mat img_bw;
          cv::threshold(im_gray, img_bw, 128.0, 255.0, cv::THRESH_BINARY);
          LineBins.push_back(img_bw);
          char save_path[128];
          sprintf(save_path, "binary/%d_%s",i,image_name.c_str());
          cv::imwrite(save_path, img_bw);
          cv::Point pt(line.m_initial_lines[i].x, line.m_initial_lines[i].y);
          offsetPts.push_back(pt);
      }
      */

      line.vis_rotated_lines(line.m_rotated_lines);
      //line.vis_lines(line.m_initial_lines);
      //line.vis_pairs(line.m_pairs);
      //line.vis_pairs(line.m_final_pairs);
      return 0;
}

int GetTextLine(const cv::Mat& img,
                caffe::Detector* detector,
                std::vector<cv::Mat>& LineImgs, 
                std::vector<cv::Mat>& LineBins,
                std::vector<cv::Point>& offsetPts,
                std::vector< std::vector<cv::Rect> >& char_pos) {

      std::vector<caffe::Box> dets;
//      pthread_mutex_lock(&det_mutex);
      detector->Detect(img, dets);
//      pthread_mutex_unlock(&det_mutex);

      std::vector<text::TextChar> chars;
      //line generation
      caffe::TransBox2TextChar(dets, chars);
      //std::cout << "Detect " << chars.size() << " chars " << std::endl;
      std::vector<text::TextChar> nms_boxes = text::nms_boxes(chars, 0.5);
      text::TextLine line(img, nms_boxes);
      line.m_direction = text::HOR_ONLY;
      //line.m_image_name = imglist[i];
      //line.m_save_dir = savedir;
      line.gen_text_pairs();
      //std::cout << "Gen text pairs done " << line.m_pairs.size() << " pairs" << std::endl;
      line.merge_text_pairs();
      line.gen_initial_lines();
      line.merge_initial_lines();
      line.get_all_rotated_lines();
      line.crop_and_rotate_lines(LineImgs, char_pos);

      for(unsigned int i = 0; i < LineImgs.size(); ++i) {
          cv::Mat im_gray;
          cv::cvtColor(LineImgs[i], im_gray, CV_RGB2GRAY);
          cv::Mat img_bw;
          cv::threshold(im_gray, img_bw, 0, 255.0, cv::THRESH_OTSU);
          if (reverse_bw(img_bw)) {
              img_bw = 255 - img_bw;
          }

          LineBins.push_back(img_bw);
          cv::Point pt(line.m_initial_lines[i].x, line.m_initial_lines[i].y);
          offsetPts.push_back(pt);
      }
      //line.vis_lines(line.m_final_lines);
      //line.vis_pairs(line.m_pairs);
      //line.vis_pairs(line.m_final_pairs);
      
    return 0 ;
}

}//end of namespace
