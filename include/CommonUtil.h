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
        inline float compute_y(float x) {
            return (m_a + m_b * x);
        }
        void print_pts();
        
        float m_a;
        float m_b;
        std::vector<CType> m_pts;
};


std::vector<std::string> get_filelist(const std::string& dir_name);
std::string get_name_prefix(const std::string& name);
}//end of namespace
#endif
