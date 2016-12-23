/*
 * Brief: Utils for most commonly used functions
 * Author: wangyuzhuo@bytedance.com
 * Date: 2016/12/21 15:12
 */

#ifndef _COMMONUTIL_H_
#define _COMMONUTIL_H_

#include <vector>
#include <string>

namespace common {
std::vector<std::string> get_filelist(const std::string& dir_name);
std::string get_name_prefix(const std::string& name);
}//end of namespace
#endif
