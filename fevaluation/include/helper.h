#pragma once

#include <string>
#include <iostream>
#include <unistd.h>
#include <dirent.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <libgen.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <array>
#include <cstring>
#include <fstream>
#include <vector>
#include <unordered_map>

namespace feval {
inline void accumlate(const std::vector<double>& src, std::vector<double>& dest) {
    for (size_t i = 0; i < src.size(); i++) {
        dest[i] += src[i];
    }
}
inline void decumlate(const std::vector<double>& src, std::vector<double>& dest) {
    for (size_t i = 0; i < src.size(); i++) {
        dest[i] -= src[i];
    }
}

}