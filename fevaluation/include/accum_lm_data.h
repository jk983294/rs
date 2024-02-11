#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <zerg_file.h>

using namespace ztool;

namespace feval {
enum class RLMFileType : int32_t {
    None, Feather, Fst
};
struct DayData {
    std::vector<int>* x_ukeys{nullptr};
    std::vector<int>* x_dates{nullptr};
    std::vector<int>* x_ticks{nullptr};
    std::vector<std::vector<double>*> pXs;
    std::vector<std::string> xNames;
    std::vector<double> y;
    std::vector<bool> y_tradable;
    std::unordered_map<uint64_t, std::unordered_map<uint64_t, uint64_t>> ukey2tick2pos;

    void build_index();
};
struct DayMoment {
    DayMoment() = default;
    DayMoment(size_t x_len);
    void snap(size_t x_len, const std::vector<double>& XTX, const std::vector<double>& XTy, const std::vector<double>& XTX_intercept, double Xy_intercept);
    void accum(size_t x_len, std::vector<double>& XTX, std::vector<double>& XTy, std::vector<double>& XTX_intercept, double& Xy_intercept) const;
    void decum(size_t x_len, std::vector<double>& XTX, std::vector<double>& XTy, std::vector<double>& XTX_intercept, double& Xy_intercept) const;
    std::vector<double> m_XTX; // lower tri
    std::vector<double> m_XTy;
    std::vector<double> m_XTX_intercept;
    double m_Xy_intercept = 0;
};
struct X_Data {
    std::string x_path_pattern;
    std::string m_x_pattern;
    std::unordered_map<std::string, bool> m_x_names;
    RLMFileType m_x_type{RLMFileType::None};
    std::unordered_map<int, std::string> m_x2files;
    InputData id;
    InputData mid;
    void clear();
    uint64_t read(DayData& d, std::string path);
    void merge_read(DayData& d, std::string path);
    bool check_missing(const std::vector<int>& dates) const;
};

struct Y_Data {
    std::string m_path_pattern;
    std::string m_y_name;
    std::string m_tradable_name;
    RLMFileType m_y_type{RLMFileType::None};
    std::unordered_map<int, std::string> m_y2files;
    InputData id;
    void clear();
    void read(DayData& d, std::string path, uint64_t rows);
    bool check_missing(const std::vector<int>& dates) const;
};

RLMFileType get_file_type(std::string file_pattern);
}