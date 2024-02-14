#pragma once

#include <helper.h>
#include <zerg_fst.h>
#include <Eigen/Dense>
#include "accum_lm_data.h"

using namespace ztool;

namespace feval {
struct BarraTransformer {
    BarraTransformer();
    void work();
    void work_single(int date);
    void set_x(std::string x_file_pattern, std::string x_pattern, const std::vector<std::string>& x_names);
    void set_output(std::string output_file_pattern);
    void set_start_date(int v) { m_start_date = v; }
    void set_end_date(int v) { m_end_date = v; }

    X_Data m_x_data;
    InputData m_fp_id;
    string m_fp_dir;
    string m_output_dir;
    RLMFileType m_output_type{RLMFileType::None};
    int m_start_date{-1}, m_end_date{-1};
    std::vector<int>* fp_ukeys{nullptr};
    std::vector<int>* fp_dates{nullptr};
    std::vector<double>* fp_exposure{nullptr};
    std::vector<double>* fp_weight{nullptr};
    std::vector<std::string>* fp_factor{nullptr};
    std::vector<int> fp_factor_int;
    std::unordered_map<std::string, int> m_factor2idx;
    std::unordered_map<int, int> m_ukey2idx;
    std::vector<int> m_universe;
    Eigen::MatrixXd m_weight_mat;
    std::vector<std::vector<double>> m_ukey2exposure;
    std::vector<std::vector<double>*> m_f_features, m_b_features;
    std::string trading_day_file{"/dat/crawled/bao/trading_day.txt"};
    std::vector<int> all_dates;
    std::vector<int> dates;
    int threads{30};
    std::vector<std::string> barra_risk_indices;
    std::vector<std::string> barra_industries;
    std::vector<std::string> m_all_factor;

private:
    void clear_single();
    void arrange();
    bool load_fp(int date);
    void build_weight_matrix();
    void build_exposure();
    void build_universe(const DayData& d);
    void save(const DayData& d, int date);
};
}  // namespace feval