#pragma once

//#define FEVAL_BUILD_WITH_CUDA

#include <vector>
#include <string>
#include <unordered_map>
#include <accum_lm_data.h>
#ifdef FEVAL_BUILD_WITH_CUDA
#include <feval_gpu_xx.h>
#endif

namespace feval {
struct AccumLM {
    void train();
    void fit(std::string x_file_pattern, std::string y_file_pattern, std::string x_pattern, std::string y_name,
        int start_date = -1, int end_date = -1);
    void clear();
    void export_y_hat(std::string file_);
    std::vector<double> get_beta() const { return m_betas.back(); }
    std::vector<std::vector<double>> get_betas() const { return m_betas; }
    std::vector<double> get_y() const { return m_y; }
    std::vector<double> get_y_hat() const { return m_y_hat; }
    std::vector<double> get_XTX() const { return m_XTX; }
    std::vector<double> get_XTy() const { return m_XTy; }
    std::vector<std::string> get_x_names() const { return m_xNames; }
    std::vector<int> get_dates() const { return dates; }
    void set_thread_num(int n) { threads = n; }
    void set_lambda(double v) { m_ridge_lambda = v; }
    void set_fit_intercept(bool v) { m_fit_intercept = v; }
    void set_use_cuda(bool v) { m_use_cuda = v; }
    void set_cuda_memory_size(size_t v) { m_cuda_memory_size = v; }
    void set_skip_days(size_t v) { m_skip_days = v; }
    void set_max_days(size_t v) { m_max_days = v; }
    void set_tradable_name(std::string v) { m_y_data.m_tradable_name = v; }
    void clear_x2_setting() { m_x2_datum.clear(); }
    void set_x(std::string x_file_pattern, std::string x_pattern, const std::vector<std::string>& x_names);
    void set_y(std::string y_file_pattern, std::string y_name);
    void set_rolling_days(int v) { m_rolling_days = v; }
    void set_start_date(int v) { m_start_date = v; }
    void set_end_date(int v) { m_end_date = v; }
    void set_enable_x2(bool v) { m_enable_x2 = v; }
    void add_x2_setting(std::string x2_path_pattern, std::string x2_pattern, const std::vector<std::string>& x2_cols);
    void set_eval_date(int start_date = -1, int end_date = -1) {
        m_eval_start_date = start_date;
        m_eval_end_date = end_date;
    }

private:
    void process(int day, size_t di);
    void arrage_groups();
    void calc_moment(DayData& d, size_t di);
    void calc_model(size_t di);
    void eval();
    size_t count_ticks(const int* src, size_t len);

public:
    bool m_fit_intercept{false};
    bool m_use_cuda{false};
    bool m_enable_x2{false};
    int m_start_date{-1}, m_end_date{-1};
    int m_eval_start_date{-1}, m_eval_end_date{-1};
    int threads{30};
    double m_ridge_lambda{0};
    double m_Xy_intercept{0};
    size_t m_rolling_days{1};
    size_t m_x_len{0};
    size_t m_n_rows{0};
    size_t m_n_ticks{0};
    size_t m_skip_days{1};
    size_t m_max_days{0};
    size_t m_real_max_days{0};
    size_t m_cuda_memory_size{1600 * 5000 * 240};
    std::vector<int> dates;
    std::vector<size_t> m_date2idx;
    std::vector<std::vector<int>> groups;
    std::vector<double> m_XTX_intercept;
    std::vector<double> m_XTX, part_XTX;
    std::vector<double> m_XTy;
    std::vector<double> m_y;
    std::vector<double> m_y_hat;
    std::vector<int> m_ukeys, m_DataDates, m_ticktimes;
    std::vector<std::vector<double>> m_betas;
    std::vector<std::string> m_xNames;
    X_Data m_x_data;
    Y_Data m_y_data;
    std::vector<DayMoment> m_moments;
    std::vector<X_Data> m_x2_datum;
    uint64_t m_total_time{0}, m_total_time1{0}, m_total_time2{0}, m_total_time3{0}, m_total_time4{0};
#ifdef FEVAL_BUILD_WITH_CUDA
    FevalGpuXX m_cuda;
#endif
};
}
