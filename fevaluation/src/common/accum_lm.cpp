#include <accum_lm.h>
#include <zerg_fst.h>
#include <zerg_feather.h>
#include <zerg_template.h>
#include <helper.h>
#include <regex>
#include <omp.h>
#include <math_stats.h>
#include <Eigen/Dense>
#include <chrono>

using namespace std::chrono;
using namespace ztool;
using namespace ornate;


namespace feval {
void AccumLM::clear() {
    m_x_data.clear();
    m_y_data.clear();
    dates.clear();
    m_date2idx.clear();
    groups.clear();
    m_XTX.clear();
    m_XTy.clear();
    m_XTX_intercept.clear();
    m_y.clear();
    m_y_hat.clear();
    m_betas.clear();
    m_xNames.clear();
    m_ukeys.clear();
    m_DataDates.clear();
    m_ticktimes.clear();
    m_moments.clear();
    m_n_rows = 0;
    m_Xy_intercept = 0;
}
void AccumLM::set_x(std::string x_file_pattern, std::string x_pattern, const std::vector<std::string>& x_names) {
    for (auto& xn : x_names) m_x_data.m_x_names[xn] = true;
    m_x_data.m_x_pattern = x_pattern;
    m_x_data.x_path_pattern = x_file_pattern;
    m_x_data.m_x_type = get_file_type(x_file_pattern);
}
void AccumLM::set_y(std::string y_file_pattern, std::string y_name) {
    m_y_data.m_y_name = y_name;
    m_y_data.m_path_pattern = y_file_pattern;
    m_y_data.m_y_type = get_file_type(y_file_pattern);
}

void AccumLM::train() {
    if (threads <= 0) threads = omp_get_max_threads();
    else threads = std::min(threads, omp_get_max_threads());
    omp_set_num_threads(threads);
    arrage_groups();

    printf("thread num = %d, rolling_days=%zu, skip_days=%zu, max_days=%zu,%zu\n", threads, m_rolling_days, m_skip_days, m_max_days, m_real_max_days);
    printf("x=%s, y=%s\n", m_x_data.x_path_pattern.c_str(), m_y_data.m_path_pattern.c_str());
    printf("x_pattern=%s, y_name=%s, tradable=%s, x_cols_size=%zu\n",
        m_x_data.m_x_pattern.c_str(), m_y_data.m_y_name.c_str(), m_y_data.m_tradable_name.c_str(), m_x_data.m_x_names.size());
    printf("x_type=%d, y_type=%d\n", as_integer(m_x_data.m_x_type), as_integer(m_y_data.m_y_type));
    if (m_enable_x2) {
        for (const auto& x2_d : m_x2_datum) {
            printf("x2_path=%s, pattern=%s, x_cols_size=%zu, x_type=%d\n",
                x2_d.x_path_pattern.c_str(), x2_d.m_x_pattern.c_str(), x2_d.m_x_names.size(), as_integer(x2_d.m_x_type));
        }
    }
    printf("lambda=%f, fit_intercept=%d\n", m_ridge_lambda, m_fit_intercept);
    
    size_t di = 0;
    for (size_t i = 0; i < groups.size(); i++) {
        for (int day : groups[i]) {
            steady_clock::time_point t0 = steady_clock::now();
            process(day, di);
            steady_clock::time_point t1 = steady_clock::now();
            m_total_time += nanoseconds{t1 - t0}.count();
            m_date2idx[di] = m_y.size();
            printf("%s handle %d %zu/%zu\n", now_local_string().c_str(), day, di, dates.size());
            std::cout << std::flush; // flush out
            di++;
        }
        if (not groups[i].empty()) calc_model(di - 1);
    }
    eval();
#ifdef FEVAL_BUILD_WITH_CUDA
    m_cuda.release();
#endif
}

void AccumLM::eval() {
    if (dates.empty()) {
        printf("eval skip, empty dates\n");
        return;
    }
    size_t s1 = 0, s2 = 0;
    int d1 = m_eval_start_date, d2 = m_eval_end_date;
    if (d1 < 0) dates.front();
    if (d2 < 0) dates.back();
    auto itr = std::lower_bound(dates.begin(), dates.end(), d1);
    auto itr1 = std::lower_bound(dates.begin(), dates.end(), d2);
    if (itr == dates.end()) s1 = m_date2idx.back();
    else s1 = m_date2idx[itr - dates.begin()];
    if (itr1 == dates.end()) s2 = m_date2idx.back();
    else s2 = m_date2idx[itr1 - dates.begin()];
    double pcor = corr(m_y.data() + s1, m_y_hat.data() + s1, s2 - s1);
    double rcor_ = rcor(m_y.data() + s1, m_y_hat.data() + s1, s2 - s1);
    double rcor_pos = rcor(m_y.data() + s1, m_y_hat.data() + s1, s2 - s1, 1, 0);
    printf("s1=%zu, len=%zu, pcor=%f, rcor=%f, rcor_pos=%f\n", s1, s2 - s1, pcor, rcor_, rcor_pos);

    // double t0 = (double)m_total_time / 1000. / 1000. / dates.size();
    // double t1 = (double)m_total_time1 / 1000. / 1000. / dates.size();
    // double t2 = (double)m_total_time2 / 1000. / 1000. / dates.size();
    // double t3 = (double)m_total_time3 / 1000. / 1000. / dates.size();
    // double t4 = (double)m_total_time4 / 1000. / 1000. / dates.size();
    // printf("time %f,%f,%f,%f,%f, percent=%f,%f,%f,%f\n", t0, t1, t2, t3, t4, t1 / t0, t2 / t0, t3 / t0, t4 / t0);
}

void AccumLM::calc_model(size_t di) {
    if (m_fit_intercept) {
        std::vector<double> xtx_vec((m_x_len + 1) * (m_x_len + 1));
        for (size_t i = 0; i < m_x_len; i++) {
            std::copy(m_XTX.data() + (i * m_x_len), m_XTX.data() + ((i + 1) * m_x_len), xtx_vec.data() + (i * (m_x_len + 1)));
            xtx_vec[(i + 1) * (m_x_len + 1) - 1] = m_XTX_intercept[i];
        }
        std::copy(m_XTX_intercept.data(), m_XTX_intercept.data() + (m_x_len + 1), xtx_vec.data() + (m_x_len * (m_x_len + 1)));

        for (size_t i = 0; i < m_x_len + 1; i++) {
            xtx_vec[i * (m_x_len + 1) + i] += m_ridge_lambda;
        }

        std::vector<double> xty_vec = m_XTy;
        xty_vec.push_back(m_Xy_intercept);
        Eigen::Map<const Eigen::MatrixXd> xtx(xtx_vec.data(), m_x_len + 1, m_x_len + 1);
        Eigen::Map<const Eigen::VectorXd> xty(xty_vec.data(), m_x_len + 1);
        Eigen::VectorXd coef = xtx.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(xty);

        std::copy(coef.data(), coef.data() + m_x_len, m_betas[di].data());
        m_betas[di].back() = *(coef.data() + m_x_len);
    } else {
        std::vector<double> xtx_vec = m_XTX;
        for (size_t i = 0; i < m_x_len; i++) {
            xtx_vec[i * m_x_len + i] += m_ridge_lambda;
        }
        
        Eigen::Map<const Eigen::MatrixXd> xtx(xtx_vec.data(), m_x_len, m_x_len);
        Eigen::Map<const Eigen::VectorXd> xty(m_XTy.data(), m_x_len);
        Eigen::VectorXd coef = xtx.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(xty);
        std::copy(coef.data(), coef.data() + m_x_len, m_betas[di].data());
        m_betas[di].back() = 0;
    }
    
    double pcor = corr(m_y, m_y_hat);
    printf("%s calc_model, pcor=%f\n", now_local_string().c_str(), pcor);
}

size_t AccumLM::count_ticks(const int* src, size_t len) {
    std::unordered_map<int, bool> cnt;
    for (size_t i = 0; i < len; i++) {
        cnt[src[i]] = true;
    }
    m_n_ticks = cnt.size();
    return m_n_ticks;
}

void AccumLM::calc_moment(DayData& d, size_t di) {
    size_t x_len = d.xNames.size();
    if (m_x_len == 0) {
        m_x_len = x_len;
        m_xNames = d.xNames;
        m_XTX.resize(m_x_len * m_x_len, 0);
        part_XTX = m_XTX;
        m_XTy.resize(m_x_len, 0);
        m_XTX_intercept.resize(m_x_len + 1, 0);
        m_y.reserve(dates.size() * d.y.size());
        m_y_hat.reserve(dates.size() * d.y.size());
        m_DataDates.reserve(dates.size() * d.y.size());
        m_ticktimes.reserve(dates.size() * d.y.size());
        m_ukeys.reserve(dates.size() * d.y.size());
        m_Xy_intercept = 0;
        m_betas.resize(dates.size(), std::vector<double>(m_x_len + 1, NAN));
        count_ticks(d.x_ticks->data(), d.x_ticks->size());
        if (m_n_ticks == 0) {
            throw std::runtime_error("n_ticks error!");
        }
#ifdef FEVAL_BUILD_WITH_CUDA
        if (m_use_cuda) {
            if (not m_cuda.init(m_x_len, m_cuda_memory_size)) {
                throw std::runtime_error("cuda memory not enough!");
            }
        }
#endif
        if (m_real_max_days > m_moments.size()) {
            m_moments.resize(m_real_max_days, DayMoment(m_x_len));
        }
        printf("x_len=%zu n_ticks=%zu\n", x_len, m_n_ticks);
    } else if (not is_identical(m_xNames, d.xNames)) {
        throw std::runtime_error("x column differ!");
    }

    std::vector<double> XTX(m_XTX.size(), 0);
    std::vector<double> XTy(m_XTy.size(), 0);
    std::vector<double> XTX_intercept(m_XTX_intercept.size(), 0);
    double Xy_intercept = 0;

    size_t _len = d.y.size();
    size_t good_len = 0;
    std::vector<bool> is_y_good(_len, true);
    for (size_t ni = 0; ni < _len; ni++) {
        if (!std::isfinite(d.y[ni]) || !d.y_tradable[ni]) {
            is_y_good[ni] = false;
            d.y[ni] = 0;
        } else {
            good_len++;
            Xy_intercept += d.y[ni];
        }
    }

#pragma omp parallel for
    for (size_t xi = 0; xi < x_len; xi++) {
        auto& vec = (*d.pXs[xi]);
        for (size_t ni = 0; ni < _len; ni++) {
            if (is_y_good[ni]) {
                if (not std::isfinite(vec[ni])) {
                    vec[ni] = 0;
                }
            } else {
                vec[ni] = 0;
            }
        }
    }

#pragma omp parallel for
    for (size_t xi = 0; xi < x_len; xi++) {
        auto& vec = (*d.pXs[xi]);
        double val = 0;
        for (size_t ni = 0; ni < _len; ni++) {
            val += vec[ni] * d.y[ni];
        }
        XTy[xi] = val;
    }

    steady_clock::time_point t0 = steady_clock::now();
#ifdef FEVAL_BUILD_WITH_CUDA
    if (m_use_cuda) {
        uint64_t cuda_col_size = m_cuda_memory_size / x_len;
        uint64_t times = _len / cuda_col_size + 1;
        uint64_t one_len = _len / times + 1;
        if (_len % times == 0) one_len = _len / times;
        for (size_t i = 0; i < times; i++) {
            uint64_t offset = one_len * i;
            uint64_t curr_len = one_len;
            if (offset + curr_len > _len) {
                curr_len = _len - offset;
            }
            // printf("xx _len=%zu, x_len=%zu, offset=%zu, len=%zu\n", _len, x_len, offset, curr_len);
            if (curr_len > 0) {
                m_cuda.calc(d.pXs, part_XTX.data(), offset, curr_len);
                accumlate(part_XTX, XTX);
                // std::fill(XTX.begin(), XTX.end(), 0);
            }
        }
    } else
#endif
    {
        #pragma omp parallel for
        for (size_t xi = 0; xi < x_len; xi++) {
            auto& vec = (*d.pXs[xi]);
            for (size_t xi1 = 0; xi1 <= xi; xi1++) {
                auto& vec1 = (*d.pXs[xi1]);
                double val = 0;
                for (size_t ni = 0; ni < _len; ni++) {
                    val += vec[ni] * vec1[ni];
                }
                XTX[xi * x_len + xi1] = val;
                XTX[xi1 * x_len + xi] = val;
            }
        }
    }

    steady_clock::time_point t1 = steady_clock::now();
    m_total_time4 += nanoseconds{t1 - t0}.count();
    
    accumlate(XTX, m_XTX);
    accumlate(XTy, m_XTy);
    if (m_fit_intercept) {
        #pragma omp parallel for
        for (size_t xi = 0; xi < x_len; xi++) {
            auto& vec1 = (*d.pXs[xi]);
            double val = 0;
            for (size_t ni = 0; ni < _len; ni++) {
                val += vec1[ni];
            }
            XTX_intercept[xi] = val;
        }
        XTX_intercept[x_len] = good_len;
        accumlate(XTX_intercept, m_XTX_intercept);
        m_Xy_intercept += Xy_intercept;
    }

    if (m_real_max_days > 0) {
        size_t curr_idx = di % m_real_max_days;
        if (di >= m_real_max_days) {
            m_moments[curr_idx].decum(x_len, m_XTX, m_XTy, m_XTX_intercept, m_Xy_intercept);
        }
        m_moments[curr_idx].snap(x_len, XTX, XTy, XTX_intercept, Xy_intercept);
    }

    if (di > 0) {
        m_betas[di] = m_betas[di - 1];
    }
    std::vector<double> y_hat(d.y.size(), NAN);
    if (di >= m_skip_days) {
        const auto _coefs = m_betas[di - m_skip_days];
#pragma omp parallel for
        for (size_t ni = 0; ni < _len; ni++) {
            double val = 0;
            for (size_t xi = 0; xi < x_len; xi++) {
                val += (*d.pXs[xi])[ni] * _coefs[xi];
            }
            y_hat[ni] = val + _coefs.back();
        }
    }
    m_ukeys.insert(m_ukeys.end(), d.x_ukeys->begin(), d.x_ukeys->end());
    m_DataDates.insert(m_DataDates.end(), d.x_dates->begin(), d.x_dates->end());
    m_ticktimes.insert(m_ticktimes.end(), d.x_ticks->begin(), d.x_ticks->end());
    for (size_t ni = 0; ni < _len; ni++) {
        if (not is_y_good[ni]) {
            d.y[ni] = NAN;
            y_hat[ni] = NAN;
        }
    }
    m_y.insert(m_y.end(), d.y.begin(), d.y.end());
    m_y_hat.insert(m_y_hat.end(), y_hat.begin(), y_hat.end());
    m_n_rows += good_len;
}

void AccumLM::arrage_groups() {
    std::unordered_map<std::string, std::string> xs = path_wildcard(path_join(m_x_data.x_path_pattern));

    for (auto& item : xs) {
        int cob = std::stoi(item.first);
        if ((m_start_date > 0 && cob < m_start_date) || (m_end_date > 0 && cob > m_end_date)) {
            continue;
        } else {
            dates.emplace_back(cob);
            m_x_data.m_x2files[cob] = item.second;
        }
    }
    std::sort(dates.begin(), dates.end());
    m_date2idx.resize(dates.size(), 0);

    size_t start_ = 0;
    while(start_ < dates.size()) {
        size_t end_ = start_ + m_rolling_days;
        if (end_ > dates.size()) end_ = dates.size();
        groups.push_back(std::vector<int>(dates.begin() + start_, dates.begin() + end_));
        start_ = end_;
        // printf("group %d %d\n", groups.back().front(), groups.back().back());
    }

    std::unordered_map<std::string, std::string> ys = path_wildcard(path_join(m_y_data.m_path_pattern));
    for (auto& item : ys) {
        int cob = std::stoi(item.first);
        if (m_x_data.m_x2files.find(cob) != m_x_data.m_x2files.end()) {
            m_y_data.m_y2files[cob] = item.second;
            // printf("add y %d %s\n", cob, item.second.c_str());
        }
    }
    if (!m_y_data.check_missing(dates)) {
        throw std::runtime_error(m_y_data.m_path_pattern + " missing date");
    }

    if (m_enable_x2) {
        for (auto& x2_d : m_x2_datum) {
            std::unordered_map<std::string, std::string> xs = path_wildcard(path_join(x2_d.x_path_pattern));
            for (auto& item : xs) {
                int cob = std::stoi(item.first);
                if (m_x_data.m_x2files.find(cob) != m_x_data.m_x2files.end()) {
                    x2_d.m_x2files[cob] = item.second;
                }
            }
            if (!x2_d.check_missing(dates)) {
                throw std::runtime_error(x2_d.m_x_pattern + " missing date");
            }
        }
    }

    if (m_max_days > 0) {
        if (m_max_days >= dates.size()) m_real_max_days = 0;
        else m_real_max_days = m_max_days;
    } else m_real_max_days = 0;
}

void AccumLM::process(int day, size_t di) {
    DayData d;
    steady_clock::time_point t0 = steady_clock::now();
    uint64_t rows = m_x_data.read(d, m_x_data.m_x2files[day]);
    d.build_index();
    if (m_enable_x2) {
        for (auto& x2_d : m_x2_datum) {
            x2_d.merge_read(d, x2_d.m_x2files[day]);
        }
    }
    steady_clock::time_point t1 = steady_clock::now();
    m_total_time1 += nanoseconds{t1 - t0}.count();
    if (rows == 0) return;

    m_y_data.read(d, m_y_data.m_y2files[day], rows);
    steady_clock::time_point t2 = steady_clock::now();
    calc_moment(d, di);
    steady_clock::time_point t3 = steady_clock::now();
     m_total_time2 += nanoseconds{t2 - t1}.count();
     m_total_time3 += nanoseconds{t3 - t2}.count();
}

void AccumLM::fit(std::string x_file_pattern, std::string y_file_pattern, std::string x_pattern, std::string y_name,
    int start_date, int end_date) {
    X_Data tmp_x_data;
    tmp_x_data.m_x_pattern = x_pattern;
    tmp_x_data.x_path_pattern = x_file_pattern;
    tmp_x_data.m_x_type = get_file_type(x_file_pattern);
    Y_Data tmp_y_data;
    tmp_y_data.m_path_pattern = y_file_pattern;
    tmp_y_data.m_y_name = y_name;

    std::vector<int> tmp_dates;
    std::unordered_map<std::string, std::string> xs = path_wildcard(path_join(x_file_pattern));
    for (auto& item : xs) {
        int cob = std::stoi(item.first);
        if ((start_date > 0 && cob < start_date) || (end_date > 0 && cob > end_date)) {
            continue;
        } else {
            tmp_dates.emplace_back(cob);
            tmp_x_data.m_x2files[cob] = item.second;
        }
    }
    std::sort(tmp_dates.begin(), tmp_dates.end());
    std::unordered_map<std::string, std::string> ys = path_wildcard(path_join(y_file_pattern));
    for (auto& item : ys) {
        int cob = std::stoi(item.first);
        if (tmp_x_data.m_x2files.find(cob) != tmp_x_data.m_x2files.end()) {
            tmp_y_data.m_y2files[cob] = item.second;
        }
    }

    std::vector<double> tmp_y;
    std::vector<double> tmp_y_hat;
    size_t cnt = 0;
    for (int day : tmp_dates) {
        printf("%s handle %d %zu/%zu\n", now_local_string().c_str(), day, cnt, tmp_dates.size());
        std::cout << std::flush; // flush out
        DayData d;
        uint64_t rows = tmp_x_data.read(d, tmp_x_data.m_x2files[day]);
        if (rows == 0) continue;
        tmp_y_data.read(d, tmp_y_data.m_y2files[day], rows);

        size_t x_len = d.xNames.size();
        if (not is_identical(m_xNames, d.xNames)) {
            throw std::runtime_error("x column differ!");
        }

        size_t _len = d.y.size();
        std::vector<bool> is_y_good(_len, true);
        for (size_t ni = 0; ni < _len; ni++) {
            if (not std::isfinite(d.y[ni])) {
                is_y_good[ni] = false;
            }
        }

        #pragma omp parallel for
        for (size_t xi = 0; xi < x_len; xi++) {
            auto& vec = (*d.pXs[xi]);
            for (size_t ni = 0; ni < _len; ni++) {
                if (is_y_good[ni]) {
                    if (not std::isfinite(vec[ni])) {
                        vec[ni] = 0;
                    }
                }
            }
        }

        tmp_y.insert(tmp_y.end(), d.y.begin(), d.y.end());
        std::vector<double> y_hat(d.y.size(), NAN);
        if (not m_betas.empty()) {
            const auto _coefs = m_betas.back();
            #pragma omp parallel for
            for (size_t ni = 0; ni < _len; ni++) {
                if (is_y_good[ni]) {
                    double val = 0;
                    for (size_t xi = 0; xi < x_len; xi++) {
                        val += (*d.pXs[xi])[ni] * _coefs[xi];
                    }
                    val += _coefs.back();
                    y_hat[ni] = val;
                }
            }
        }
        tmp_y_hat.insert(tmp_y_hat.end(), y_hat.begin(), y_hat.end());
        cnt++;
    }
    
    double pcor = corr(tmp_y, tmp_y_hat);
    double rcor_ = rcor(tmp_y, tmp_y_hat);
    double rcor_pos = rcor(tmp_y, tmp_y_hat, 1, 1);
    printf("pcor=%f, rcor=%f, rcor_pos=%f\n", pcor, rcor_, rcor_pos);
}

void AccumLM::export_y_hat(std::string file_) {
    std::vector<OutputColumnOption> options;
    options.push_back({3, m_ukeys.data(), "ukey"});
    options.push_back({3, m_ticktimes.data(), "ticktime"});
    options.push_back({3, m_DataDates.data(), "DataDate"});
    // options.push_back({1, m_y.data(), "y"});
    options.push_back({1, m_y_hat.data(), "y_hat"});
    if (end_with(file_, "feather")) {
        write_feather(file_, m_ukeys.size(), options);
    } else if (end_with(file_, "fst")) {
        write_fst(file_, m_ukeys.size(), options);
    } else {
        printf("export_y_hat failed due to unknown file type.\n");
    }
}

void AccumLM::add_x2_setting(std::string x2_path_pattern, std::string x2_pattern, const std::vector<std::string>& x2_cols) {
    m_x2_datum.push_back({});
    auto& x2_data = m_x2_datum.back();
    x2_data.x_path_pattern = x2_path_pattern;
    x2_data.m_x_pattern = x2_pattern;
    x2_data.m_x_type = get_file_type(x2_path_pattern);
    for (auto& xn : x2_cols) x2_data.m_x_names[xn] = true;
}
}