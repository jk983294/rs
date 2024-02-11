#include <helper.h>
#include <math_stats.h>
#include <math_stats_once.h>
#include <omp.h>
#include <zerg_fst.h>
#include <zerg_template.h>
#include <numeric>
#include <regex>

using namespace feval;
using namespace ztool;
using namespace ornate;
using std::string;

struct DailyY1 {
    struct FeatureCorrStats {
        rolling_all_once pcor;
        rolling_all_once rcor;
    };
    struct FeatureOnceStats {
        FeatureOnceStats() = default;
        FeatureOnceStats(int nY) {
            corr_stats.resize(nY);
        }
        std::vector<FeatureCorrStats> corr_stats;
        rolling_all_once f_;
    };

    void work() {
        if (threads == 0) threads = omp_get_max_threads();
        printf("thread num = %d\n", threads);
        omp_set_num_threads(threads);

        auto xs = path_wildcard(path_join(m_x_dir, "*.fst"));
        std::vector<std::pair<std::string, std::string>> todos;
        for (auto& item : xs) {
            int cob = std::stoi(item.first);
            if ((m_start_date > 0 && cob < m_start_date) || (m_end_date > 0 && cob > m_end_date)) {
                continue;
            } else {
                todos.emplace_back(item.first, item.second);
            }
        }

        std::sort(todos.begin(), todos.end(), [](auto& l, auto& r) { return l.first < r.first; });
        for (size_t i = 0; i < todos.size(); ++i) {
            auto& item = todos[i];
            printf("%s handle %s %zu/%zu\n", now_local_string().c_str(), item.second.c_str(), i, todos.size());
            std::cout << std::flush; // flush out
            work_single(item.first, item.second);
        }
        save_result();
    }

    void save_result();

    std::string m_x_dir;
    std::string m_output_dir{"./"};
    std::string m_y_pattern = "^y";
    std::string m_x_pattern;
    std::vector<std::string> m_yNames, m_xNames;
    size_t m_y_len{0};
    size_t m_x_len{0};
    int m_start_date{-1}, m_end_date{-1};
    int threads{0};
    std::vector<FeatureOnceStats> m_once_stats;

private:
    void work_single(const string& date_str, const string& path);
};

static void help() {
    std::cout << "Program options:" << std::endl;
    std::cout << "  -h                                    list help" << std::endl;
    std::cout << "  -x arg (=)                          dir of x fst files" << std::endl;
    std::cout << "  -o arg (=)                          output dir / reduce input dir" << std::endl;
    std::cout << "  -p arg (=^y)                          pattern of y" << std::endl;
    std::cout << "  -q arg (=)                          pattern of x" << std::endl;
    std::cout << "  -s arg (=-1)                          start date" << std::endl;
    std::cout << "  -e arg (=-1)                          end date" << std::endl;
    std::cout << "  -t arg (=0)                          thread num" << std::endl;
    printf("daily_y1 -q cneq_ -x ~/junk/y_eval/data/ -o ~/junk/y_eval/output/\n");
}

int main(int argc, char** argv) {
    DailyY1 dy;
    string config;
    int opt;
    while ((opt = getopt(argc, argv, "hx:p:q:s:e:t:o:")) != -1) {
        switch (opt) {
            case 'x':
                dy.m_x_dir = std::string(optarg);
                break;
            case 'o':
                dy.m_output_dir = std::string(optarg);
                break;
            case 'p':
                dy.m_y_pattern = std::string(optarg);
                break;
            case 'q':
                dy.m_x_pattern = std::string(optarg);
                break;
            case 's':
                dy.m_start_date = std::stoi(optarg);
                break;
            case 'e':
                dy.m_end_date = std::stoi(optarg);
                break;
            case 't':
                dy.threads = std::stoi(optarg);
                break;
            case 'h':
            default:
                help();
                return 0;
        }
    }

    dy.work();
    return 0;
}

void DailyY1::save_result() {
    std::vector<double> na_ratio(m_x_len, NAN);
    std::vector<double> mean_(m_x_len, NAN);
    std::vector<double> sd_(m_x_len, NAN);
    std::vector<double> skew(m_x_len, NAN);
    std::vector<double> kurt(m_x_len, NAN);
    std::vector<double> pos_ratio(m_x_len, NAN);
    std::vector<double> neg_ratio(m_x_len, NAN);
    std::vector<double> high(m_x_len, NAN);
    std::vector<double> low(m_x_len, NAN);
    std::vector<std::vector<double>> pcor(m_y_len, std::vector<double>(m_x_len, NAN));
    std::vector<std::vector<double>> rcor(m_y_len, std::vector<double>(m_x_len, NAN));
    for (uint64_t i = 0; i < m_x_len; ++i) {
        auto& stat = m_once_stats[i];
        na_ratio[i] = stat.f_.na_ratio();
        skew[i] = stat.f_.get_skew();
        kurt[i] = stat.f_.get_kurt();
        auto item = stat.f_.get_mean_sd();
        mean_[i] = item.first;
        sd_[i] = item.second;
        item = stat.f_.get_high_low();
        high[i] = item.first;
        low[i] = item.second;
        pos_ratio[i] = stat.f_.pos_ratio();
        neg_ratio[i] = stat.f_.neg_ratio();
        for (uint64_t j = 0; j < m_y_len; ++j) {
            pcor[j][i] = stat.corr_stats[j].pcor.get_mean_sd().first;
            rcor[j][i] = stat.corr_stats[j].rcor.get_mean_sd().first;
        }
    }

    std::vector<OutputColumnOption> options;
    options.push_back({4, &m_xNames, "x_name"});
    for (uint64_t j = 0; j < m_y_len; ++j) {
        options.push_back({1, pcor[j].data(), m_yNames[j] + "_pcor"});
        options.push_back({1, rcor[j].data(), m_yNames[j] + "_rcor"});
    }
    options.push_back({1, mean_.data(), "mean"});
    options.push_back({1, sd_.data(), "sd"});
    options.push_back({1, high.data(), "high"});
    options.push_back({1, low.data(), "low"});
    options.push_back({1, skew.data(), "skew"});
    options.push_back({1, kurt.data(), "kurt"});
    options.push_back({1, na_ratio.data(), "na_ratio"});
    options.push_back({1, pos_ratio.data(), "pos_ratio"});
    options.push_back({1, neg_ratio.data(), "neg_ratio"});
    write_fst(path_join(m_output_dir, "result.fst"), m_x_len, options);
}

void DailyY1::work_single(const string& date_str, const string& path) {
    FstReader x_reader;
    InputData id;
    x_reader.read(path, id);
    std::regex x_regex(m_x_pattern);
    std::regex y_regex(m_y_pattern);
    std::vector<int>* x_ukeys{nullptr};
    std::vector<int>* x_dates{nullptr};
    std::vector<int>* x_ticks{nullptr};
    std::vector<std::vector<double>*> pXs;
    std::vector<std::string> xNames;
    std::vector<std::vector<double>*> pYs;
    std::vector<std::string> yNames;
    for (auto& col : id.cols) {
        if (col.type == 1) {
            auto& vec = *reinterpret_cast<std::vector<double>*>(col.data);
            if (std::regex_search(col.name, x_regex)) {
                xNames.push_back(col.name);
                pXs.push_back(&vec);
                //printf("add x col %s for %s\n", col.name.c_str(), date_str.c_str());
            } else if (std::regex_search(col.name, y_regex)) {
                yNames.push_back(col.name);
                pYs.push_back(&vec);
                //printf("add y col %s for %s\n", col.name.c_str(), date_str.c_str());
            }
        } else if (col.type == 3) {
            auto& vec = *reinterpret_cast<std::vector<int>*>(col.data);
            if (col.name == "ukey")
                x_ukeys = &vec;
            else if (col.name == "ticktime")
                x_ticks = &vec;
            else if (col.name == "DataDate")
                x_dates = &vec;
        }
    }

    if (x_ukeys == nullptr || x_ticks == nullptr || x_dates == nullptr) {
        throw std::runtime_error("no ukey/tick/date column");
    }
    if (xNames.empty()) {
        throw std::runtime_error("empty x column");
    }
    if (yNames.empty()) {
        throw std::runtime_error("empty y column");
    }
    if (id.rows == 0) {
        printf("WARN! empty table %s\n", path.c_str());
        return;
    }

    if (m_xNames.empty()) {
        m_xNames = xNames;
        m_yNames = yNames;
        m_x_len = m_xNames.size();
        m_y_len = m_yNames.size();
        m_once_stats.resize(m_x_len, FeatureOnceStats(m_y_len));
    } else if (not is_identical(m_xNames, xNames)) {
        throw std::runtime_error("x column differ " + path);
    } else if (not is_identical(m_yNames, yNames)) {
        throw std::runtime_error("y column differ " + path);
    }

    std::unordered_map<int, std::vector<size_t>> tick2row_pos;
    for (uint64_t i = 0; i < id.rows; ++i) {
        tick2row_pos[(*x_ticks)[i]].push_back(i);
    }

#pragma omp parallel for
    for (uint64_t j = 0; j < m_x_len; ++j) {
        auto& stat = m_once_stats[j];
        auto& vec = *pXs[j];
        for (uint64_t i = 0; i < id.rows; ++i) {
            stat.f_(vec[i]);
        }

        for (uint64_t k = 0; k < m_y_len; ++k) {
            auto& vec_y = *pYs[k];

            for (auto& item : tick2row_pos) {
                auto& idxs = item.second;
                std::vector<double> xs(idxs.size(), NAN);
                std::vector<double> ys(idxs.size(), NAN);
                for (uint64_t p = 0; p < idxs.size(); ++p) {
                    xs[p] = vec[idxs[p]];
                    ys[p] = vec_y[idxs[p]];
                }

                stat.corr_stats[k].pcor(corr(xs, ys));
                stat.corr_stats[k].rcor(rcor(xs, ys));
            }
        }
    }
}

