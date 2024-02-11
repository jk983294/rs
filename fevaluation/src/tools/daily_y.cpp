#include <zerg_fst.h>
#include <helper.h>
#include <regex>
#include <numeric>
#include <omp.h>
#include <zerg_template.h>
#include <math_stats_once.h>
#include <math_stats.h>

using namespace feval;
using namespace ztool;
using namespace ornate;
using std::string;

struct DailyY {
    struct DailyYOnceStats {
        rolling_all_once pcor;
        rolling_all_once pcor_pos;
        rolling_all_once rcor;
        rolling_all_once rcor_pos;
    };
    void reduce_work() {
        read_names();
        auto xs = path_wildcard(path_join(m_x_dir, "*.fst"));
        std::vector<std::pair<std::string, std::string>> todos;
        for (auto& item : xs) {
            if (item.first == "x_name" || item.first == "y_name") continue;
            else todos.emplace_back(item.first, item.second);
        }
        std::sort(todos.begin(), todos.end(), [](auto& l, auto& r) { return l.first < r.first; });
        for (size_t i = 0; i < todos.size(); ++i) {
            auto& item = todos[i];
            printf("%s handle %s %zu/%zu\n", now_local_string().c_str(), item.second.c_str(), i, todos.size());
            std::cout << std::flush; // flush out
            reduce_single(item.first, item.second);
        }
        save_reduce_result();
    }
    void work() {
        if (threads == 0) threads = omp_get_max_threads();
        printf("thread num=%d\n", threads);
        omp_set_num_threads(threads);

        if (m_reduce) {
            reduce_work();
            return;
        }

        read_y();
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

        m_date_len = todos.size();
        //reserve(m_date_len * m_ys.size() * 1000);
        std::sort(todos.begin(), todos.end(), [](auto& l, auto& r) { return l.first < r.first; });
        for (size_t i = 0; i < todos.size(); ++i) {
            auto& item = todos[i];
            printf("%s handle %s %zu/%zu\n", now_local_string().c_str(), item.second.c_str(), i, todos.size());
            std::cout << std::flush; // flush out
            work_single(item.first, item.second);
        }
        save_result();
    }

    void read_y();
    void save_result();
    void reserve(size_t len);
    void read_names();

    FstReader y_reader;
    InputData y_id;
    std::string m_y_path, m_x_dir;
    std::string m_output_dir{"./"};
    std::string m_y_pattern = "^y";
    std::string m_x_pattern;
    std::vector<int>* m_y_ukey{nullptr};
    std::vector<int>* m_y_date{nullptr};
    std::vector<int>* m_y_tick{nullptr};
    std::vector<std::vector<double>*> m_ys;
    std::vector<std::string> m_yNames, m_xNames;
    std::unordered_map<std::string, size_t> m_xNames2Idx;
    std::unordered_map<uint64_t, std::pair<size_t, size_t>> m_y_tick_date_pos;
    std::unordered_map<int, std::vector<size_t>> m_y_date2keys;
    size_t m_y_len{0};
    size_t m_x_len{0};
    size_t m_date_len{0};
    int m_start_date{-1}, m_end_date{-1};
    int threads{1};
    std::vector<int> m_result_date, m_result_tick, m_y_idx, m_x_idx;
    std::vector<double> m_result_pcor, m_result_pcor_pos, m_result_rcor, m_result_rcor_pos;
    bool m_reduce{false};
    std::vector<DailyYOnceStats> m_once_stats;

private:
    uint64_t get_key(int date, int tick) { return uint64_t(date) * 1000000000 + uint64_t(tick); };
    void work_single(const string& date_str, const string& path);
    void reduce_single(const string& date_str, const string& path);
    void save_reduce_result();
};

static void help() {
    std::cout << "Program options:" << std::endl;
    std::cout << "  -h                                    list help" << std::endl;
    std::cout << "  -x arg (=)                          dir of x fst files" << std::endl;
    std::cout << "  -y arg (=)                          path of y fst file" << std::endl;
    std::cout << "  -o arg (=)                          output dir / reduce input dir" << std::endl;
    std::cout << "  -p arg (=^y)                          pattern of y" << std::endl;
    std::cout << "  -q arg (=)                          pattern of x" << std::endl;
    std::cout << "  -s arg (=-1)                          start date" << std::endl;
    std::cout << "  -e arg (=-1)                          end date" << std::endl;
    std::cout << "  -t arg (=0)                          thread num" << std::endl;
    std::cout << "  -d arg (=false)                          reduce mode" << std::endl;
}

int main(int argc, char** argv) {
    DailyY dy;
    string config;
    int opt;
    while ((opt = getopt(argc, argv, "hx:y:p:q:s:e:t:o:d")) != -1) {
        switch (opt) {
            case 'x':
                dy.m_x_dir = std::string(optarg);
                break;
            case 'y':
                dy.m_y_path = std::string(optarg);
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
            case 'd':
                dy.m_reduce = true;
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

void DailyY::save_reduce_result() {
    size_t len = m_yNames.size() * m_xNames.size();
    std::vector<double> pcor_mean(len, NAN);
    std::vector<double> pcor_sd(len, NAN);
    std::vector<double> pcor_na(len, NAN);
    std::vector<double> pcor_pos_mean(len, NAN);
    std::vector<double> pcor_pos_sd(len, NAN);
    std::vector<double> pcor_pos_na(len, NAN);
    std::vector<double> rcor_mean(len, NAN);
    std::vector<double> rcor_sd(len, NAN);
    std::vector<double> rcor_na(len, NAN);
    std::vector<double> rcor_pos_mean(len, NAN);
    std::vector<double> rcor_pos_sd(len, NAN);
    std::vector<double> rcor_pos_na(len, NAN);
    std::vector<std::string> x_names(len);
    std::vector<std::string> y_names(len);
    for (size_t i = 0; i < m_yNames.size(); ++i) {
        for (size_t j = 0; j < m_xNames.size(); ++j) {
            size_t idx = i * m_xNames.size() + j;
            pcor_mean[idx] = m_once_stats[idx].pcor.get_mean_sd().first;
            pcor_sd[idx] = m_once_stats[idx].pcor.get_mean_sd().second;
            pcor_na[idx] = m_once_stats[idx].pcor.na_ratio();
            pcor_pos_mean[idx] = m_once_stats[idx].pcor_pos.get_mean_sd().first;
            pcor_pos_sd[idx] = m_once_stats[idx].pcor_pos.get_mean_sd().second;
            pcor_pos_na[idx] = m_once_stats[idx].pcor_pos.na_ratio();
            rcor_mean[idx] = m_once_stats[idx].rcor.get_mean_sd().first;
            rcor_sd[idx] = m_once_stats[idx].rcor.get_mean_sd().second;
            rcor_na[idx] = m_once_stats[idx].rcor.na_ratio();
            rcor_pos_mean[idx] = m_once_stats[idx].rcor_pos.get_mean_sd().first;
            rcor_pos_sd[idx] = m_once_stats[idx].rcor_pos.get_mean_sd().second;
            rcor_pos_na[idx] = m_once_stats[idx].rcor_pos.na_ratio();
            x_names[idx] = m_xNames[j];
            y_names[idx] = m_yNames[i];
        }
    }
    std::vector<OutputColumnOption> options;
    options.push_back({4, &x_names, "x_name"});
    options.push_back({4, &y_names, "y_name"});
    options.push_back({1, pcor_mean.data(), "pcor_mean"});
    options.push_back({1, pcor_sd.data(), "pcor_sd"});
    options.push_back({1, pcor_na.data(), "pcor_na"});
    options.push_back({1, pcor_pos_mean.data(), "pcor_pos_mean"});
    options.push_back({1, pcor_pos_sd.data(), "pcor_pos_sd"});
    options.push_back({1, pcor_pos_na.data(), "pcor_pos_na"});
    options.push_back({1, rcor_mean.data(), "rcor_mean"});
    options.push_back({1, rcor_sd.data(), "rcor_sd"});
    options.push_back({1, rcor_na.data(), "rcor_na"});
    options.push_back({1, rcor_pos_mean.data(), "rcor_pos_mean"});
    options.push_back({1, rcor_pos_sd.data(), "rcor_pos_sd"});
    options.push_back({1, rcor_pos_na.data(), "rcor_pos_na"});
    write_fst(path_join(m_output_dir, "reduced.fst"), pcor_mean.size(), options);
}

void DailyY::save_result() {
    std::vector<OutputColumnOption> options;
    options.push_back({3, m_result_date.data(), "DataDate"});
    options.push_back({3, m_result_tick.data(), "ticktime"});
    options.push_back({3, m_y_idx.data(), "y_idx"});
    options.push_back({3, m_x_idx.data(), "x_idx"});
    options.push_back({1, m_result_pcor.data(), "pcor"});
    options.push_back({1, m_result_pcor_pos.data(), "pcor_pos"});
    options.push_back({1, m_result_rcor.data(), "rcor"});
    options.push_back({1, m_result_rcor_pos.data(), "rcor_pos"});
    write_fst("result.fst", m_result_date.size(), options);

    options.clear();
    std::vector<int> seq(m_yNames.size());
    std::iota(seq.begin(), seq.end(), 0);
    options.push_back({3, seq.data(), "y_idx"});
    options.push_back({4, &m_yNames, "y_name"});
    write_fst("y_name.fst", m_yNames.size(), options);

    options.clear();
    std::vector<int> seq1(m_xNames.size());
    std::iota(seq1.begin(), seq1.end(), 0);
    options.push_back({3, seq1.data(), "x_idx"});
    options.push_back({4, &m_xNames, "x_name"});
    write_fst("x_name.fst", m_xNames.size(), options);
}

void DailyY::reduce_single(const string& name, const string& path) {
    FstReader reader;
    InputData id;
    reader.read(path, id);
    std::vector<int>* y_idx{nullptr};
    std::vector<int>* x_idx{nullptr};
    std::vector<double>* pcor{nullptr};
    std::vector<double>* pcor_pos{nullptr};
    std::vector<double>* rcor{nullptr};
    std::vector<double>* rcor_pos{nullptr};
    for (auto& col : id.cols) {
        if (col.type == 1) {
            auto& vec = *reinterpret_cast<std::vector<double>*>(col.data);
            if (col.name == "pcor") pcor = &vec;
            else if (col.name == "pcor_pos") pcor_pos = &vec;
            else if (col.name == "rcor") rcor = &vec;
            else if (col.name == "rcor_pos") rcor_pos = &vec;
        } else if (col.type == 3) {
            auto& vec = *reinterpret_cast<std::vector<int>*>(col.data);
            if (col.name == "y_idx") y_idx = &vec;
            else if (col.name == "x_idx") x_idx = &vec;
        }
    }

    if (y_idx == nullptr || x_idx == nullptr || pcor == nullptr|| pcor_pos == nullptr
        || rcor == nullptr|| rcor_pos == nullptr) {
        throw std::runtime_error("no required column");
    }

    for (size_t i = 0; i < id.rows; ++i) {
        size_t idx = (*y_idx)[i] * m_xNames.size() + (*x_idx)[i];
        m_once_stats[idx].pcor((*pcor)[i]);
        m_once_stats[idx].pcor_pos((*pcor_pos)[i]);
        m_once_stats[idx].rcor((*rcor)[i]);
        m_once_stats[idx].rcor_pos((*rcor_pos)[i]);
    }
}
void DailyY::work_single(const string& date_str, const string& path) {
    FstReader x_reader;
    InputData id;
    x_reader.read(path, id);
    std::regex x_regex(m_x_pattern);
    std::vector<int>* x_ukeys{nullptr};
    std::vector<int>* x_dates{nullptr};
    std::vector<int>* x_ticks{nullptr};
    std::vector<std::vector<double>*> pXs;
    std::vector<std::string> xNames;
    for (auto& col : id.cols) {
        if (col.type == 1) {
            auto& vec = *reinterpret_cast<std::vector<double>*>(col.data);
            if (std::regex_search(col.name, x_regex)) {
                xNames.push_back(col.name);
                pXs.push_back(&vec);
                // printf("add x col %s for %s\n", col.name.c_str(), date_str.c_str());
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
    if (id.rows == 0) {
        printf("WARN! empty table %s\n", path.c_str());
        return;
    }

    if (m_xNames.empty()) {
        m_xNames = xNames;
        for (size_t i = 0; i < m_xNames.size(); ++i) {
            m_xNames2Idx[m_xNames[i]] = i;
        }
    } else if (not is_identical(m_xNames, xNames)) {
        throw std::runtime_error("x column differ " + path);
    }

    if (m_x_len == 0) {
        m_x_len = m_xNames.size();
        reserve(m_date_len * m_ys.size() * m_x_len);
    }

    std::unordered_map<uint64_t, std::vector<size_t>> x_key2row_pos;
    for (uint64_t i = 0; i < id.rows; ++i) {
        uint64_t key = get_key((*x_dates)[i], (*x_ticks)[i]);
        x_key2row_pos[key].push_back(i);
    }

    int date = (*x_dates).front();
    auto itr = m_y_date2keys.find(date);
    if (itr == m_y_date2keys.end()) {
        printf("WARN! date %d not in y table %s\n", date, path.c_str());
        return;
    }
    const std::vector<size_t>& keys = itr->second;
    size_t key_len = keys.size();
    size_t cor_len = m_ys.size() * pXs.size();
    size_t total_len = key_len * cor_len;

    std::vector<int> result_date(total_len, date), result_tick(total_len), y_idx(total_len), x_idx(total_len);
    std::vector<double> result_pcor(total_len, NAN), result_pcor_pos(total_len, NAN);
    std::vector<double> result_rcor(total_len, NAN), result_rcor_pos(total_len, NAN);

#pragma omp parallel for
    for (size_t k = 0; k < key_len; ++k) {
        size_t key_ = keys[k];
        auto itr1 = m_y_tick_date_pos.find(key_); // must exist
        auto range = itr1->second;
        size_t y_start = range.first, y_end = range.second;

        auto itr2 = x_key2row_pos.find(key_);
        if (itr2 == x_key2row_pos.end()) {
            printf("WARN! no key %zu in x file %s\n", key_, path.c_str());
            continue;
        }
        const std::vector<size_t>& x_row_poses = itr2->second;

        std::vector<int> y_ukeys(m_y_ukey->data() + y_start, m_y_ukey->data() + y_end + 1);
        std::unordered_map<int, int> y_ukey2ii;
        for (size_t ii = 0; ii < y_ukeys.size(); ++ii) y_ukey2ii[y_ukeys[ii]] = ii;

        for (size_t i = 0; i < m_ys.size(); ++i) {
            std::vector<double> ys(m_ys[i]->data() + y_start, m_ys[i]->data() + y_end + 1);
            for (size_t j = 0; j < pXs.size(); ++j) {
                const std::vector<double>& all_xs = *pXs[j];
                std::vector<double> xs(ys.size(), NAN);

                for (auto x_row_pos_ : x_row_poses) {
                    int x_ukey = (*x_ukeys)[x_row_pos_];
                    auto itr3 = y_ukey2ii.find(x_ukey);
                    if (itr3 != y_ukey2ii.end()) {
                        xs[(size_t)itr3->second] = all_xs[x_row_pos_];
                    }
                }

                double pcor = ornate::corr(xs, ys);
                double pcor_pos = ornate::corr(xs, ys, 0, 1);
                double rcor = ornate::rcor(xs, ys);
                double rcor_pos = ornate::rcor(xs, ys, 0, 1);
                //printf("yi=%zu xi=%zu %d %zu pcor %f,%f,%f,%f\n", i, j, date, key_ % 1000000000, pcor, pcor_pos, rcor, rcor_pos);

                result_tick[k * cor_len + i * pXs.size() + j] = key_ % 1000000000;
                y_idx[k * cor_len + i * pXs.size() + j] = i;
                x_idx[k * cor_len + i * pXs.size() + j] = j;
                result_pcor[k * cor_len + i * pXs.size() + j] = pcor;
                result_pcor_pos[k * cor_len + i * pXs.size() + j] = pcor_pos;
                result_rcor[k * cor_len + i * pXs.size() + j] = rcor;
                result_rcor_pos[k * cor_len + i * pXs.size() + j] = rcor_pos;
            }
        }
    }

    m_result_date.insert(m_result_date.end(), result_date.begin(), result_date.end());
    m_result_tick.insert(m_result_tick.end(), result_tick.begin(), result_tick.end());
    m_y_idx.insert(m_y_idx.end(), y_idx.begin(), y_idx.end());
    m_x_idx.insert(m_x_idx.end(), x_idx.begin(), x_idx.end());
    m_result_pcor.insert(m_result_pcor.end(), result_pcor.begin(), result_pcor.end());
    m_result_pcor_pos.insert(m_result_pcor_pos.end(), result_pcor_pos.begin(), result_pcor_pos.end());
    m_result_rcor.insert(m_result_rcor.end(), result_rcor.begin(), result_rcor.end());
    m_result_rcor_pos.insert(m_result_rcor_pos.end(), result_rcor_pos.begin(), result_rcor_pos.end());
}

void DailyY::reserve(size_t len) {
    m_result_date.reserve(len);
    m_result_tick.reserve(len);
    m_y_idx.reserve(len);
    m_x_idx.reserve(len);
    m_result_pcor.reserve(len);
    m_result_pcor_pos.reserve(len);
    m_result_rcor.reserve(len);
    m_result_rcor_pos.reserve(len);
}

void DailyY::read_names() {
    FstReader reader;
    InputData id;
    reader.read(path_join(m_output_dir, "x_name.fst"), id);
    for (auto& col : id.cols) {
        if (col.name == "x_name") {
            auto& vec = *reinterpret_cast<std::vector<std::string>*>(col.data);
            m_xNames = vec;
        }
    }

    FstReader reader1;
    InputData id1;
    reader1.read(path_join(m_output_dir, "y_name.fst"), id1);
    for (auto& col : id1.cols) {
        if (col.name == "y_name") {
            auto& vec = *reinterpret_cast<std::vector<std::string>*>(col.data);
            m_yNames = vec;
        }
    }
    printf("read %zu y, %zu x\n", m_yNames.size(), m_xNames.size());
    m_once_stats.resize(m_yNames.size() * m_xNames.size());
}
void DailyY::read_y() {
    std::regex y_regex(m_y_pattern);
    y_reader.read(m_y_path, y_id);
    printf("load y rows=%zu\n", y_id.rows);
    for (auto& col : y_id.cols) {
        if (col.type == 1) {
            auto& vec = *reinterpret_cast<std::vector<double>*>(col.data);

            if (std::regex_search(col.name, y_regex)) {
                m_yNames.push_back(col.name);
                m_ys.push_back(&vec);
                printf("add y col %s\n", col.name.c_str());
            }
        } else if (col.type == 3) {
            auto& vec = *reinterpret_cast<std::vector<int>*>(col.data);
            if (col.name == "ukey") m_y_ukey = &vec;
            else if (col.name == "ticktime") m_y_tick = &vec;
            else if (col.name == "DataDate") m_y_date = &vec;
        }
    }

    if (m_y_ukey == nullptr || m_y_tick == nullptr || m_y_date == nullptr) {
        throw std::runtime_error("no ukey/tick/date column");
    }
    if (m_yNames.empty()) {
        throw std::runtime_error("empty y column");
    }

    m_y_tick_date_pos.reserve(100000);
    uint64_t last_key = 0;
    int last_date = 0;
    for (uint64_t i = 0; i < y_id.rows; ++i) {
        uint64_t key = get_key((*m_y_date)[i], (*m_y_tick)[i]);
        if (key < last_key && last_date == (*m_y_date)[i]) {
            throw std::runtime_error("y key not sorted " + std::to_string(key) + " <> " + std::to_string(last_key));
        }
        auto itr = m_y_tick_date_pos.find(key);
        if (itr == m_y_tick_date_pos.end()) {
            m_y_tick_date_pos[key] = {i, i};
            m_y_date2keys[(*m_y_date)[i]].push_back(key);
        } else {
            m_y_tick_date_pos[key].second = i;
        }
        last_key = key;
        last_date = (*m_y_date)[i];
    }
    m_y_len = m_y_tick_date_pos.size();
    printf("total y entry = %zu\n", m_y_len);
}