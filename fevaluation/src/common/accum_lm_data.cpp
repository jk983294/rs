#include <accum_lm_data.h>
#include <zerg_fst.h>
#include <zerg_feather.h>
#include <regex>
#include <helper.h>

using namespace ztool;

namespace feval {
DayMoment::DayMoment(size_t x_len) {
    m_XTX.resize(x_len * (x_len + 1) / 2);
    m_XTy.resize(x_len);
    m_XTX_intercept.resize(x_len + 1);
}

void DayMoment::snap(size_t x_len, const std::vector<double>& XTX, const std::vector<double>& XTy, const std::vector<double>& XTX_intercept, double Xy_intercept) {
    for (size_t i = 0; i < x_len; i++) {
        std::copy(XTX.begin() + i * x_len, XTX.begin() + i * x_len + (i + 1), m_XTX.data() + i * (i + 1) / 2);
    }
    
    std::copy(XTy.begin(), XTy.end(), m_XTy.data());
    std::copy(XTX_intercept.begin(), XTX_intercept.end(), m_XTX_intercept.data());
    m_Xy_intercept = Xy_intercept;
}
void DayMoment::accum(size_t x_len, std::vector<double>& XTX, std::vector<double>& XTy, std::vector<double>& XTX_intercept, double& Xy_intercept) const {
    //#pragma omp parallel for
    for (size_t i = 0; i < x_len; i++) {
        const auto* pd = m_XTX.data() + i * (i + 1) / 2;
        for (size_t j = 0; j <= i; j++) {
            if (i == j) {
                XTX[i * x_len + j] += pd[j];
            } else {
                XTX[i * x_len + j] += pd[j];
                XTX[j * x_len + i] += pd[j];
            }
        }
    }
    accumlate(m_XTy, XTy);
    accumlate(m_XTX_intercept, XTX_intercept);
    Xy_intercept += m_Xy_intercept;
}
void DayMoment::decum(size_t x_len, std::vector<double>& XTX, std::vector<double>& XTy, std::vector<double>& XTX_intercept, double& Xy_intercept) const {
    //#pragma omp parallel for
    for (size_t i = 0; i < x_len; i++) {
        const auto* pd = m_XTX.data() + i * (i + 1) / 2;
        for (size_t j = 0; j <= i; j++) {
            if (i == j) {
                XTX[i * x_len + j] -= pd[j];
            } else {
                XTX[i * x_len + j] -= pd[j];
                XTX[j * x_len + i] -= pd[j];
            }
        }
    }
    decumlate(m_XTy, XTy);
    decumlate(m_XTX_intercept, XTX_intercept);
    Xy_intercept -= m_Xy_intercept;
}

void X_Data::clear() {
    x_path_pattern = "";
    m_x_pattern = "";
    m_x_names.clear();
    m_x2files.clear();
    m_x_type = RLMFileType::None;
    id.clear();
    mid.clear();
}

bool X_Data::check_missing(const std::vector<int>& dates) const {
    for (int d : dates) {
        if (m_x2files.find(d) == m_x2files.end()) {
            // throw std::runtime_error(m_path_pattern + " missing date " + std::to_string(d));
            printf("%s missing date %d\n", m_x_pattern.c_str(), d);
            return false;
        }
    }
    return true;
}

uint64_t X_Data::read(DayData& d, std::string path) {
    id.clear();
    uint64_t rows{0};
    std::vector<OutputColumnOption>* cols{nullptr};
    if (m_x_type == RLMFileType::Fst) {
        FstReader x_reader;
        x_reader.read(path, id);
        rows = id.rows;
        cols = &id.cols;
    } else if (m_x_type == RLMFileType::Feather) {
        FeatherReader x_reader1;
        x_reader1.read(path, id);
        rows = id.rows;
        cols = &id.cols;
    }
    std::regex x_regex(m_x_pattern);

    for (auto& col : *cols) {
        if (col.type == 1) {
            auto& vec = *reinterpret_cast<std::vector<double>*>(col.data);
            if (m_x_names.find(col.name) != m_x_names.end() || (!m_x_pattern.empty() && std::regex_search(col.name, x_regex))) {
                d.xNames.push_back(col.name);
                d.pXs.push_back(&vec);
            }
        } else if (col.type == 3) {
            auto& vec = *reinterpret_cast<std::vector<int>*>(col.data);
            if (col.name == "ukey")
                d.x_ukeys = &vec;
            else if (col.name == "ticktime")
                d.x_ticks = &vec;
            else if (col.name == "DataDate")
                d.x_dates = &vec;
        }
    }

    if (d.x_ukeys == nullptr || d.x_ticks == nullptr || d.x_dates == nullptr) {
        throw std::runtime_error("no ukey/tick/date column " + path);
    }
    if (d.xNames.empty()) {
        throw std::runtime_error("empty x column " + path);
    }
    if (rows == 0) {
        printf("WARN! empty x table %s\n", path.c_str());
    }
    return rows;
}

void X_Data::merge_read(DayData& d, std::string path) {
    id.clear();
    uint64_t rows{0};
    std::vector<OutputColumnOption>* cols{nullptr};
    if (m_x_type == RLMFileType::Fst) {
        FstReader x_reader;
        x_reader.read(path, id);
        rows = id.rows;
        cols = &id.cols;
    } else if (m_x_type == RLMFileType::Feather) {
        FeatherReader x_reader1;
        x_reader1.read(path, id);
        rows = id.rows;
        cols = &id.cols;
    }
    std::regex x_regex(m_x_pattern);

    std::vector<int>* _ukeys{nullptr};
    std::vector<int>* _dates{nullptr};
    std::vector<int>* _ticks{nullptr};
    std::vector<OutputColumnOption> to_merges;
    for (auto& col : *cols) {
        if (col.type == 1) {
            if (m_x_names.find(col.name) != m_x_names.end() || (!m_x_pattern.empty() && std::regex_search(col.name, x_regex))) {
                if (std::find(d.xNames.begin(), d.xNames.end(), col.name) != d.xNames.end()) {
                    throw std::runtime_error("dupe column " + col.name + " in " + path);
                }
                to_merges.push_back(col);
            }
        } else if (col.type == 3) {
            auto& vec = *reinterpret_cast<std::vector<int>*>(col.data);
            if (col.name == "ukey")
                _ukeys = &vec;
            else if (col.name == "ticktime")
                _ticks = &vec;
            else if (col.name == "DataDate")
                _dates = &vec;
        }
    }

    if (_ukeys == nullptr || _ticks == nullptr || _dates == nullptr) {
        throw std::runtime_error("no ukey/tick/date column in " + path);
    }
    if (to_merges.empty()) {
        throw std::runtime_error("empty to_merges x column " + path);
    }
    if (rows == 0) {
        printf("WARN! empty to_merges x table %s\n", path.c_str());
        return;
    }

    mid.clear();
    for (auto& col : to_merges) {
        auto& vec = *reinterpret_cast<std::vector<double>*>(col.data);
        auto pVec = mid.new_double_vec(0);
        pVec->resize(d.x_dates->size(), NAN);
        uint64_t merge_cnt = 0;
        for (uint64_t i = 0; i < _ukeys->size(); ++i) {
            auto itr1 = d.ukey2tick2pos.find((*_ukeys)[i]);
            if (itr1 == d.ukey2tick2pos.end()) continue;
            auto& tick2pos = itr1->second;
            auto itr2 = tick2pos.find((*_ticks)[i]);
            if (itr2 == tick2pos.end()) continue;
            (*pVec)[itr2->second] = vec[i];
            merge_cnt++;
        }
        d.xNames.push_back(col.name);
        d.pXs.push_back(pVec);
        // printf("merge %s %zu,%zu\n", col.name.c_str(), merge_cnt, _ukeys->size());
    }
}

void Y_Data::clear() {
    m_path_pattern = "";
    m_y_name = "";
    m_tradable_name = "";
    m_y2files.clear();
    m_y_type = RLMFileType::None;
    id.clear();
}

bool Y_Data::check_missing(const std::vector<int>& dates) const {
    for (int d : dates) {
        if (m_y2files.find(d) == m_y2files.end()) {
            // throw std::runtime_error(m_path_pattern + " missing date " + std::to_string(d));
            printf("%s missing date %d\n", m_path_pattern.c_str(), d);
            return false;
        }
    }
    return true;
}

void Y_Data::read(DayData& d, std::string path, uint64_t rows) {
    id.clear();
    if (path.empty()) {
        printf("WARN! empty y file for %s\n", path.c_str());
        return;
    }
    std::vector<OutputColumnOption>* cols{nullptr};
    uint64_t y_rows = 0;
    if (m_y_type == RLMFileType::Fst) {
        FstReader y_reader;
        y_reader.read(path, id);
        cols = &id.cols;
        y_rows = id.rows;
    } else if (m_y_type == RLMFileType::Feather) {
        FeatherReader y_reader1;
        y_reader1.read(path, id);
        cols = &id.cols;
        y_rows = id.rows;
    }

    std::vector<int>* y_ukeys{nullptr};
    std::vector<int>* y_dates{nullptr};
    std::vector<int>* y_ticks{nullptr};
    std::vector<double>* py{nullptr};
    std::vector<double>* py_tradable{nullptr};
    for (auto& col : *cols) {
        if (col.type == 1) {
            auto& vec = *reinterpret_cast<std::vector<double>*>(col.data);
            if (col.name == m_y_name) {
                py = &vec;
            } else if (col.name == m_tradable_name) {
                py_tradable = &vec;
            }
        } else if (col.type == 3) {
            auto& vec = *reinterpret_cast<std::vector<int>*>(col.data);
            if (col.name == "ukey")
                y_ukeys = &vec;
            else if (col.name == "ticktime")
                y_ticks = &vec;
            else if (col.name == "DataDate")
                y_dates = &vec;
        }
    }
    if (y_ukeys == nullptr || y_ticks == nullptr || y_dates == nullptr || py == nullptr) {
        throw std::runtime_error("no ukey/tick/date column in y file " + path);
    }
    if (!m_tradable_name.empty() && py_tradable == nullptr) {
        throw std::runtime_error("no " + m_tradable_name + " in y file " + path);
    }
    d.y.resize(d.x_dates->size(), NAN);
    d.y_tradable.resize(d.x_dates->size(), true);
    size_t y_cnt = 0, good_y_cnt = 0, untradable = 0;
    for (uint64_t i = 0; i < y_rows; ++i) {
        auto itr1 = d.ukey2tick2pos.find((*y_ukeys)[i]);
        if (itr1 == d.ukey2tick2pos.end()) continue;
        auto& tick2pos = itr1->second;
        auto itr2 = tick2pos.find((*y_ticks)[i]);
        if (itr2 == tick2pos.end()) continue;
        d.y[itr2->second] = (*py)[i];
        if (py_tradable && (*py_tradable)[i] < 0.9) {
            d.y_tradable[itr2->second] = false;
            ++untradable;
        }
        y_cnt++;
        if (std::isfinite((*py)[i])) ++good_y_cnt;
    }
    // printf("read y %zu,%zu,%zu\n", y_cnt, good_y_cnt, untradable);
}

void DayData::build_index() {
    for (uint64_t i = 0; i < x_ukeys->size(); ++i) {
        ukey2tick2pos[(*x_ukeys)[i]][(*x_ticks)[i]] = i;
    }
}

RLMFileType get_file_type(std::string file_pattern) {
    if (end_with(file_pattern, "fst")) return RLMFileType::Fst;
    else if (end_with(file_pattern, "feather")) return RLMFileType::Feather;
    else return RLMFileType::None;
}
}