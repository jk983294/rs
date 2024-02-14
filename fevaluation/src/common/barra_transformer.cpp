#include <barra_transformer.h>
#include <omp.h>
#include <zerg_template.h>

namespace feval {
void BarraTransformer::work() {
    if (threads <= 0)
        threads = omp_get_max_threads();
    else
        threads = std::min(threads, omp_get_max_threads());
    omp_set_num_threads(threads);
    ztool::read_trading_days(trading_day_file, all_dates);
    if (all_dates.empty()) throw std::runtime_error("empty trading_day_file " + trading_day_file);

    arrange();
    if (!ztool::IsFileExisted(m_output_dir)) {
        ztool::mkdirp(m_output_dir, 0777);
    }

    printf("thread num=%d\n", threads);
    printf("x=%s, x_pattern=%s, x_cols_size=%zu, x_type=%d\n", m_x_data.x_path_pattern.c_str(),
           m_x_data.m_x_pattern.c_str(), m_x_data.m_x_names.size(), as_integer(m_x_data.m_x_type));
    printf("output=%s, output_type=%d\n", m_output_dir.c_str(), as_integer(m_output_type));

    size_t di = 0;
    for (int date : dates) {
        work_single(date);
        printf("%s handle %d %zu/%zu\n", now_local_string().c_str(), date, di, dates.size());
        di++;
    }
}

void BarraTransformer::clear_single() {
    fp_ukeys = nullptr;
    fp_dates = nullptr;
    fp_exposure = nullptr;
    fp_weight = nullptr;
    fp_factor = nullptr;
    m_factor2idx.clear();
    m_ukey2idx.clear();
    m_universe.clear();
    fp_factor_int.clear();
    m_fp_id.clear();
    m_x_data.id.clear();
    m_x_data.mid.clear();
    m_ukey2exposure.clear();
    m_f_features.clear();
    m_b_features.clear();
}
void BarraTransformer::work_single(int date) {
    clear_single();
    if (not load_fp(date)) return;

    DayData d;
    m_x_data.read(d, m_x_data.m_x2files[date]);

    build_universe(d);
    build_weight_matrix();
    build_exposure();

    m_f_features.resize(d.pXs.size(), nullptr);
    m_b_features.resize(d.pXs.size(), nullptr);

    std::unordered_map<int, std::vector<size_t>> tick2pos;
    for (size_t i = 0; i < d.x_ticks->size(); ++i) {
        tick2pos[(*d.x_ticks)[i]].push_back(i);
    }

#pragma omp parallel for
    for (size_t i = 0; i < d.pXs.size(); ++i) {
        auto feature_ret = d.pXs[i];
        auto f_f = m_x_data.id.new_double_vec(d.x_ukeys->size());
        auto b_f = m_x_data.id.new_double_vec(d.x_ukeys->size());
        std::fill(f_f->begin(), f_f->end(), NAN);
        std::fill(b_f->begin(), b_f->end(), NAN);
        m_f_features[i] = f_f;
        m_b_features[i] = b_f;

        for (auto& item : tick2pos) {
            const std::vector<size_t>& idxes = item.second;
            Eigen::VectorXd feature(m_universe.size());
            feature.setZero();
            for (auto idx : idxes) {
                int ukey = (*d.x_ukeys)[idx];
                int univ_idx = m_ukey2idx[ukey];
                feature[univ_idx] = (*feature_ret)[idx];
            }

            Eigen::VectorXd F = feature.transpose() * m_weight_mat;

            for (auto idx : idxes) {
                int ukey = (*d.x_ukeys)[idx];
                int univ_idx = m_ukey2idx[ukey];
                const auto& exposure_ = m_ukey2exposure[univ_idx];
                Eigen::Map<const Eigen::VectorXd> exposure_vec(exposure_.data(), m_all_factor.size());
                double f_ret_val = (exposure_vec.array() * F.array()).sum();
                (*f_f)[idx] = f_ret_val;
                (*b_f)[idx] = (*feature_ret)[idx] - f_ret_val;
            }
        }
    }

    save(d, date);
}

void BarraTransformer::build_exposure() {
    m_ukey2exposure.resize(m_universe.size(), std::vector<double>(m_all_factor.size(), 0));
    for (size_t i = 0; i < fp_exposure->size(); ++i) {
        int ukey = (*fp_ukeys)[i];
        auto itr = m_ukey2idx.find(ukey);
        if (itr == m_ukey2idx.end()) continue;
        int ukey_idx = itr->second;
        int factor = fp_factor_int[i];
        m_ukey2exposure[ukey_idx][factor] = (*fp_exposure)[i];
    }
}

void BarraTransformer::build_weight_matrix() {
    fp_factor_int.resize(fp_factor->size(), 0);
    for (size_t i = 0; i < fp_factor->size(); ++i) {
        const auto& f = (*fp_factor)[i];
        auto itr = m_factor2idx.find(f);
        if (itr != m_factor2idx.end()) {
            fp_factor_int[i] = itr->second;
        } else {
            throw std::runtime_error("unknown factor " + f);
        }
    }
    m_weight_mat.resize(m_universe.size(), m_all_factor.size());
    m_weight_mat.setZero();
    for (size_t i = 0; i < fp_weight->size(); ++i) {
        int ukey = (*fp_ukeys)[i];
        auto itr = m_ukey2idx.find(ukey);
        if (itr == m_ukey2idx.end()) continue;
        int ukey_idx = itr->second;
        int factor = fp_factor_int[i];
        m_weight_mat(ukey_idx, factor) = (*fp_weight)[i];
    }
}

void BarraTransformer::build_universe(const DayData& d) {
    std::unordered_map<int, bool> ukeys;
    for (int ukey : *d.x_ukeys) {
        if (not ukeys[ukey]) {
            ukeys[ukey] = true;
            m_ukey2idx[ukey] = m_universe.size();
            m_universe.push_back(ukey);
        }
    }
}

bool BarraTransformer::load_fp(int date) {
    auto itr = std::find(all_dates.begin(), all_dates.end(), date);
    if (itr == all_dates.end()) throw std::runtime_error("cannot find " + std::to_string(date) + " in calendar");
    if (itr == all_dates.begin()) throw std::runtime_error(std::to_string(date) + " is first day in calendar");
    int pre_date = *(itr - 1);
    string fp_path = ztool::path_join(m_fp_dir, std::to_string(pre_date) + ".fst");
    FstReader::read(fp_path, m_fp_id);

    for (auto& col : m_fp_id.cols) {
        if (col.type == 1) {
            auto& vec = *reinterpret_cast<std::vector<double>*>(col.data);
            if (col.name == "Exposure")
                fp_exposure = &vec;
            else if (col.name == "Weight")
                fp_weight = &vec;
        } else if (col.type == 3) {
            auto& vec = *reinterpret_cast<std::vector<int>*>(col.data);
            if (col.name == "ukey")
                fp_ukeys = &vec;
            else if (col.name == "DataDate")
                fp_dates = &vec;
        } else if (col.type == 4) {
            auto& vec = *reinterpret_cast<std::vector<std::string>*>(col.data);
            if (col.name == "Factor") fp_factor = &vec;
        }
    }

    if (fp_ukeys == nullptr || fp_dates == nullptr || fp_exposure == nullptr || fp_weight == nullptr ||
        fp_factor == nullptr) {
        throw std::runtime_error("no ukey/date/exposure/weight/factor column in " + fp_path);
    }
    if (m_fp_id.rows == 0) {
        printf("WARN! empty fp table %s\n", fp_path.c_str());
        return false;
    }
    return true;
}

BarraTransformer::BarraTransformer() {
    barra_risk_indices = {"BETA",   "MOMENTUM", "SIZE",     "EARNYILD", "RESVOL",
                          "GROWTH", "BTOP",     "LEVERAGE", "LIQUIDTY", "SIZENL"};
    barra_industries = {"AERODEF", "AIRLINE",  "AUTO",     "BANKS",    "BEV",     "BLDPROD",  "CHEM",     "CNSTENG",
                        "COMSERV", "CONMAT",   "CONSSERV", "DVFININS", "ELECEQP", "ENERGY",   "FOODPROD", "HDWRSEMI",
                        "HEALTH",  "HOUSEDUR", "INDCONG",  "LEISLUX",  "MACH",    "MARINE",   "MATERIAL", "MEDIA",
                        "MTLMIN",  "PERSPRD",  "RDRLTRAN", "REALEST",  "RETAIL",  "SOFTWARE", "TRDDIST",  "UTILITIE"};
    m_all_factor = barra_risk_indices;
    m_all_factor.insert(m_all_factor.end(), barra_industries.begin(), barra_industries.end());
    m_all_factor.push_back("COUNTRY");

    for (size_t i = 0; i < m_all_factor.size(); ++i) {
        m_factor2idx[m_all_factor[i]] = i;
    }
}

void BarraTransformer::save(const DayData& d, int date) {
    string file_ = ztool::path_join(m_output_dir, std::to_string(date));
    if (m_output_type == RLMFileType::Fst)
        file_ += ".fst";
    else if (m_output_type == RLMFileType::Feather)
        file_ += ".feather";

    std::vector<OutputColumnOption> options;
    options.push_back({3, d.x_ukeys->data(), "ukey"});
    options.push_back({3, d.x_ticks->data(), "ticktime"});
    options.push_back({3, d.x_dates->data(), "DataDate"});
    for (size_t i = 0; i < d.pXs.size(); ++i) {
        options.push_back({1, m_f_features[i]->data(), "f_" + d.xNames[i]});
        options.push_back({1, m_b_features[i]->data(), "b_" + d.xNames[i]});
    }
    write_fst(file_, d.x_ukeys->size(), options);
}

void BarraTransformer::set_x(std::string x_file_pattern, std::string x_pattern,
                             const std::vector<std::string>& x_names) {
    for (auto& xn : x_names) m_x_data.m_x_names[xn] = true;
    m_x_data.m_x_pattern = x_pattern;
    m_x_data.x_path_pattern = x_file_pattern;
    m_x_data.m_x_type = get_file_type(x_file_pattern);
}

void BarraTransformer::set_output(std::string output_file_pattern) {
    m_output_dir = Dirname(output_file_pattern);
    m_output_type = get_file_type(output_file_pattern);
}

void BarraTransformer::arrange() {
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
}
}  // namespace feval