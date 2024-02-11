#include <zerg_fst.h>
#include <helper.h>
#include <Eigen/Dense>

using namespace ztool;
using std::string;

struct RetDecompose {
    RetDecompose();
    void work();

    InputData m_fp_id;
    InputData m_feature_id;
    string m_fp_path;
    string m_feature_path;
    std::vector<int>* fp_ukeys{nullptr};
    std::vector<int>* fp_dates{nullptr};
    std::vector<double>* fp_exposure{nullptr};
    std::vector<double>* fp_weight{nullptr};
    std::vector<std::string>* fp_factor{nullptr};
    std::vector<int> fp_factor_int;
    std::vector<int>* feature_ukeys{nullptr};
    std::vector<int>* feature_dates{nullptr};
    std::vector<int>* feature_ticks{nullptr};
    std::vector<double>* feature_ret{nullptr};
    std::vector<std::string> barra_risk_indices;
    std::vector<std::string> barra_industries;
    std::vector<std::string> m_all_factor;
    std::unordered_map<std::string, int> m_factor2idx;
    std::unordered_map<int, int> m_ukey2idx;
    std::vector<int> m_universe;
    Eigen::MatrixXd m_weight_mat;
    std::vector<std::vector<double>> m_ukey2exposure;
    std::vector<double> m_f_ret, m_b_ret;

private:
    bool load_fp();
    bool load_ret();
    void build_weight_matrix();
    void build_exposure();
    void build_universe();
    void save();
};

static void help() {
    std::cout << "Program options:" << std::endl;
    std::cout << "  -h                                    list help" << std::endl;
}

int main(int argc, char** argv) {
    RetDecompose rd;
    string config;
    int opt;
    while ((opt = getopt(argc, argv, "h")) != -1) {
        switch (opt) {
            case 'h':
            default:
                help();
                return 0;
        }
    }

    rd.work();
    return 0;
}

void RetDecompose::work() {
    if (not load_fp()) return;
    if (not load_ret()) return;

    build_universe();
    build_weight_matrix();
    build_exposure();

    m_f_ret.resize(feature_ticks->size(), NAN);
    m_b_ret.resize(feature_ticks->size(), NAN);

    std::unordered_map<int, std::vector<size_t>> tick2pos;
    for (size_t i = 0; i < feature_ticks->size(); ++i) {
        tick2pos[(*feature_ticks)[i]].push_back(i);
    }

    for (auto& item : tick2pos) {
        const std::vector<size_t>& idxes = item.second;
        Eigen::VectorXd feature(m_universe.size());
        feature.setZero();
        for (auto idx : idxes) {
            int ukey = (*feature_ukeys)[idx];
            int univ_idx = m_ukey2idx[ukey];
            feature[univ_idx] = (*feature_ret)[idx];
        }

        Eigen::VectorXd F = feature * m_weight_mat;

        for (auto idx : idxes) {
            int ukey = (*feature_ukeys)[idx];
            int univ_idx = m_ukey2idx[ukey];
            const auto& exposure_ = m_ukey2exposure[univ_idx];
            Eigen::Map<const Eigen::VectorXd> exposure_vec(exposure_.data(), m_all_factor.size());
            double f_ret_val = (exposure_vec.array() * F.array()).sum();
            m_f_ret[idx] = f_ret_val;
            m_b_ret[idx] = (*feature_ret)[idx] - f_ret_val;
        }
    }

    save();
}

void RetDecompose::build_exposure() {
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

void RetDecompose::build_weight_matrix() {
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

void RetDecompose::build_universe() {
    std::unordered_map<int, bool> ukeys;
    for (int ukey : *feature_ukeys) {
        if (not ukeys[ukey]) {
            ukeys[ukey] = true;
            m_ukey2idx[ukey] = m_universe.size();
            m_universe.push_back(ukey);
        }
    }
}

bool RetDecompose::load_fp() {
    FstReader::read(m_fp_path, m_fp_id);

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
            if (col.name == "Factor")
                fp_factor = &vec;
        }
    }

    if (fp_ukeys == nullptr || fp_dates == nullptr || fp_exposure == nullptr
        || fp_weight == nullptr || fp_factor == nullptr) {
        throw std::runtime_error("no ukey/date/exposure/weight/factor column");
    }
    if (m_fp_id.rows == 0) {
        printf("WARN! empty fp table %s\n", m_fp_path.c_str());
        return false;
    }
    return true;
}
bool RetDecompose::load_ret() {
    FstReader::read(m_feature_path, m_feature_id);

    for (auto& col : m_feature_id.cols) {
        if (col.type == 1) {
            auto& vec = *reinterpret_cast<std::vector<double>*>(col.data);
            if (col.name == "ret") feature_ret = &vec;
        } else if (col.type == 3) {
            auto& vec = *reinterpret_cast<std::vector<int>*>(col.data);
            if (col.name == "ukey")
                feature_ukeys = &vec;
            else if (col.name == "ticktime")
                feature_ticks = &vec;
            else if (col.name == "DataDate")
                feature_dates = &vec;
        }
    }

    if (feature_ukeys == nullptr || feature_ticks == nullptr || feature_dates == nullptr
        || feature_ret == nullptr) {
        throw std::runtime_error("no ukey/date/ticktime/ret column");
    }
    if (m_fp_id.rows == 0) {
        printf("WARN! empty ret table %s\n", m_feature_path.c_str());
        return false;
    }
    return true;
}

RetDecompose::RetDecompose() {
    barra_risk_indices = {"BETA", "MOMENTUM", "SIZE","EARNYILD", "RESVOL", "GROWTH",
                          "BTOP", "LEVERAGE", "LIQUIDTY", "SIZENL"};
    barra_industries = {"AERODEF", "AIRLINE", "AUTO", "BANKS", "BEV", "BLDPROD", "CHEM",
                        "CNSTENG", "COMSERV", "CONMAT", "CONSSERV", "DVFININS", "ELECEQP",
                        "ENERGY", "FOODPROD", "HDWRSEMI", "HEALTH", "HOUSEDUR", "INDCONG",
                        "LEISLUX", "MACH", "MARINE", "MATERIAL", "MEDIA", "MTLMIN", "PERSPRD",
                        "RDRLTRAN", "REALEST", "RETAIL", "SOFTWARE", "TRDDIST", "UTILITIE"};
    m_all_factor = barra_risk_indices;
    m_all_factor.insert(m_all_factor.end(), barra_industries.begin(), barra_industries.end());
    m_all_factor.push_back("COUNTRY");

    for (size_t i = 0; i < m_all_factor.size(); ++i) {
        m_factor2idx[m_all_factor[i]] = i;
    }
}

void RetDecompose::save() {
    string file_ = "/tmp/out.fst";
    std::vector<OutputColumnOption> options;
    options.push_back({3, feature_ukeys->data(), "ukey"});
    options.push_back({3, feature_ticks->data(), "ticktime"});
    options.push_back({3, feature_dates->data(), "DataDate"});
    options.push_back({1, m_f_ret.data(), "f_ret"});
    options.push_back({1, m_b_ret.data(), "b_ret"});
    write_fst(file_, feature_ukeys->size(), options);
}