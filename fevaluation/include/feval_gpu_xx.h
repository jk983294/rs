#include <cstdint>
#include <vector>

struct FevalGpuXX {
    ~FevalGpuXX();
    void release();
    bool init(uint64_t _n_col, uint64_t _len);
    void calc(const std::vector<std::vector<double>*>& pXs, double * h_XTX, uint64_t offset, uint64_t _len);

    uint64_t n_col = 0;
    uint32_t BLOCKS = 0;
    double *d_datum{nullptr};
    double *d_XTX{nullptr};
};