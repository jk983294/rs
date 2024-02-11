#include <getopt.h>
#include <chrono>
#include <iostream>
#include <random>
#include <thread>
#include <omp.h>

using namespace std;
using namespace std::chrono;


int main(int argc, char** argv) {
    constexpr int n_col = 1600;
    constexpr int n_ukey = 5000;
    constexpr int n_tick = 240;
    int threads = 30;
    int rounds = 10;

    int opt;
    while ((opt = getopt(argc, argv, "t:r:h")) != -1) {
        switch (opt) {
            case 't':
                threads = std::stoi(optarg);
                break;
            case 'r':
                rounds = std::stoi(optarg);
                break;
            case 'h':
            default:
                return 0;
        }
    }
    omp_set_num_threads(threads);
    std::vector<std::vector<double>> datum(n_col, std::vector<double>(n_ukey * n_tick, NAN));
    {
        random_device rd;  // non-deterministic generator
        #pragma omp parallel for num_threads(threads)
        for (int i = 0; i < n_col; ++i) {
            mt19937 generator(rd());
            normal_distribution<double> uid(0, 1);
            for (int j = 0; j < n_ukey * n_tick; j++) {
                datum[i][j] = uid(generator);
            }
        }
    }

    std::vector<std::pair<int, int>> m_xx_seq;
    m_xx_seq.reserve(n_col * n_col / 2);
    for (int xi = 0; xi < n_col; xi++) {
        for (int xi1 = 0; xi1 <= xi; xi1++) {
            m_xx_seq.emplace_back(xi, xi1);
        }
    }

    cout << "init finish" << endl;

    std::vector<double> XTX(n_col * n_col, 0);
    steady_clock::time_point t0 = steady_clock::now();
    for (int k = 0; k < rounds; k++) {
        #pragma omp parallel for num_threads(threads)
        for (size_t i = 0; i < m_xx_seq.size(); i++) {
            int xi = m_xx_seq[i].first;
            int xi1 = m_xx_seq[i].second;
            auto& vec = datum[xi];
            auto& vec1 = datum[xi1];
            double val = 0;
            for (size_t ni = 0; ni < n_ukey * n_tick; ni++) {
                val += vec[ni] * vec1[ni];
            }
            if (xi == xi1) {
                XTX[xi * n_col + xi1] += val;
            } else {
                XTX[xi * n_col + xi1] += val;
                XTX[xi1 * n_col + xi] += val;
            }
        }
    }
    steady_clock::time_point t1 = steady_clock::now();
    auto t = nanoseconds{t1 - t0}.count();
    double avg_t = (double)t / 1000. / 1000. / 1000. / rounds;
    printf("t=%d, r=%d, %f %f\n", threads, rounds, avg_t, XTX[0]);
    return 0;
}
