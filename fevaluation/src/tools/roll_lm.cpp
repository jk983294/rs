#include <accum_lm.h>
#include <unistd.h>
#include <iostream>
#include <fstream>

static void help();

std::vector<std::string> get_f_names() {
    std::vector<std::string> ret;
    std::ifstream in("/tmp/x.col");
    if (!in) {
        printf("get_f_names can't open file\n");
        return ret;
    }

    std::string line;
    while (getline(in, line)) {
        if (line.empty()) continue;
        ret.push_back(line);
    }
    return ret;
}

int main(int argc, char** argv) {
    std::string config;
    feval::AccumLM rlm;

     std::string m_x_p{"~/result/*.fst"};
     std::string  m_y_p{"~/ys/feather/*.feather"};
     std::string m_y_name = "y30m1";
     std::string m_x_pattern = "cneq_";
     int rolling_days = 1;
     int start_date = -1;
     int end_date = -1;
     std::vector<std::string> x_names;
     rlm.set_cuda_memory_size(50 * 240 * 4000);
     std::string x2_pattern = "cneq_";
     std::vector<std::string> x2_cols;
     rlm.add_x2_setting("~/result/extra/*.fst", x2_pattern, x2_cols);
     rlm.set_enable_x2(true);

    int opt;
    while ((opt = getopt(argc, argv, "hix:y:p:q:s:e:t:d:l:m:")) != -1) {
        switch (opt) {
            case 'x':
                m_x_p = std::string(optarg);
                break;
            case 'i':
                rlm.m_fit_intercept = true;
                break;
            case 'l':
                rlm.set_lambda(std::stod(optarg));
                break;
            case 'y':
                m_y_p = std::string(optarg);
                break;
            case 'p':
                m_y_name = std::string(optarg);
                break;
            case 'q':
                m_x_pattern = std::string(optarg);
                break;
            case 's':
                start_date = std::stoi(optarg);
                break;
            case 'e':
                end_date = std::stoi(optarg);
                break;
            case 't':
                rlm.threads = std::stoi(optarg);
                break;
            case 'd':
                rolling_days = std::stoi(optarg);
                break;
            case 'm':
                rlm.set_max_days(std::stoi(optarg));
                break;
            case 'h':
            default:
                help();
                return 0;
        }
    }

    // rlm.set_eval_date(20230101, 20231231);
    rlm.set_use_cuda(true);
    rlm.set_tradable_name("tradable");
    rlm.set_x(m_x_p, m_x_pattern, x_names);
    rlm.set_y(m_y_p, m_y_name);
    rlm.set_rolling_days(rolling_days);
    rlm.set_start_date(start_date);
    rlm.set_end_date(end_date);
    rlm.train();

    const auto& betas = rlm.get_betas();
    const auto& beta = betas.back();

    printf("\nintercept:%f\ncoef:\n", beta.back());
    for (size_t i = 0; i < rlm.m_x_len; i++) {
        printf("%f,", beta[i]);
        if ((i + 1) % 15 == 0) printf("\n");
    }
    printf("\n");
    // rlm.export_y_hat("~/result/y_hat.fst");
    return 0;
}

static void help() {
    std::cout << "Program options:" << std::endl;
    std::cout << "  -h                                    list help" << std::endl;
    std::cout << "  -x arg (=)                          dir of x files" << std::endl;
    std::cout << "  -y arg (=)                          dir of y files" << std::endl;
    std::cout << "  -p arg (=y30m1)                          name of y" << std::endl;
    std::cout << "  -q arg (=)                          pattern of x" << std::endl;
    std::cout << "  -s arg (=-1)                          start date" << std::endl;
    std::cout << "  -e arg (=-1)                          end date" << std::endl;
    std::cout << "  -t arg (=0)                          thread num" << std::endl;
    std::cout << "  -l arg (=0)                          ridge lambda" << std::endl;
    std::cout << "  -i                                   fit intercept" << std::endl;
    printf("roll_lm -q cneq_ -x ~/test/data/*.fst -y ~/test/y/*.feather\n");
}
