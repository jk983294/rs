#include <barra_transformer.h>
#include <helper.h>

using namespace ztool;
using std::string;

static void help() {
    std::cout << "Program options:" << std::endl;
    std::cout << "  -h                                    list help" << std::endl;
}

int main(int argc, char** argv) {
    feval::BarraTransformer rd;

    std::string m_x_p{"~/result/*.fst"};
    std::string m_output_p{"~/ys/feather/*.feather"};
    std::string m_x_pattern = "cneq_";
    int start_date = -1;
    int end_date = -1;
    std::vector<std::string> x_names;

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

    rd.set_x(m_x_p, m_x_pattern, x_names);
    rd.set_output(m_output_p);
    rd.set_start_date(start_date);
    rd.set_end_date(end_date);
    rd.work();
    return 0;
}
