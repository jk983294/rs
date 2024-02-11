#include <columnfactory.h>
#include <fststore.h>
#include <fsttable.h>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <experimental/filesystem>

using namespace std;
namespace fs = std::experimental::filesystem;

static void output(string prefix, int date, vector<size_t>& index, vector<int>& ukeys, vector<double>& weight);

int main() {
    std::string path_ = "~/result/index_weight.fst";
    FstTable table;
    ColumnFactory columnFactory;
    std::vector<int> keyIndex;
    StringArray selectedCols;

    std::unique_ptr<StringColumn> col_names(new StringColumn());
    FstStore fstStore(path_);
    fstStore.fstRead(table, nullptr, 1, -1, &columnFactory, keyIndex, &selectedCols, &*col_names);

    uint64_t rows = table.NrOfRows();
    vector<int> dates(rows, 0);
    vector<int> ukeys(rows, 0);
    vector<double> w50(rows, 0);
    vector<double> w300(rows, 0);
    vector<double> w500(rows, 0);
    vector<double> w1000(rows, 0);
    printf("rows=%zu\n", rows);
    for (uint64_t i = 0; i < selectedCols.Length(); ++i) {
        std::shared_ptr<DestructableObject> column;
        FstColumnType type;
        std::string colName;
        short int colScale;
        std::string annotation;
        table.GetColumn(i, column, type, colName, colScale, annotation);
        colName = selectedCols.GetElement(i);
        printf("type=%d, name=%s, scale=%d, Annotation=%s\n", type, colName.c_str(), colScale,
               annotation.c_str());

        if (type == FstColumnType::INT_32) {
            auto i_vec = std::dynamic_pointer_cast<IntVector>(column);
            auto ptr = i_vec->Data();
            int* data = nullptr;
            if (colName == "TradeDate") data = dates.data();
            else if (colName == "ukey") data = ukeys.data();
            if (data) {
                for (uint64_t j = 0; j < rows; ++j) {
                    data[j] = ptr[j];
                }
            }
        } else if (type == FstColumnType::DOUBLE_64) {
            auto i_vec = std::dynamic_pointer_cast<DoubleVector>(column);
            auto ptr = i_vec->Data();
            double* data = nullptr;
            if (colName == "w50") data = w50.data();
            else if (colName == "w300") data = w300.data();
            else if (colName == "w500") data = w500.data();
            else if (colName == "w1000") data = w1000.data();
            if (data) {
                for (uint64_t j = 0; j < rows; ++j) {
                    data[j] = ptr[j];
                }
            }
        }
    }

    printf("%d %f %f %f  %f\n", dates.back(), w50.back(), w300.back(), w500.back(), w1000.back());
    std::unordered_map<int, vector<size_t>> date2index;
    for (uint64_t j = 0; j < rows; ++j) {
        date2index[dates[j]].push_back(j);
    }
    for (auto& item : date2index) {
        vector<size_t>& index = item.second;
        int date = item.first;
        output( "./index_weights/i50/", date, index, ukeys, w50);
        output( "./index_weights/i300/", date, index, ukeys, w300);
        output( "./index_weights/i500/", date, index, ukeys, w500);
        output( "./index_weights/i1000/", date, index, ukeys, w1000);
    }
    return 0;
}

void output(string prefix, int date, vector<size_t>& index, vector<int>& ukeys, vector<double>& weight) {
    if (not fs::exists(prefix)) fs::create_directory(prefix);
    string file = prefix + std::to_string(date) + ".cmp";
    ofstream ofs(file, ofstream::out | ofstream::trunc);
    if (ofs) {
        for (size_t idx : index) {
            ofs << ukeys[idx] << "," << weight[idx] << "\n";
        }
        ofs.close();
    }
    printf("output to %s %zu\n", file.c_str(), index.size());
}
