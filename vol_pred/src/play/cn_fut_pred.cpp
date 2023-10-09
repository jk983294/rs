#include <CtpUtils.h>
#include <FutureTick.h>
#include <MarketDataStruct.h>
#include <zerg_cn_fut.h>
#include <zerg_file.h>
#include <zerg_gz.h>
#include <zerg_util.h>
#include <VolumeEstimate.hpp>

using namespace std;

struct CnFutPred {
    void analysis(const string& path);

    void work();

    int cob{20230922};
    string ins{"cu2312"};
    FutureTick prev_tick;
    vector<CtpInstrumentProfile> profiles;
    unordered_map<string, const CtpInstrumentProfile*> ins2profile;
};

void help() {
    std::cout << "Program options:" << std::endl;
    std::cout << "  -h                                    list help" << std::endl;
    std::cout << "  -d                                    cob default=20230922" << std::endl;
    std::cout << "example:" << std::endl;
    std::cout << "cn_fut_pred -d 20230922" << std::endl;
}

int main(int argc, char** argv) {
    CnFutPred pred;
    int opt;
    while ((opt = getopt(argc, argv, "hd:")) != -1) {
        switch (opt) {
            case 'd':
                pred.cob = std::stoi(optarg);
                break;
            case 'h':
                help();
                return 0;

            default:
                cerr << "unknown option" << endl;
                return 0;
        }
    }

    pred.work();
    return 0;
}

void CnFutPred::analysis(const string& path) {
    string path1 = ztool::GetAbsolutePath(path);
    GzBufferReader<FuturesMarketDataFieldL1> reader;
    if (!reader.open(path1)) {
        cerr << "cannot open " << path1 << endl;
        return;
    }
    const CtpInstrumentProfile* profile = ins2profile[ins];
    if (profile == nullptr) {
        printf("no profile %s found\n", ins.c_str());
        return;
    }
    long price_tick = convert_price(profile->PriceTick);

    while (!reader.iseof()) {
        int num = reader.read();
        if (num < 0) {
            cerr << "error when reading " << path1 << endl;
            abort();
        }
        for (int i = 0; i < num; ++i) {
            const auto& data = reader.buffer[i];
            if (data.InstrumentID == ins) {
                printf("%s %s %d\n", data.InstrumentID, data.UpdateTime, data.UpdateMillisec);
                FutureTick tick(data);

                if (prev_tick.valid && tick.Volume - prev_tick.Volume > 0) {
                    VolumeEstimate est;
                    est.set(price_tick, prev_tick.BidPrice[0], prev_tick.AskPrice[0], prev_tick.BidVolume[0],
                            prev_tick.AskVolume[0], prev_tick.LastPrice, tick.BidPrice[0], tick.AskPrice[0],
                            tick.BidVolume[0], tick.AskVolume[0], tick.LastPrice,
                            (tick.Turnover - prev_tick.Turnover) / profile->VolumeMultiple,
                            tick.Volume - prev_tick.Volume, tick.OpenInterest - prev_tick.OpenInterest,
                            profile->ExchangeID == "CZCE");

                    std::vector<FutureTrade>& tvec = est.split_trade();
                    for (auto& t : tvec) {
                        printf("trade p=%ld v=%u, bs=%c, oc=%c\n", t.Price, t.Volume, t.BS, t.OC);
                    }
                }

                memcpy(&prev_tick, &tick, sizeof(tick));
                prev_tick.valid = true;
            }
        }
    }
}

void CnFutPred::work() {
    string dir_pattern = "/dat/ctp/${YYYY}/${MM}/${DD}/";
    string file_dir = ztool::replace_time_placeholder(dir_pattern, cob);
    string p2_day = ztool::path_join(file_dir, "instruments.list.day");
    string p2_night = ztool::path_join(file_dir, "instruments.list");
    if (ztool::IsFileExisted(p2_day)) {
        profiles = ctp_profile_from_csv(p2_day);
    } else if (ztool::IsFileExisted(p2_night)) {
        profiles = ctp_profile_from_csv(p2_night);
    }
    for (const auto& profile : profiles) {
        ins2profile[profile.InstrumentID] = &profile;
    }

    if (ztool::IsDir(file_dir)) {
        string cob_str = std::to_string(cob);
        string p1 = ztool::path_join(file_dir, "md.night." + cob_str + ".gz");
        if (ztool::IsFileExisted(p1)) analysis(p1);

        string p2 = ztool::path_join(file_dir, "md.day." + cob_str + ".gz");
        if (ztool::IsFileExisted(p2)) analysis(p2);
    }
}
