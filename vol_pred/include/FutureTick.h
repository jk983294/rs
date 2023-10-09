#pragma once
#include <MarketDataStruct.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

namespace vpred {
const double epsilon = 0.000000001f;
inline long convert_price(double price) { return (long)((price + epsilon) * 10000L); }

struct FutureTick {
    FutureTick() : valid(0) {}

    FutureTick(const FuturesMarketDataFieldL1 &depth) {
        strncpy(InstrumentID, depth.InstrumentID, sizeof(InstrumentID));
        LastPrice = convert_price(depth.LastPrice);
        Volume = depth.Volume + epsilon;
        Turnover = convert_price(depth.Turnover);
        OpenInterest = depth.OpenInterest + epsilon;
        ClosePrice = convert_price(depth.LastPrice);
        strncpy(UpdateTime, depth.UpdateTime, sizeof(UpdateTime));
        UpdateMillisec = depth.UpdateMillisec;
        BidPrice[0] = convert_price(depth.BidPrice1);
        BidVolume[0] = depth.BidVolume1 + epsilon;
        AskPrice[0] = convert_price(depth.AskPrice1);
        AskVolume[0] = depth.AskVolume1 + epsilon;
    }

    char InstrumentID[16]{};
    char ExchangeID[8]{};
    long LastPrice{};
    long PreClosePrice{};
    long Volume{};
    long Turnover{};
    long OpenInterest{};
    long ClosePrice{};
    char UpdateTime[16]{};
    long BidPrice[5]{};
    long BidVolume[5]{};
    long AskPrice[5]{};
    long AskVolume[5]{};
    long AveragePrice;
    int32_t UpdateMillisec;
    int32_t valid : 1;
    int32_t RESERVED_00 : 31;

    void Dump() const {
        printf("=====\n");
        printf("%s %s, %ld %.2f\n", UpdateTime, InstrumentID, Volume, Turnover / 10000.0);
        printf("  %.4f %4ld; %.4f %4ld\n", BidPrice[0] / 10000.0, BidVolume[0], AskPrice[0] / 10000.0, AskVolume[0]);
        printf("  %.4f %4ld; %.4f %4ld\n", BidPrice[1] / 10000.0, BidVolume[1], AskPrice[1] / 10000.0, AskVolume[1]);
        printf("  %.4f %4ld; %.4f %4ld\n", BidPrice[2] / 10000.0, BidVolume[2], AskPrice[2] / 10000.0, AskVolume[2]);
        printf("  %.4f %4ld; %.4f %4ld\n", BidPrice[3] / 10000.0, BidVolume[3], AskPrice[3] / 10000.0, AskVolume[3]);
        printf("  %.4f %4ld; %.4f %4ld\n", BidPrice[4] / 10000.0, BidVolume[4], AskPrice[4] / 10000.0, AskVolume[4]);
    }
};

struct FutureTrade {
    int64_t Price{0};
    uint32_t Volume{0};
    char BS{'u'};
    char OC{'u'};
};
}  // namespace vpred
