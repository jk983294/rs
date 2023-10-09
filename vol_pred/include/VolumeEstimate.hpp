#pragma once

#include <cmath>
#include <vector>
#include "FutureTick.h"

using namespace vpred;

constexpr double PRICE_MULTI = 1000;

class VolumeEstimate {
public:
    void set(int64_t tick, int64_t prev_bid_price, int64_t prev_ask_price, int64_t prev_bid_volume,
             int64_t prev_ask_volume, int64_t prev_last_price, int64_t bid_price, int64_t ask_price, int64_t bid_volume,
             int64_t ask_volume, int64_t last_price, int64_t amount, int64_t volume, int64_t oi_diff, bool is_czce);

    std::vector<FutureTrade>& split_trade();
    std::vector<FutureTrade>& split_trade_ip();

private:
    int64_t lee_ready(int64_t p);
    void add_trade(int64_t p, int64_t v);
    void crude_split();
    bool recursive_split(int64_t n, int64_t remain_volume, int64_t remain_amount);
    bool ip_split(const std::vector<double>& prices_old, int choice_volume_weights = 1);

    void split_type();

    int64_t tick_;
    int64_t prev_bid_price_{0};
    int64_t prev_ask_price_{0};
    int64_t prev_bid_volume_{0};
    int64_t prev_ask_volume_{0};
    int64_t prev_last_price_{0};
    int64_t bid_price_{0};
    int64_t ask_price_{0};
    int64_t bid_volume_{0};
    int64_t ask_volume_{0};
    int64_t last_price_{0};
    int64_t amount_{0};
    int64_t volume_{0};
    int64_t oi_diff_{0};

    int64_t lowest_price_;
    int64_t highest_price_;
    int64_t OInit_{0};
    int64_t CInit_{0};
    int64_t oo_{0};
    int64_t cc_{0};
    int64_t oc_{0};
    std::vector<FutureTrade> trades_;
    bool isCZCE_ = false;
};
