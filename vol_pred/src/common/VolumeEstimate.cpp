#include "VolumeEstimate.hpp"
#include <printemps/printemps.h>
#include <cstdlib>
#include <iostream>

using namespace std;

void VolumeEstimate::split_type() {
    int64_t D = abs(oi_diff_);
    int64_t oo_cc = D + std::max(0l, std::lround(0.5 * (2.0 * volume_ / 3.0 - D)) * 2);
    if (oo_cc > volume_) oo_cc = volume_;
    oo_ = (oo_cc + oi_diff_) / 2;
    cc_ = (oo_cc - oi_diff_) / 2;
    oc_ = volume_ - oo_cc;
    OInit_ = oo_ + std::lround(oc_ * 0.5);
    CInit_ = volume_ - OInit_;
}

int64_t VolumeEstimate::lee_ready(int64_t p) {
    double smid = 0.5 * double(prev_ask_price_ + prev_bid_price_);
    if (p > smid + 0.1)
        return 1;
    else if (p < smid - 0.1)
        return -1;
    else {
        smid = 0.5 * double(ask_price_ + bid_price_);
        if (p > smid + 0.1)
            return 1;
        else if (p < smid - 0.1)
            return -1;
        else {
            if (p > prev_last_price_)
                return 1;
            else if (p < prev_last_price_)
                return -1;
        }
    }

    return 0;
}

void VolumeEstimate::add_trade(int64_t p, int64_t v) {
    if (p > 0 && v > 0) {
        int64_t sign = lee_ready(p);
        int64_t ov = std::lround(double(OInit_) / double(OInit_ + CInit_) * v);
        ov = std::max(static_cast<int64_t>(0), std::min(v, ov));
        int64_t cv = v - ov;
        OInit_ -= ov;
        CInit_ -= cv;
        if (ov > 0) {
            FutureTrade trd;
            trd.Price = p;
            trd.Volume = ov;
            trd.OC = 'O';
            trd.BS = sign > 0 ? 'B' : 'S';
            trades_.push_back(trd);
        }

        if (cv > 0) {
            FutureTrade trd;
            trd.Price = p;
            trd.Volume = cv;
            trd.OC = 'C';
            trd.BS = sign > 0 ? 'B' : 'S';
            trades_.push_back(trd);
        }
    }
}

std::vector<FutureTrade>& VolumeEstimate::split_trade() {
    trades_.clear();
    split_type();

    if (isCZCE_ || volume_ == 1 || last_price_ * volume_ == amount_) {
        add_trade(last_price_, volume_);
        return trades_;
    } else {
        double ap = static_cast<double>(amount_) / volume_;
        lowest_price_ = std::min(last_price_, std::min(prev_bid_price_, bid_price_));
        highest_price_ = std::max(last_price_, std::max(prev_ask_price_, ask_price_));
        if (lowest_price_ > ap) lowest_price_ = static_cast<int64_t>(ap / tick_) * tick_;
        if (highest_price_ < ap) highest_price_ = (static_cast<int64_t>(ap / tick_ - 1e-10) + 1) * tick_;
        int64_t vol = std::max(5 * tick_, highest_price_ - lowest_price_);
        lowest_price_ = std::max(tick_, lowest_price_ - vol);
        highest_price_ += vol;
        for (int64_t n = 2; n <= 3; ++n) {
            if (recursive_split(n, volume_, amount_)) {
                return trades_;
            }
        }
        cout << "|crude split|";
        printf("|crude split|\n");
        crude_split();
    }

    return trades_;
}

std::vector<FutureTrade>& VolumeEstimate::split_trade_ip() {
    std::vector<FutureTrade>& trades_old = split_trade();
    std::vector<double> prices_old;
    for (auto& tt : trades_old) {
        prices_old.push_back(tt.Price);
    }

    trades_.clear();
    split_type();

    if (isCZCE_ || volume_ == 1) {
        add_trade(last_price_, volume_);
        return trades_;
    } else {
        double ap = static_cast<double>(amount_) / volume_;
        lowest_price_ = std::min(last_price_, std::min(prev_bid_price_, bid_price_));
        highest_price_ = std::max(last_price_, std::max(prev_ask_price_, ask_price_));
        if (lowest_price_ > ap) lowest_price_ = static_cast<int64_t>(ap / tick_) * tick_;
        if (highest_price_ < ap) highest_price_ = (static_cast<int64_t>(ap / tick_ - 1e-10) + 1) * tick_;
        int64_t vol = std::max(5 * tick_, highest_price_ - lowest_price_);
        lowest_price_ = std::max(tick_, lowest_price_ - vol);
        highest_price_ += vol;
        if (ip_split(prices_old)) {
            long split_dollar_volume_sum = 0.0;
            long split_volume_sum = 0.0;
            for (auto& tt : trades_) {
                split_dollar_volume_sum += tt.Price * tt.Volume;
                split_volume_sum += tt.Volume;
            }
            if ((split_dollar_volume_sum == amount_) && (split_volume_sum == volume_)) {
                return trades_;
            } else {
                std::cout << "Inequality!!!" << std::endl
                          << "split_dollar_volume_sum=" << split_dollar_volume_sum << " amount_=" << amount_
                          << std::endl
                          << "split_volume_sum=" << split_volume_sum << " volume_=" << volume_ << std::endl;
            }
        }
        return split_trade();
    }
}

void VolumeEstimate::crude_split() {
    double ap = static_cast<double>(amount_) / volume_;
    int64_t p;
    if (ap > last_price_) {
        p = prev_ask_price_;
        while (p < ap + 0.1) p += tick_;
    } else {
        p = prev_bid_price_;
        while (p > ap - 0.1 && p > 0) p -= tick_;
    }
    int64_t v = std::lround(double(amount_ - last_price_ * volume_) / double(p - last_price_));
    add_trade(p, v);
    add_trade(last_price_, volume_ - v);
}

bool VolumeEstimate::recursive_split(int64_t n, int64_t remain_volume, int64_t remain_amount) {
    if (n > remain_volume) return false;
    // amount /= M * 10000;

    if (n == 2) {
        double ap = static_cast<double>(remain_amount) / remain_volume;
        int64_t pm = last_price_ + remain_amount - last_price_ * remain_volume;
        double pb = double(remain_amount - last_price_) / double(remain_volume - 1);
        int64_t pmin, pmax;
        if (ap > last_price_) {
            pmin = std::max(last_price_ + tick_, static_cast<int64_t>(pb / tick_) * tick_);
            pmax = std::min(pm, highest_price_);
        } else {
            pmax = std::min(last_price_ - tick_, static_cast<int64_t>((pb / tick_ - 1e-10) + 1) * tick_);
            pmin = std::max(pm, lowest_price_);
        }
        if (pmin > pmax) return false;

        for (int64_t p = pmin; p <= pmax; p += tick_) {
            // come from the basic assumption: remain_amount = p * v + last_price_ * (remain_volume - v)
            int64_t v = (remain_amount - last_price_ * remain_volume) / (p - last_price_);
            if (v < remain_volume && v * p + last_price_ * (remain_volume - v) == remain_amount) {
                add_trade(p, v);
                add_trade(last_price_, remain_volume - v);
                return true;
            }
        }

        return false;
    } else {
        int64_t NP = (highest_price_ - lowest_price_) / tick_ + 1;
        int64_t NV = remain_volume - n + 1;
        int64_t np = 0;
        int64_t signp = 1;
        int64_t i = 0;
        int64_t p = last_price_;
        while (np < NP) {
            p += (signp *= -1) * (++i) * tick_;
            if (p <= highest_price_ && p >= lowest_price_) {
                ++np;
                int64_t nv = 0;
                int64_t signv = 1;
                int64_t j = 0;
                int64_t v = std::max(remain_volume / n, static_cast<int64_t>(1));
                while (nv < NV) {
                    v += (signv *= -1) * (j++);
                    if (v <= NV && v >= 1) {
                        ++nv;
                        if (recursive_split(n - 1, remain_volume - v, remain_amount - v * p)) {
                            add_trade(p, v);
                            return true;
                        }
                    }
                }
            } else
                break;
        }
        return false;
    }
}

bool ArePriceEqual(double a, double b, double tick) { return fabs(a - b) < 0.1 * tick; }

bool VolumeEstimate::ip_split(const std::vector<double>& prices_old, int choice_volume_weights) {
    printemps::model::IPModel model;
    printemps::option::Option option;
    option.general.time_max = 0.0005;
    option.tabu_search.time_max = 0.0005;
    option.local_search.time_max = 0.0005;
    option.lagrange_dual.time_max = 0.0005;

    int64_t real_tick = tick_ * 10;  // TODO: check: tick_ seems too small to be true???
    int64_t n_slot = static_cast<int64_t>((highest_price_ - lowest_price_) / real_tick) + 1;
    while (n_slot > 5) {
        real_tick *= 2;
        n_slot = static_cast<int64_t>((highest_price_ - lowest_price_) / real_tick) + 1;
    }

    std::vector<double> vec_prices;
    double tmp_price = lowest_price_;
    for (auto i_slot = 0; i_slot < n_slot; i_slot++) {
        vec_prices.push_back(tmp_price);
        tmp_price += real_tick;
    }
    for (auto pp : prices_old) {
        vec_prices.push_back(pp);
    }
    vec_prices.push_back(last_price_);
    vec_prices.push_back(ask_price_);
    vec_prices.push_back(bid_price_);
    vec_prices.push_back(prev_ask_price_);
    vec_prices.push_back(prev_bid_price_);
    vec_prices.push_back(lowest_price_);
    vec_prices.push_back(highest_price_);
    std::sort(vec_prices.begin(), vec_prices.end());
    vec_prices.erase(std::unique(vec_prices.begin(), vec_prices.end()), vec_prices.end());

    n_slot = vec_prices.size();
    auto& x = model.create_variables("x", n_slot, 0, volume_);
    auto& g = model.create_constraints("g", 2);

    std::vector<int64_t> vec_volume_weights;
    double hyper_param_volume_weights;
    if (choice_volume_weights == 2) {
        hyper_param_volume_weights = 2.0;
    }
    for (auto i_slot = 0; i_slot < n_slot; i_slot++) {
        x(i_slot) = 0;
        tmp_price = vec_prices[i_slot];
        if (ArePriceEqual(tmp_price, last_price_, tick_)) {
            // at least 1
            x(i_slot).set_bound(1, volume_);
            // g(2) = x(i_slot) >= 1;
            x(i_slot) = 1;
        }

        if (choice_volume_weights == 1) {
            if (ArePriceEqual(tmp_price, bid_price_, tick_)) {
                vec_volume_weights.push_back(1.0 + bid_volume_);
            } else if (ArePriceEqual(tmp_price, ask_price_, tick_)) {
                vec_volume_weights.push_back(1.0 + ask_volume_);
            } else if (ArePriceEqual(tmp_price, prev_bid_price_, tick_)) {
                vec_volume_weights.push_back(1.0 + prev_bid_volume_);
            } else if (ArePriceEqual(tmp_price, prev_ask_price_, tick_)) {
                vec_volume_weights.push_back(1.0 + prev_ask_volume_);
            } else {
                vec_volume_weights.push_back(0.0);
            }
        } else if (choice_volume_weights == 2) {
            double this_volume_weight = hyper_param_volume_weights;
            if (tmp_price < bid_price_) {
                this_volume_weight -= bid_volume_ / (bid_volume_ + ask_volume_);
            }
            if (tmp_price > ask_price_) {
                this_volume_weight -= ask_volume_ / (bid_volume_ + ask_volume_);
            }
            if (tmp_price < prev_bid_price_) {
                this_volume_weight -= prev_bid_volume_ / (prev_bid_volume_ + prev_ask_volume_);
            }
            if (tmp_price > prev_ask_price_) {
                this_volume_weight -= prev_ask_volume_ / (prev_bid_volume_ + prev_ask_volume_);
            }
            vec_volume_weights.push_back(this_volume_weight);
        }
    }

    auto& total_dollar_volume = model.create_expression("total_dollar_volume", x.dot(vec_prices));
    auto& total_volume = model.create_expression("total_volume", x.sum());
    auto& volume_sim = model.create_expression("volume_sim", x.dot(vec_volume_weights));

    g(0) = total_dollar_volume == amount_;
    g(1) = total_volume == volume_;
    model.maximize(volume_sim);

    auto result = printemps::solver::solve(&model, option);

    if (result.solution.is_feasible()) {
        for (auto i_slot = 0; i_slot < n_slot; i_slot++) {
            if (result.solution.variables("x").values(i_slot) > 0.5) {
                add_trade(vec_prices[i_slot], static_cast<int64_t>(result.solution.variables("x").values(i_slot)));
            }
        }
        return true;
    } else {
        std::cout << "!!!infeasible!!!" << std::endl;
        return false;
    }
}

void VolumeEstimate::set(int64_t tick, int64_t prev_bid_price, int64_t prev_ask_price, int64_t prev_bid_volume,
                         int64_t prev_ask_volume, int64_t prev_last_price, int64_t bid_price, int64_t ask_price,
                         int64_t bid_volume, int64_t ask_volume, int64_t last_price, int64_t amount, int64_t volume,
                         int64_t oi_diff, bool is_czce) {
    tick_ = tick;
    prev_bid_price_ = prev_bid_price;
    prev_ask_price_ = prev_ask_price;
    prev_bid_volume_ = prev_bid_volume;
    prev_ask_volume_ = prev_ask_volume;
    prev_last_price_ = prev_last_price;
    bid_price_ = bid_price;
    ask_price_ = ask_price;
    bid_volume_ = bid_volume;
    ask_volume_ = ask_volume;
    last_price_ = last_price;
    amount_ = amount;
    volume_ = volume;
    oi_diff_ = oi_diff;
    isCZCE_ = is_czce;
}
