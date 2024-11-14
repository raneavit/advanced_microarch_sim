#ifndef PROBABILISTIC_COUNTER_H
#define PROBABILISTIC_COUNTER_H

#include <iostream>
#include <random>
#include <cinttypes>
#include <vector>
#include <random>
#include <cstdlib>
using namespace std;

class ProbabilisticCounter {
private:
    uint16_t max_value = 0xffff; //since we are using a 16 bit lfsr
    uint16_t count;
    uint16_t count_sat;

    uint16_t lfsr; //16 bit lfsr
    random_device rd;


public:
    ProbabilisticCounter(uint32_t count_sat);

    void increment(vector<double> increment_data);

    void decrement(vector<double> increment_data);
    void setCount(uint16_t count_in);

    int getCount();
};

#endif //PROBABILISTIC_COUNTER_H