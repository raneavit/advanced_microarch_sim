#include "probabilisticCounter.h"


ProbabilisticCounter::ProbabilisticCounter(uint32_t count_sat_in){
    count = 0;
    count_sat = count_sat_in;
}

void ProbabilisticCounter::increment(vector<double> increment_data) {
    mt19937 gen((uint64_t) rd()); // seed the generator
    uniform_int_distribution<> distr(0, max_value); // define the range
    lfsr  = distr(gen); //generate random
    if (lfsr <= increment_data[0]*max_value){
        count = count + increment_data[1];
        if(count > count_sat) count = count_sat;
    }
}

void ProbabilisticCounter::decrement(vector<double> increment_data) {
    mt19937 gen((uint64_t) rd()); // seed the generator
    uniform_int_distribution<> distr(0, max_value); // define the range
    lfsr  = distr(gen); //generate random
    if (lfsr <= increment_data[0]*max_value){
        if(increment_data[1] == 0){
            count = 0;
        }
        else{
        count = count - increment_data[1];
        if(count < 0) count = 0;
        }
    }
}

void ProbabilisticCounter::setCount(uint16_t count_in){
    count = count_in;
}

int ProbabilisticCounter::getCount(){
    return count;
}