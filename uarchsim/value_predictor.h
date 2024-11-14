#ifndef VALUE_PREDICTOR_H
#define VALUE_PREDICTOR_H
#include <cinttypes>
#include <iostream>
#include <fstream>
using namespace std;
#include "payload.h"



class value_predictor {
	

public:

    bool pred_int_alu;
    bool pred_fpu_alu;
    bool pred_load;

    virtual void predict(uint64_t index) {};
    virtual void inject_val_dispatch(uint64_t index) {};
    virtual void set_val_in_vpq(uint64_t vpq_index, uint64_t val) {};
    virtual void train_at_retire() {};
    virtual void checkpoint(uint64_t& chkpt_vpq_tail, bool& chkpt_vpq_tail_phase) {};
    virtual void restore(uint64_t recover_vpq_tail, bool recover_vpq_tail_phase) {};
    virtual bool stall_prediction(uint64_t bundle_size) {};
    virtual void squash_vpq() {};
    // virtual void set_stats(stats_t* _stats){this->stats = _stats;}
    virtual void dump_stats(FILE* fp) {};
    virtual bool eligible(payload_t *pay) {};
    virtual void print_info(FILE* fp) {};

};


#endif //VALUE_PREDICTOR_H

