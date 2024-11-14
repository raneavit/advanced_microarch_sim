#ifndef ESVP_BRANCH_CONTEXT_H
#define ESVP_BRANCH_CONTEXT_H
#include <cinttypes>
#include <vector>
#include <map>
using namespace std;

#include "circularFIFO.h"
#include "probabilisticCounter.h"

// Forward declaring pipeline_t class.
class pipeline_t;
class stats_t;


class eSVP_branch_context  : public value_predictor {
    private:

    payload *PAY;
	pipeline_t *proc;	// This is needed by PAY->map_to_actual() and PAY->predict().
    renamer *REN;
    uint64_t bhr;
    BPinterface_t *CBP;

    uint64_t prediction_mode;
    uint64_t vpq_size;
    uint64_t confidence_mode;
    uint64_t indexBits;
    uint64_t tagBits;
    uint64_t svp_table_size;
    uint64_t confidence_threshold;
    uint64_t confidence_increment;
    uint64_t confidence_decrement;
    uint64_t replace_stride;
    uint64_t replace_entry;
    uint64_t vpq_full_policy;
    uint16_t max_age;
    uint32_t bhr_bits;

    map<uint16_t, vector<double> > increment_data;
    map<uint16_t, vector<double> > decrement_data;
    

    //Stride Value Predictor Entry Structure
    typedef struct svp_entry 
    {
        // bool valid;
        uint64_t tag;
        ProbabilisticCounter *conf;
        uint64_t retired_val;
        int32_t stride;
        uint64_t instance;
        ProbabilisticCounter *age;
    } svp_entry;

    svp_entry *svp_table;

    typedef struct vpq_entry 
    {
        // bool valid;
        uint64_t pc_tag;
        uint64_t pc_index;
        uint64_t value;
    } vpq_entry;

    //Prediction Modes
    enum prediction_modes 
    {
        PERFECT_VAL_PRED = 1,
        REAL_VAL_PRED = 0
    };

    //Confidence Modes
    enum confidence_modes 
    {
        ORACLE_VAL_CONF = 1,
        REAL_VAL_CONF = 0
    };


    // stats
    uint64_t ie_ins_type;
    uint64_t ie_ins_drop;
    uint64_t e_ins;
    uint64_t miss_ins;
    uint64_t ins_conf_corr;
    uint64_t loads_conf_corr;
    uint64_t alu_s_conf_corr;
    uint64_t alu_c_conf_corr;
    uint64_t fp_conf_corr;
    uint64_t ins_conf_incorr;
    uint64_t loads_conf_incorr;
    uint64_t alu_s_conf_incorr;
    uint64_t alu_c_conf_incorr;
    uint64_t fp_conf_incorr;
    uint64_t ins_unconf_corr;
    uint64_t ins_unconf_incorr;

    stats_t* stats;

    enum counter_types 
    {
        LOAD,
        SIMPLE_INT_ALU,
        COMPLEX_INT_ALU,
        FP_ALU,
        AGE,
        REDUCE_CONF,
        DEFAULT
    };
	

public:

    CircularFIFO<vpq_entry> *vpq;
	
	eSVP_branch_context(uint64_t vpq_size_in, uint64_t prediction_mode_in, uint64_t confidence_mode_in, uint64_t indexBits_in, 
        uint64_t tagBits_in, uint64_t confidence_threshold_in, uint64_t confidence_increment_in, uint64_t confidence_decrement_in, 
        uint64_t replace_stride_in, uint64_t replace_entry_in, bool pred_int_alu_in, 
        bool pred_fpu_alu_in, bool pred_load_in, uint64_t vpq_full_policy_in, uint16_t max_age, uint32_t bhr_bits_in, payload *PAY_in, pipeline_t *proc_in, renamer* REN_in, BPinterface_t *CBP_in);

	~eSVP_branch_context();

    void predict(uint64_t index);
    void perf_val_pred(uint64_t index);
    void real_val_pred(uint64_t index);
    uint64_t getTag(uint64_t pc);
    uint64_t getIndex(uint64_t pc);
    void inject_val_dispatch(uint64_t index);
    void set_val_in_vpq(uint64_t vpq_index, uint64_t val);
    void train_at_retire();
    uint64_t find_instances(uint64_t tag, uint64_t index);
    void repair_vpq(uint64_t vpq_tail, uint64_t vpq_tail_phase);
    void checkpoint(uint64_t& chkpt_vpq_tail, bool& chkpt_vpq_tail_phase);
    void restore(uint64_t recover_vpq_tail, bool recover_vpq_tail_phase);
    bool stall_prediction(uint64_t bundle_size);
    void squash_vpq();
    //void set_stats(stats_t* _stats){this->stats = _stats;}
    void dump_stats(FILE* fp);
    bool eligible(payload_t *pay);
    void configure_probabilistic_counters();
    void print_info(FILE* fp);
    void update_bhr(uint64_t index);

};


#endif //STRIDE_VAL_PRED_H
