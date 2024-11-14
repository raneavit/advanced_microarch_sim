#include "pipeline.h"

eSVP::eSVP(uint64_t vpq_size_in, uint64_t prediction_mode_in, uint64_t confidence_mode_in, uint64_t indexBits_in, 
        uint64_t tagBits_in, uint64_t confidence_threshold_in, uint64_t confidence_increment_in, uint64_t confidence_decrement_in, 
        uint64_t replace_stride_in, uint64_t replace_entry_in, bool pred_int_alu_in, 
        bool pred_fpu_alu_in, bool pred_load_in, uint64_t vpq_full_policy_in, uint16_t max_age_in, payload *PAY_in, pipeline_t *proc_in, renamer* REN_in)
{
    vpq_size = vpq_size_in;
    indexBits = indexBits_in;
    tagBits = tagBits_in;
    confidence_mode = confidence_mode_in;
    prediction_mode = prediction_mode_in;
    confidence_threshold = confidence_threshold_in;
    confidence_increment = confidence_increment_in;
    confidence_decrement = confidence_decrement_in;
    replace_stride = replace_stride_in;
    replace_entry = replace_entry_in;
    pred_int_alu = pred_int_alu_in;
    pred_fpu_alu = pred_fpu_alu_in;
    pred_load = pred_load_in;
    vpq_full_policy = vpq_full_policy_in;
    max_age = max_age_in;

    PAY = PAY_in;
    proc = proc_in;
    REN = REN_in;

    ie_ins_type = 0;
    ie_ins_drop = 0;
    e_ins  = 0;
    miss_ins  = 0;
    ins_conf_corr  = 0;
    ins_conf_incorr = 0;
    ins_unconf_corr = 0;
    ins_unconf_incorr = 0;

    loads_conf_corr = 0;
    alu_s_conf_corr = 0;
    alu_c_conf_corr = 0;
    fp_conf_corr = 0;

    loads_conf_incorr = 0;
    alu_s_conf_incorr = 0;
    alu_c_conf_incorr = 0;
    fp_conf_incorr = 0;

    svp_table_size = pow(2, indexBits);
    
    svp_table = (svp_entry*) malloc(svp_table_size * sizeof(svp_entry));
    vpq = new CircularFIFO<vpq_entry>(vpq_size);

    configure_probabilistic_counters();

    for(uint64_t i = 0;i<svp_table_size;i++)
    {
          svp_table[i].tag = 0;
           svp_table[i].conf = new ProbabilisticCounter(confidence_threshold);
           svp_table[i].retired_val = 0;
           svp_table[i].stride = 0;
           svp_table[i].instance = 0; 
           svp_table[i].age = new ProbabilisticCounter(max_age);
    }
}

void eSVP::predict(uint64_t index){
    switch(prediction_mode){
        case PERFECT_VAL_PRED : 
            perf_val_pred(index);
            break;
        case REAL_VAL_PRED :
            real_val_pred(index);
            break;
    }
}

void eSVP::perf_val_pred(uint64_t index){
    db_t* actual;
    uint64_t pred_val_db;
    if(PAY->buf[index].good_instruction){
        actual = proc->get_pipe()->peek(PAY->buf[index].db_index);
        pred_val_db = actual->a_rdst[0].value;
        PAY->buf[index].is_value_predicted = true;
        PAY->buf[index].predicted_value = pred_val_db;
    }
    
}

void eSVP::real_val_pred(uint64_t index){
    db_t* actual;
    uint64_t pred_val_db, pred_val, pc;
    uint64_t svp_index;
    vpq_entry temp;
    bool entry_allocated;

    //Find actual in case of oracle confidence prediction
    if(PAY->buf[index].good_instruction){
        actual = proc->get_pipe()->peek(PAY->buf[index].db_index);
        pred_val_db = actual->a_rdst[0].value;
    }

    pc = PAY->buf[index].pc;

    entry_allocated=false;
    //Push entry onto VPQ if a slot is free
    temp.pc = pc;
    if (vpq->getFreeEntries() > 0) {
        PAY->buf[index].vpq_index = vpq->tail;
        vpq->push(temp);
        PAY->buf[index].in_vpq = true;
        entry_allocated = true;
    }
    else {PAY->buf[index].in_vpq = false; PAY->buf[index].vp_eligible = false;}
    

    //Set is_predicted to a default of false
    PAY->buf[index].is_value_predicted = false;

    if(entry_allocated){
        //Index into table;
        svp_index = getIndex(pc);

        if(((tagBits>0) && (getTag(pc) == svp_table[svp_index].tag)) || (tagBits == 0)){ //We are using tag bits, tag matches i.e it is a hit or if we arent using tags it is always a hit
            PAY->buf[index].svp_miss = false;
            //Speculatively increment instances
            svp_table[svp_index].instance = svp_table[svp_index].instance + 1;
            
            //Make a prediction
            pred_val = svp_table[svp_index].retired_val + svp_table[svp_index].instance*svp_table[svp_index].stride;

            if(confidence_mode == ORACLE_VAL_CONF){
                if(PAY->buf[index].good_instruction){
                    if(pred_val_db == pred_val) PAY->buf[index].is_value_predicted = true;
                    PAY->buf[index].predicted_value = pred_val;
                }
            }
            else if(confidence_mode == REAL_VAL_CONF){
                if(svp_table[svp_index].conf->getCount() == confidence_threshold){
                    PAY->buf[index].is_value_predicted = true;
                }
                PAY->buf[index].predicted_value = pred_val; 
            }

            //Reset age
            svp_table[svp_index].age->decrement(decrement_data[AGE]);
        }
        else {
            PAY->buf[index].svp_miss = true;
            svp_table[svp_index].age->increment(decrement_data[AGE]);
        }
    }
    
}

void eSVP::inject_val_dispatch(uint64_t index){
    REN->write(PAY->buf[index].C_phys_reg, PAY->buf[index].predicted_value);
    REN->set_ready(PAY->buf[index].C_phys_reg);
}


uint64_t eSVP::getTag(uint64_t pc){
    uint64_t bitmask, temp;
    temp = (pc >> (2 + indexBits));
    bitmask = (1 << tagBits) - 1;
    return (temp & bitmask);
}

uint64_t eSVP::getIndex(uint64_t pc){
    uint64_t bitmask, temp;
    temp = (pc >> 2);
    bitmask = (1 << indexBits) - 1;
    return (temp & bitmask);
}

void eSVP::set_val_in_vpq(uint64_t vpq_index, uint64_t val){

    if(prediction_mode == REAL_VAL_PRED) vpq->fifoList[vpq_index].value = val;
 
}

void eSVP::train_at_retire(){
    uint64_t svp_index;
    vpq_entry temp;
    int64_t new_stride;
    uint64_t pred_val;
    uint64_t act_val;
    bool if_val_pred;

    if(!PAY->buf[PAY->head].vp_eligible)
    { 
        ie_ins_type++;
        return;
    }

    if(prediction_mode == PERFECT_VAL_PRED) {
        e_ins++;
        if(PAY->buf[PAY->head].is_value_predicted) ins_conf_corr++;
        return;
    }

    else{
        if(PAY->buf[PAY->head].vp_eligible && PAY->buf[PAY->head].in_vpq)
        e_ins++;
        else 
        ie_ins_drop++;
    }

    assert((PAY->buf[PAY->head].vpq_index == vpq->head) && (PAY->buf[PAY->head].in_vpq));

    assert(vpq->getNoOfEntries()>0);
    temp = vpq->pop();
    svp_index = getIndex(temp.pc);

   
    if(((tagBits>0) && (getTag(temp.pc) == svp_table[svp_index].tag)) || (tagBits == 0)){
        new_stride = temp.value - svp_table[svp_index].retired_val;
        if(new_stride == svp_table[svp_index].stride) 
            { 
            switch(PAY->buf[PAY->head].fu){
                case FU_LS:
                    svp_table[svp_index].conf->increment(increment_data[LOAD]);
                    break;
                case FU_LS_FP:
                    svp_table[svp_index].conf->increment(increment_data[LOAD]);
                    break;
                case FU_ALU_S:
                    svp_table[svp_index].conf->increment(increment_data[SIMPLE_INT_ALU]);
                    break;
                case FU_ALU_C:
                    svp_table[svp_index].conf->increment(increment_data[COMPLEX_INT_ALU]);
                    break;
                case FU_ALU_FP:
                    svp_table[svp_index].conf->increment(increment_data[FP_ALU]);
                    break;
                default:
                    svp_table[svp_index].conf->increment(increment_data[DEFAULT]);
            } 
            } 

        else{
            if(svp_table[svp_index].conf->getCount() <=replace_stride) 
            svp_table[svp_index].stride = new_stride;

            switch(PAY->buf[PAY->head].fu){
                case FU_LS:
                    svp_table[svp_index].conf->decrement(decrement_data[LOAD]);
                    break;
                case FU_LS_FP:
                    svp_table[svp_index].conf->decrement(decrement_data[LOAD]);
                    break;
                case FU_ALU_S:
                    svp_table[svp_index].conf->decrement(decrement_data[SIMPLE_INT_ALU]);
                    break;
                case FU_ALU_C:
                    svp_table[svp_index].conf->decrement(decrement_data[COMPLEX_INT_ALU]);
                    break;
                case FU_ALU_FP:
                    svp_table[svp_index].conf->decrement(decrement_data[FP_ALU]);
                    break;
                default:
                    svp_table[svp_index].conf->decrement(decrement_data[DEFAULT]);
            } 

        } 
        svp_table[svp_index].retired_val = temp.value;
        svp_table[svp_index].instance--;
    }

    else{
        if(svp_table[svp_index].conf->getCount() <=replace_entry){
            svp_table[svp_index].tag = getTag(temp.pc);
            svp_table[svp_index].conf->setCount(0);
            svp_table[svp_index].retired_val = temp.value;
            svp_table[svp_index].stride = temp.value;
            svp_table[svp_index].instance = find_instances(temp.pc);//walk vpq
            svp_table[svp_index].age->setCount(0);
        }

        if((svp_table[svp_index].age->getCount() == max_age) && (max_age!=0)){
            svp_table[svp_index].conf->decrement(decrement_data[REDUCE_CONF]);
        }
    }
    // under stats update
    pred_val = PAY->buf[PAY->head].predicted_value;
    act_val = temp.value;
    if_val_pred = PAY->buf[PAY->head].is_value_predicted;

     if(PAY->buf[PAY->head].svp_miss) 
     miss_ins++;
     else{
            if(pred_val == act_val){
                if(if_val_pred)
                    {
                        ins_conf_corr++;
                        switch(PAY->buf[PAY->head].fu){
                            case FU_LS:
                                loads_conf_corr++;
                                break;
                            case FU_LS_FP:
                                loads_conf_corr++;
                                break;
                            case FU_ALU_S:
                                alu_s_conf_corr++;
                                break;
                            case FU_ALU_C:
                                alu_c_conf_corr++;
                                break;
                            case FU_ALU_FP:
                                fp_conf_corr++;
                                break;
                        }

                    }
                else
                    ins_unconf_corr++;
            } 
            else{
                if(if_val_pred){
                    ins_conf_incorr++;
                        switch(PAY->buf[PAY->head].fu){
                            case FU_LS:
                                loads_conf_incorr++;
                                break;
                            case FU_LS_FP:
                                loads_conf_incorr++;
                                break;
                            case FU_ALU_S:
                                alu_s_conf_incorr++;
                                break;
                            case FU_ALU_C:
                                alu_c_conf_incorr++;
                                break;
                            case FU_ALU_FP:
                                fp_conf_incorr++;
                                break;
                        }
                }
                else
                    ins_unconf_incorr++;
            }
       }

}

uint64_t eSVP::find_instances(uint64_t pc){
    uint64_t search_index;
    uint64_t instances = 0;
    search_index = vpq->head;

    do { //This works with the assumption that VPQ isnt empty
        if(vpq->fifoList[search_index].pc == pc) instances++;
        search_index = ((search_index+1) == vpq->listSize) ? 0 : (search_index+1);
    }
    while(search_index != vpq->tail);

    return instances;
}

void eSVP::checkpoint(uint64_t& chkpt_vpq_tail, bool& chkpt_vpq_tail_phase){
    chkpt_vpq_tail = vpq->tail;
    chkpt_vpq_tail_phase = vpq->tailPhase;
}


void eSVP::restore(uint64_t recover_vpq_tail, bool recover_vpq_tail_phase){
    uint64_t svp_index;
    while((vpq->tail != recover_vpq_tail) || (vpq->tailPhase != recover_vpq_tail_phase)){
        if(vpq->tail == 0) {vpq->tail = vpq_size-1 ; vpq->tailPhase = (!vpq->tailPhase);}
        else vpq->tail = vpq->tail-1;
        svp_index = getIndex(vpq->fifoList[vpq->tail].pc);

        if(((tagBits>0) && (getTag(vpq->fifoList[vpq->tail].pc) == svp_table[svp_index].tag)) || (tagBits == 0)) svp_table[svp_index].instance--;

    }
}

bool eSVP::stall_prediction(uint64_t bundle_size){
    uint64_t free_space;
    free_space = vpq->getFreeEntries();

    if(prediction_mode == REAL_VAL_PRED){
        if(vpq_full_policy == 1) return false;
        else{
            if(free_space >= bundle_size) return false;
            else return true;
        }
    }
    else return false;
}

void eSVP::squash_vpq(){
    vpq->tail = vpq->head;
    vpq->tailPhase = vpq->headPhase;

    for(uint64_t i = 0;i<svp_table_size;i++)
    {
        svp_table[i].instance = 0; 
    }

}

void eSVP::dump_stats(FILE* fp) {
	int ins_total;
    int ie_ins = ie_ins_drop + ie_ins_type;
    ins_total = ie_ins + e_ins;

	fprintf(fp, "VPU MEASUREMENTS-----------------------------------\n");

	fprintf(fp, "vpmeas_ineligible  : %lu (%.2f%%)\n",ie_ins,100.0*(double)ie_ins/(double)ins_total);
    fprintf(fp, "vpmeas_ineligible_type  : %lu (%.2f%%)\n",ie_ins_type,100.0*(double)ie_ins_type/(double)ins_total);
    fprintf(fp, "vpmeas_ineligible_drop  : %lu (%.2f%%)\n",ie_ins_drop,100.0*(double)ie_ins_drop/(double)ins_total);
    fprintf(fp, "vpmeas_eligible  : %lu (%.2f%%)\n",e_ins,100.0*(double)e_ins/(double)ins_total);
    fprintf(fp, "vpmeas_miss  : %lu (%.2f%%)\n",miss_ins,100.0*(double)miss_ins/(double)ins_total);
    fprintf(fp, "vpmeas_conf_corr  : %lu (%.2f%%)\n",ins_conf_corr,100.0*(double)ins_conf_corr/(double)ins_total);

    fprintf(fp, "\t load_conf_corr  : %lu (%.2f%%)\n",loads_conf_corr,100.0*(double)loads_conf_corr/(double)ins_total);
    fprintf(fp, "\t alu_s_conf_corr  : %lu (%.2f%%)\n",alu_s_conf_corr,100.0*(double)alu_s_conf_corr/(double)ins_total);
    fprintf(fp, "\t alu_c_conf_corr  : %lu (%.2f%%)\n",alu_c_conf_corr,100.0*(double)alu_c_conf_corr/(double)ins_total);
    fprintf(fp, "\t fp_conf_corr  : %lu (%.2f%%)\n",fp_conf_corr,100.0*(double)fp_conf_corr/(double)ins_total);

    fprintf(fp, "vpmeas_conf_incorr  : %lu (%.2f%%)\n",ins_conf_incorr,100.0*(double)ins_conf_incorr/(double)ins_total);

    fprintf(fp, "\t load_conf_incorr  : %lu (%.2f%%)\n",loads_conf_incorr,100.0*(double)loads_conf_incorr/(double)ins_total);
    fprintf(fp, "\t alu_s_conf_incorr  : %lu (%.2f%%)\n",alu_s_conf_incorr,100.0*(double)alu_s_conf_incorr/(double)ins_total);
    fprintf(fp, "\t alu_c_conf_incorr  : %lu (%.2f%%)\n",alu_c_conf_incorr,100.0*(double)alu_c_conf_incorr/(double)ins_total);
    fprintf(fp, "\t fp_conf_corr  : %lu (%.2f%%)\n",fp_conf_corr,100.0*(double)fp_conf_corr/(double)ins_total);

    fprintf(fp, "vpmeas_unconf_corr  : %lu (%.2f%%)\n",ins_unconf_corr,100.0*(double)ins_unconf_corr/(double)ins_total);
    fprintf(fp, "vpmeas_unconf_incorr  : %lu (%.2f%%)\n",ins_unconf_incorr,100.0*(double)ins_unconf_incorr/(double)ins_total);
	
}

bool eSVP::eligible(payload_t *pay) {
   if (IS_INTALU(pay->flags))
      return(pred_int_alu);     // instr. is INTALU type.  It is eligible if predINTALU is configured "true".
   else if (IS_FPALU(pay->flags))
      return(pred_fpu_alu);      // instr. is FPALU type.  It is eligible if predFPALU is configured "true".
   else if (IS_LOAD(pay->flags) && !IS_AMO(pay->flags))
      return(pred_load);      // instr. is a normal LOAD (not rare load-with-reserv).  It is eligible if predLOAD is configured "true".
   else
      return(false);     // instr. is none of the above major types, so it is never eligible
}

void eSVP::configure_probabilistic_counters(){
    increment_data[LOAD] = {0.10, 1};
    increment_data[SIMPLE_INT_ALU] = {0.05, 1};
    increment_data[COMPLEX_INT_ALU] = {0.10, 1};
    increment_data[FP_ALU] = {0.10, 1};
    increment_data[AGE] = {0.0125, 1};
    increment_data[REDUCE_CONF] = {0,0};
    increment_data[DEFAULT] = {0.1, 1};


    decrement_data[LOAD] = {1, 0};
    decrement_data[SIMPLE_INT_ALU] = {1, 0};
    decrement_data[COMPLEX_INT_ALU] = {1, 0};
    decrement_data[FP_ALU] = {1, 0};
    decrement_data[AGE] = {1, 0};
    decrement_data[REDUCE_CONF] = {0.5, 1};
    decrement_data[DEFAULT] = {1, 1}; 

}

void eSVP::print_info(FILE* fp){
    uint64_t confBits, instanceBits, ageBits, svp_entry_bits, vpq_entry_bits, totalBits, svp_entries, prob_counter_overheads;

    fprintf(fp, "\n=== VALUE PREDICTOR ============================================================\n\n");

    if(!VP_ENABLE_FLAG) fprintf(fp, "VALUE PREDICTOR = none\n");
    else if(VP_PREDICTION_MODE == true) fprintf(fp, "VALUE PREDICTOR = perfect\n"); //Perfect prediction mode
    else{
        if(VP_TYPE == 0){
        fprintf(fp, "VALUE PREDICTOR = stride (Project 4 spec. implementation)\n");
        fprintf(fp, "VPQsize = %d\n", SVP_VPQ_SIZE);
        fprintf(fp, "oracleconf = %d", confidence_mode);
        if(confidence_mode == ORACLE_VAL_CONF) fprintf(fp, " (oracle confidence)\n");
        else fprintf(fp, " (real confidence)\n");
        fprintf(fp, "# index bits = %d\n", SVP_INDEX_BITS);
        fprintf(fp, "# tag bits = %d\n", SVP_TAG_BITS);
        fprintf(fp, "confmax = %d\n", SVP_CONFMAX);
        fprintf(fp, "confinc = %d\n", SVP_CONFINC);
        fprintf(fp, "confdec = %d", SVP_CONFDEC);
        if(SVP_CONFDEC == 0) fprintf(fp, " (reset)\n");
        else fprintf(fp, "\n");
        fprintf(fp, "replace_stride = %d\n", SVP_REPLACE_STRIDE);
        fprintf(fp, "replace = %d\n", SVP_REPLACE);
        fprintf(fp, "predINTALU = %d\n", SVP_PRED_INT_ALU);
        fprintf(fp, "predFPALU = %d\n", SVP_PRED_FP_ALU);
        fprintf(fp, "predLOAD = %d\n", SVP_PRED_LOAD);
        fprintf(fp, "VPQ_full_policy = %d", SVP_VPQ_FULL_POLICY);
        if(SVP_VPQ_FULL_POLICY == 0) fprintf(fp, " (stall bundle)\n");
        else fprintf(fp, " (don't allocate VPQ entries)\n");
        }
    }
    fprintf(fp, "\n");
    svp_entries = pow(2, SVP_INDEX_BITS);

    if(prediction_mode == PERFECT_VAL_PRED){
    fprintf(fp, "COST ACCOUNTING\n");
    fprintf(fp, "   Impossible.\n");
    return;
    }

    fprintf(fp, "COST ACCOUNTING\n");
    fprintf(fp, "\tOne SVP entry:\n");
    fprintf(fp, "\t\t tag            :    %lu bits   // num_tag_bits\n", SVP_TAG_BITS);
    confBits = (uint64_t)ceil(log2((double)(SVP_CONFMAX+1)));
    fprintf(fp, "\t\t conf           :    %lu bits // formula: (uint64_t)ceil(log2((double)(confmax+1)))\n", confBits);
    fprintf(fp, "\t\t retired_value  :    64 bits // RISCV64 integer size.\n");
    fprintf(fp, "\t\t stride         :    64 bits // RISCV64 integer size. Competition opportunity: truncate stride to far fewer bits based on stride distribution of stride-predictable instructions.\n");
    instanceBits = (uint64_t)ceil(log2((double)SVP_VPQ_SIZE));
    fprintf(fp, "\t\t instance ctr   :    %lu bits // formula: (uint64_t)ceil(log2((double)VPQsize))\n", instanceBits);
    ageBits = (uint64_t)ceil(log2((double)(max_age+1)));
    fprintf(fp, "\t\t age ctr   :    %lu bits // formula: ((uint64_t)ceil(log2((double)(max_age+1)))\n", ageBits);
    fprintf(fp, "\t\t -------------------------\n");
    svp_entry_bits = SVP_TAG_BITS + confBits + 64 + 64 + instanceBits + ageBits;
    fprintf(fp, "\t\t bits/SVP entry :    %lu bits\n", svp_entry_bits);

    fprintf(fp, "\t One VPQ entry:\n");
    fprintf(fp, "\t\t PC_tag         :    %lu bits // num_tag_bits\n", SVP_TAG_BITS);
    fprintf(fp, "\t\t PC_index       :    %lu bits // num_index_bits\n", SVP_INDEX_BITS);
    fprintf(fp, "\t\t value          :    64 bits // RISCV64 integer size.\n");
    fprintf(fp, "\t\t -------------------------\n");
    vpq_entry_bits = SVP_TAG_BITS + SVP_INDEX_BITS + 64;
    fprintf(fp, "\t\t bits/VPQ entry :    %d bits\n", vpq_entry_bits);

    fprintf(fp, "\t Probabilistic Counter Overheads:\n");
    fprintf(fp, "\t\t lfsr         :    %lu bits 16-bit lfsr\n", 16);
    fprintf(fp, "\t\t maps       :    %lu bits // 14 maps x (2 values per entry) x (16 bit value) \n", 448);
    fprintf(fp, "\t\t -------------------------\n");
    prob_counter_overheads = 464;

    totalBits = svp_entries*svp_entry_bits + SVP_VPQ_SIZE*vpq_entry_bits + prob_counter_overheads;
    fprintf(fp, "\t Total storage cost (bits) = %lu (%lu SVP entries x %lu bits/SVP entry) + %lu (%lu VPQ entries x %lu bits/VPQ entry) + %lu (Probabilistic Counter Overhead) = %lu bits\n",
                                    svp_entries*svp_entry_bits, svp_entries, svp_entry_bits,
                                    SVP_VPQ_SIZE*vpq_entry_bits, SVP_VPQ_SIZE, vpq_entry_bits, prob_counter_overheads, totalBits);

    fprintf(fp, "\t Total storage cost (bytes) = %.2f B (%.2f KB) \n",totalBits/8.0, totalBits/(8.0*1024.0));


}