//Source File
#include "renamer.h"

// Private Functions
//-----------------------------------------------
uint32_t renamer::getFreeCheckpoints(){
    uint64_t temp;
    uint32_t freeCheckpoints, i;
    freeCheckpoints = 0;

    for(i=0; i<noOfUnresolvedBranches; i++){
        temp = GBM >> i;
        if((temp & 1) == 0) freeCheckpoints++;
    }

    return freeCheckpoints;
}

uint32_t renamer::getFreeGBMBit(){
    uint64_t temp;
    uint32_t i;

    for(i=0; i<noOfUnresolvedBranches; i++){
        temp = GBM >> i;
        if((temp & 1) == 0) break;
    }
    assert(i < noOfUnresolvedBranches);
    return i;
}

//------------------------------------------------

// Public Functions
//-----------------------------------------------
renamer::renamer(uint64_t n_log_regs,
    uint64_t n_phys_regs,
    uint64_t n_branches,
    uint64_t n_active){
    noOfLogicalRegisters = n_log_regs;
    physicalRegFileSize = n_phys_regs;
    noOfUnresolvedBranches = n_branches;
    activeListSize = n_active;
    freeListSize = physicalRegFileSize - noOfLogicalRegisters;
    
    assert(physicalRegFileSize > noOfLogicalRegisters);
    assert(noOfUnresolvedBranches >= 1 && noOfUnresolvedBranches <= 64);
    assert(activeListSize > 0);

    //Initialize AMT
    amt = (uint64_t*) malloc(noOfLogicalRegisters * sizeof(uint64_t));
    for(uint64_t i=0; i<noOfLogicalRegisters; i++){
        amt[i] = i;
    }

    //Initialize RMT
    rmt = (uint64_t*) malloc(noOfLogicalRegisters * sizeof(uint64_t));
    for(uint64_t i=0; i<noOfLogicalRegisters; i++){
        rmt[i] = i;
    }
    
    //Initialize PRF
    physicalRegFile = (uint64_t*) malloc(physicalRegFileSize * sizeof(uint64_t));
    physicalRegRdy = (bool*) malloc(physicalRegFileSize * sizeof(bool));
    for(uint64_t i=0; i<physicalRegFileSize; i++){
        physicalRegRdy[i] = (i < noOfLogicalRegisters) ? true : false;
    }

    //Initialize Checkpoints
    branchCheckpoints = (branchCheckpointStruct*) malloc(noOfUnresolvedBranches * sizeof(branchCheckpointStruct));
    for(uint64_t i=0; i<noOfUnresolvedBranches; i++){
        branchCheckpoints[i].shadowMapTable = (uint64_t*) malloc(noOfLogicalRegisters * sizeof(uint64_t));
    }

    //Initialize Active List and Free List
    activeList = new CircularFIFO<activeListEntry>(activeListSize);

    freeList = new CircularFIFO<uint64_t>(freeListSize);
    for(uint64_t i = 1; i<= freeListSize; i++){
        freeList->push(i+noOfLogicalRegisters);
    }
    
    GBM = 0;

    //doubts - do we put values in rmt, amt, freelist

};

bool renamer::stall_reg(uint64_t bundle_dst){
    if(freeList->getNoOfEntries() < bundle_dst) return true;
    else return false;
}

bool renamer::stall_branch(uint64_t bundle_branch){
    if(getFreeCheckpoints() < bundle_branch) return true;
    else return false;
}

uint64_t renamer::get_branch_mask(){
    return GBM;
};

uint64_t renamer::rename_rsrc(uint64_t log_reg){
    return rmt[log_reg];
};

uint64_t renamer::rename_rdst(uint64_t log_reg){
    uint64_t result;
    result = freeList->pop();
    rmt[log_reg] = result;
    return result;
};

uint64_t renamer::checkpoint(){
     uint32_t freeBitPos;
     uint64_t bitMask;

     freeBitPos = getFreeGBMBit();
     
     memcpy(branchCheckpoints[freeBitPos].shadowMapTable, rmt, noOfLogicalRegisters*sizeof(uint64_t));
     branchCheckpoints[freeBitPos].checkpointedFreeListHead = freeList->head;
     branchCheckpoints[freeBitPos].checkpointedFreeListHeadPhase = freeList->headPhase;
     branchCheckpoints[freeBitPos].checkpointedGBM = GBM;
     bitMask = 1;
     bitMask = bitMask << freeBitPos;
     GBM = GBM | bitMask;

     return freeBitPos;
};

bool renamer::stall_dispatch(uint64_t bundle_inst){
    if(activeList->getFreeEntries() < bundle_inst) return true;
    else return false;
};

uint64_t renamer::dispatch_inst(bool dest_valid,
                        uint64_t log_reg,
                        uint64_t phys_reg,
                        bool load,
                        bool store,
                        bool branch,
                        bool amo,
                        bool csr,
                        uint64_t PC){
	activeListEntry temp;
    uint64_t index;

	temp.destFlag = dest_valid;
	temp.destLogicalReg = log_reg;
	temp.destPhysicalReg = phys_reg;
	temp.complete = false;
	temp.exception = false;
	temp.loadViolation = false;
	temp.branchMispredict = false;
	temp.valueMispredict = false;
	temp.loadFlag = load;
	temp.storeFlag = store;
	temp.branchFlag = branch;
	temp.amoFlag = amo;
	temp.csrFlag = csr;
	temp.pc = PC;

    index = activeList->tail;
    assert(activeList->getFreeEntries() != 0);
    activeList->push(temp);

    return index;
                        
};

bool renamer::is_ready(uint64_t phys_reg){
    return physicalRegRdy[phys_reg];
};

void renamer::clear_ready(uint64_t phys_reg){
    physicalRegRdy[phys_reg] = false;
};

uint64_t renamer::read(uint64_t phys_reg){
    return physicalRegFile[phys_reg];
};

void renamer::set_ready(uint64_t phys_reg){
    physicalRegRdy[phys_reg] = true;
}

void renamer::write(uint64_t phys_reg, uint64_t value){
    physicalRegFile[phys_reg] = value;
};

void renamer::set_complete(uint64_t AL_index){
    activeList->fifoList[AL_index].complete = true;
};

void renamer::resolve(uint64_t AL_index,
		     uint64_t branch_ID,
		     bool correct){

    //If branch correctly predicted
    if(correct == true){
        uint64_t bitMask;
        bitMask = 1;
        bitMask = bitMask << branch_ID;
        bitMask = ~bitMask;
        GBM = GBM & bitMask;

        for(int i=0; i<noOfUnresolvedBranches; i++){
            branchCheckpoints[i].checkpointedGBM = branchCheckpoints[i].checkpointedGBM & bitMask;
        }
    }
    else{
        GBM = branchCheckpoints[branch_ID].checkpointedGBM;

        memcpy(rmt, branchCheckpoints[branch_ID].shadowMapTable, noOfLogicalRegisters*sizeof(uint64_t));
        freeList->head = branchCheckpoints[branch_ID].checkpointedFreeListHead;
        freeList->headPhase = branchCheckpoints[branch_ID].checkpointedFreeListHeadPhase;
        activeList->tail = (AL_index < activeListSize-1) ? AL_index+1 : 0;
        activeList->tailPhase = (activeList->tail > activeList->head) ? activeList->headPhase : !activeList->headPhase;
    }

};

bool renamer::precommit(bool &completed,
                       bool &exception, bool &load_viol, bool &br_misp, bool &val_misp,
	               bool &load, bool &store, bool &branch, bool &amo, bool &csr,
		       uint64_t &PC){

completed = activeList->fifoList[activeList->head].complete;
exception = activeList->fifoList[activeList->head].exception;
load_viol = activeList->fifoList[activeList->head].loadViolation;
br_misp = activeList->fifoList[activeList->head].branchMispredict;
val_misp = activeList->fifoList[activeList->head].valueMispredict;
load = activeList->fifoList[activeList->head].loadFlag;
store = activeList->fifoList[activeList->head].storeFlag;
branch = activeList->fifoList[activeList->head].branchFlag;
amo = activeList->fifoList[activeList->head].amoFlag;
csr = activeList->fifoList[activeList->head].csrFlag;
PC = activeList->fifoList[activeList->head].pc;

if(activeList->getNoOfEntries() == 0) return false; //if Active List empty
else return true;

};

void renamer::commit(){
    assert(activeList->getNoOfEntries() != 0); //Active list not empty;
    assert(activeList->fifoList[activeList->head].complete == true);
    assert(activeList->fifoList[activeList->head].exception == false);
    assert(activeList->fifoList[activeList->head].loadViolation == false);

    activeListEntry commitedEntry;
    commitedEntry = activeList->pop();
    if(commitedEntry.destFlag == true){
        freeList->push(amt[commitedEntry.destLogicalReg]);
        amt[commitedEntry.destLogicalReg] = commitedEntry.destPhysicalReg;
    }

};

void renamer::squash(){

    activeList->head = activeList->tail;
    activeList->headPhase = activeList->tailPhase;

    freeList->head = freeList->tail;
    freeList->headPhase = !freeList->tailPhase;

    memcpy(rmt, amt, noOfLogicalRegisters*sizeof(uint64_t));

    GBM = 0;

};

void renamer::set_exception(uint64_t AL_index){
    activeList->fifoList[AL_index].exception = true;
};


void renamer::set_load_violation(uint64_t AL_index){
    activeList->fifoList[AL_index].loadViolation = true;
};

void renamer::set_branch_misprediction(uint64_t AL_index){
    activeList->fifoList[AL_index].branchMispredict = true;
};

void renamer::set_value_misprediction(uint64_t AL_index){
    activeList->fifoList[AL_index].valueMispredict = true;
};

bool renamer::get_exception(uint64_t AL_index){
    bool result;
    result = activeList->fifoList[AL_index].exception;
    return result;
};