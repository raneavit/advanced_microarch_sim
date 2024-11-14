#include <cinttypes>
#include <cassert>
#include "decode.h"
#include "fetchunit_types.h"
#include "BPinterface.h"
#include "ras.h"

ras_t::ras_t(uint64_t size, ras_recover_e recovery_approach, uint64_t bq_size) {
   this->size = ((size > 0) ? size : 1);
   ras = new uint64_t[this->size];
   tos = 0;

   log = new ras_log_t[bq_size];
   this->recovery_approach = recovery_approach;
}

ras_t::~ras_t() {
}

// a call pushes its return address onto the RAS
void ras_t::push(uint64_t x) {
   ras[tos] = x;
   tos++;
   if (tos == size)
      tos = 0;
}

// a return pops its predicted return address from the RAS
uint64_t ras_t::pop() {
   tos = ((tos > 0) ? (tos - 1) : (size - 1));
   return(ras[tos]);
}

// Get 1 return target prediction.
//
// "pc" is the start PC of the fetch bundle; it is unused.
uint64_t ras_t::predict(uint64_t pc) {
   uint64_t temp = ((tos > 0) ? (tos - 1) : (size - 1));
   return(ras[temp]);
}

// Save the TOS pointer and TOS content as they exist prior to the fetch bundle.
void ras_t::save_fetch2_context() {
   fetch2_tos_pointer = tos;
   fetch2_tos_content = ras[tos];
}

// Speculatively update the RAS.
void ras_t::spec_update(uint64_t predictions, uint64_t num,                  /* unused: for speculatively updating branch history */
                        uint64_t pc, uint64_t next_pc,                       /* unused: for speculatively updating path history */
		        bool pop_ras, bool push_ras, uint64_t push_ras_pc) { /* used: for speculative RAS actions */
   if (pop_ras) {
      assert(!push_ras);
      pop();
   }
   if (push_ras) {
      assert(!pop_ras);
      push(push_ras_pc);
   }
}

// Restore the RAS after a misfetch.
void ras_t::restore_fetch2_context() {
   switch (recovery_approach) {
      case ras_recover_e::RAS_RECOVER_TOS_POINTER:
         tos = fetch2_tos_pointer;
         break;

      case ras_recover_e::RAS_RECOVER_TOS_POINTER_AND_CONTENT:
      case ras_recover_e::RAS_RECOVER_WALK:
         tos = fetch2_tos_pointer;
	 ras[tos] = fetch2_tos_content;
         break;

      default:
	 assert(0);
         break;
   }
}

void ras_t::log_begin() {
}

// Log a branch: record the TOS pointer (fetch2_tos_pointer) and TOS content (fetch2_tos_content) w.r.t. the branch.
// These are the TOS pointer and TOS content prior to the fetch bundle containing the branch:
// ** this assumes fetch bundles are terminated after calls and returns, i.e., the RAS is unchanged within a fetch bundle. **
// Also record whether the branch is a call or return instruction.
void ras_t::log_branch(uint64_t log_id,
		       btb_branch_type_e branch_type,
                       bool taken, uint64_t pc, uint64_t next_pc) { /* unused */
   log[log_id].tos_pointer = fetch2_tos_pointer;
   log[log_id].tos_content = fetch2_tos_content;
   log[log_id].iscall = ((branch_type == BTB_CALL_DIRECT) || (branch_type == BTB_CALL_INDIRECT));
   log[log_id].isreturn = (branch_type == BTB_RETURN);
}

// Restore the RAS after a mispredicted branch.
void ras_t::mispredict(uint64_t log_id,
                       bool iscond, bool taken, uint64_t next_pc) { /* unused */
   flush(log_id);
   // FIX_ME !!!
   // If we are rolling back to a mispredicted call or return, then after the rollback, incrementally update the RAS for this call or return.
}

// Restore the RAS after a full pipeline flush.
void ras_t::flush(uint64_t log_id) {
   switch (recovery_approach) {
      case ras_recover_e::RAS_RECOVER_TOS_POINTER:
         tos = log[log_id].tos_pointer;
         break;

      case ras_recover_e::RAS_RECOVER_TOS_POINTER_AND_CONTENT:
         tos = log[log_id].tos_pointer;
	 ras[tos] = log[log_id].tos_content;
         break;

      case ras_recover_e::RAS_RECOVER_WALK:
	 assert(0);
         break;

      default:
	 assert(0);
         break;
   }
}

void ras_t::commit(uint64_t log_id,
                   uint64_t pc,
		   uint64_t branch_in_bundle,
		   bool taken,
		   uint64_t next_pc) {
}
