// Optional C ABI between Tesseract's A* search and the UPMEM DPU heuristic.
//
// Tesseract links against these symbols weakly.  The UPMEM implementation is
// supplied at runtime by libtesseract_dpu_live.so, so normal CPU-only builds do
// not depend on the UPMEM SDK.

#ifndef TESSERACT_DPU_LIVE_H
#define TESSERACT_DPU_LIVE_H

#include <stddef.h>
#include <stdint.h>

#if defined(TESSERACT_DPU_WEAK_IMPORT) && defined(__GNUC__) && !defined(_WIN32)
#define TESSERACT_DPU_API __attribute__((weak))
#else
#define TESSERACT_DPU_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  uint32_t q_bits;
  uint32_t detector_penalty_q;
  uint32_t num_detectors;
  uint32_t num_errors;
  const uint32_t* error_detector_offsets;
  size_t num_error_detector_offsets;
  const uint32_t* error_detector_ids;
  size_t num_error_detector_ids;
  const uint32_t* error_costs_q;
  size_t num_error_costs;
  const char* dpu_binary;
  // Called by the bridge immediately before/after the blocking DPU launch.
  // The Tesseract host uses these hooks to return a CPU permit while an
  // independent shot waits for its DPU.
  void (*before_dpu_wait)(void* context);
  void (*after_dpu_wait)(void* context);
  void* dpu_wait_context;
  // Resident search only: detector orders as position[order * num_detectors +
  // detector]. NULL / 0 orders = identity order.
  const uint16_t* order_positions;
  uint32_t num_orders;
} tesseract_dpu_live_config_t;

typedef struct {
  uint32_t num_candidates;
  const uint32_t* candidate_detector_offsets;
  size_t num_candidate_detector_offsets;
  const uint32_t* candidate_detector_ids;
  size_t num_candidate_detector_ids;
  const uint64_t* candidate_blocked_error_words;
  size_t num_candidate_blocked_error_words;
} tesseract_dpu_live_batch_t;

TESSERACT_DPU_API void* tesseract_dpu_live_create(
    const tesseract_dpu_live_config_t* config,
    char* error_message,
    size_t error_message_capacity);

TESSERACT_DPU_API int tesseract_dpu_live_evaluate(
    void* handle,
    const tesseract_dpu_live_batch_t* batch,
    uint32_t* output_costs_q,
    size_t output_count,
    char* error_message,
    size_t error_message_capacity);

TESSERACT_DPU_API void tesseract_dpu_live_destroy(void* handle);

// DPU-resident A* search: the whole decode of one syndrome runs on one DPU and
// the host receives the selected error chain. status 0 = solved, 1 = low
// confidence (no solution), 2 = fallback (DPU capacity exceeded; decode on CPU).
typedef struct {
  uint32_t status;
  uint32_t depth;
  uint32_t expansions;
  uint32_t candidates;
  uint32_t heap_peak;
  uint32_t arena_used;
  uint32_t pushed;
  uint32_t heap_size;      // live queue entries left on the DPU (fallback resume)
  uint32_t min_num_dets;
  uint32_t max_num_dets;
} tesseract_dpu_resident_result_t;

// Copies the DPU's remaining priority queue and error-chain arena so the host
// can continue a search that exceeded the DPU capacity. heap_nodes receives
// heap_size entries of {cost_q, path_q, chain, num_dets | depth << 16}; arena
// receives arena_used entries of {parent, error | min_detector << 16}.
TESSERACT_DPU_API int tesseract_dpu_resident_export(
    void* handle,
    uint32_t* heap_nodes,
    size_t heap_capacity_entries,
    uint32_t* arena,
    size_t arena_capacity_entries,
    char* error_message,
    size_t error_message_capacity);

TESSERACT_DPU_API void* tesseract_dpu_resident_create(
    const tesseract_dpu_live_config_t* config,
    uint32_t det_beam,
    uint32_t pqlimit,
    uint32_t no_revisit_dets,
    uint32_t max_expansions,
    char* error_message,
    size_t error_message_capacity);

TESSERACT_DPU_API int tesseract_dpu_resident_decode(
    void* handle,
    uint32_t order_index,
    uint32_t det_beam,
    const uint32_t* active_detectors,
    size_t num_active_detectors,
    uint16_t* output_errors,
    size_t output_capacity,
    tesseract_dpu_resident_result_t* result,
    char* error_message,
    size_t error_message_capacity);

TESSERACT_DPU_API void tesseract_dpu_resident_destroy(void* handle);

#ifdef __cplusplus
}
#endif

#undef TESSERACT_DPU_API

#endif  // TESSERACT_DPU_LIVE_H
