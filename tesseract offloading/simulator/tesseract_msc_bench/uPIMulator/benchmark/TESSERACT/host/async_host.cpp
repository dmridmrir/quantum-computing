#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <tesseract_dpu_live.h>

extern "C" {
#include <dpu.h>
#include "../support/common.h"
}

namespace {

using Clock = std::chrono::steady_clock;
using Json = nlohmann::json;
constexpr uint32_t kNoBlockedWords = UINT32_MAX;
constexpr uint32_t kInfCost = UINT32_MAX;
constexpr size_t kDpuMramBytes = 64u * 1024u * 1024u;

double seconds_since(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

void check_dpu(dpu_error_t status, const char* operation) {
  if (status != DPU_OK) {
    throw std::runtime_error(std::string(operation) + ": " + dpu_error_to_string(status));
  }
}

struct Trace {
  uint32_t q_bits;
  uint32_t detector_penalty_q;
  uint32_t num_detectors;
  std::vector<uint32_t> e2d_offsets;
  std::vector<uint32_t> e2d_detector_ids;
  std::vector<uint32_t> error_costs_q;
  std::vector<uint32_t> candidate_offsets;
  std::vector<uint32_t> candidate_detector_ids;
  std::vector<uint64_t> candidate_blocked_words;
  std::vector<uint32_t> candidate_costs_q;
};

uint32_t append_u32(std::vector<uint8_t>& bytes,
                    const std::vector<uint32_t>& values);

void validate_offsets(const char* name,
                      const std::vector<uint32_t>& offsets,
                      size_t rows,
                      size_t ids) {
  if (offsets.size() != rows + 1) {
    throw std::runtime_error(std::string(name) +
                             " offsets have the wrong number of entries");
  }
  if (offsets.front() != 0) {
    throw std::runtime_error(std::string(name) + " offsets must start at zero");
  }
  for (size_t index = 1; index < offsets.size(); ++index) {
    if (offsets[index] < offsets[index - 1]) {
      throw std::runtime_error(std::string(name) + " offsets are not monotonic");
    }
  }
  if (static_cast<size_t>(offsets.back()) != ids) {
    throw std::runtime_error(std::string(name) +
                             " final offset does not match the ID count");
  }
}

[[maybe_unused]] Trace load_trace(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("cannot open trace: " + path);
  }
  Json value;
  input >> value;
  if (value.at("format") != "tesseract-dpu-trace-v1") {
    throw std::runtime_error("unsupported trace format");
  }

  Trace result{
      value.at("q_bits").get<uint32_t>(),
      value.at("detector_penalty_q").get<uint32_t>(),
      value.at("num_detectors").get<uint32_t>(),
      value.at("error_detector_offsets").get<std::vector<uint32_t>>(),
      value.at("error_detector_ids").get<std::vector<uint32_t>>(),
      value.at("error_cost_q").get<std::vector<uint32_t>>(),
      value.at("candidate_detector_offsets").get<std::vector<uint32_t>>(),
      value.at("candidate_detector_ids").get<std::vector<uint32_t>>(),
      value.value("candidate_blocked_error_words", std::vector<uint64_t>{}),
      value.value("candidate_cost_q", std::vector<uint32_t>{}),
  };
  if (result.q_bits > 24) {
    throw std::runtime_error("trace q_bits must be in [0,24]");
  }
  if (result.num_detectors == 0 || result.num_detectors > TESSERACT_MAX_DETECTORS) {
    throw std::runtime_error("trace detector count is outside the DPU limit");
  }
  const size_t num_errors = result.error_costs_q.size();
  if (num_errors == 0) {
    throw std::runtime_error("trace must contain at least one error");
  }
  for (size_t error = 0; error < num_errors; ++error) {
    if (result.error_costs_q[error] == kInfCost) {
      throw std::runtime_error("trace error cost uses the reserved INF value");
    }
  }
  validate_offsets("error-to-detector", result.e2d_offsets, num_errors,
                   result.e2d_detector_ids.size());
  if (result.candidate_offsets.size() < 2) {
    throw std::runtime_error("trace must describe at least one candidate");
  }
  const size_t num_candidates = result.candidate_offsets.size() - 1;
  validate_offsets("candidate-to-detector", result.candidate_offsets,
                   num_candidates, result.candidate_detector_ids.size());
  for (uint32_t detector : result.e2d_detector_ids) {
    if (detector >= result.num_detectors) {
      throw std::runtime_error("trace error detector ID is out of range");
    }
  }
  std::vector<size_t> seen(result.num_detectors,
                           std::numeric_limits<size_t>::max());
  for (size_t candidate = 0; candidate < num_candidates; ++candidate) {
    for (uint32_t pos = result.candidate_offsets[candidate];
         pos < result.candidate_offsets[candidate + 1]; ++pos) {
      uint32_t detector = result.candidate_detector_ids[pos];
      if (detector >= result.num_detectors) {
        throw std::runtime_error("trace candidate detector ID is out of range");
      }
      if (seen[detector] == candidate) {
        throw std::runtime_error("trace candidate repeats a detector ID");
      }
      seen[detector] = candidate;
    }
  }
  if (!result.candidate_costs_q.empty() &&
      result.candidate_costs_q.size() != num_candidates) {
    throw std::runtime_error("trace expected-cost count does not match candidates");
  }
  const size_t blocked_words_per_candidate = (num_errors + 63) / 64;
  if (!result.candidate_blocked_words.empty() &&
      (num_candidates > std::numeric_limits<size_t>::max() /
                            blocked_words_per_candidate ||
       result.candidate_blocked_words.size() !=
           num_candidates * blocked_words_per_candidate)) {
    throw std::runtime_error("invalid candidate blocked-error bitsets");
  }
  return result;
}

struct PreparedTrace {
  Trace trace;
  std::vector<uint32_t> d2e_offsets;
  std::vector<uint32_t> d2e_error_ids;
  std::vector<uint32_t> cost_lut_offsets;
  std::vector<uint32_t> cost_lut_values;
  std::vector<uint8_t> static_mram;
  uint32_t d2e_offsets_offset = 0;
  uint32_t d2e_error_ids_offset = 0;
  uint32_t e2d_offsets_offset = 0;
  uint32_t e2d_detector_ids_offset = 0;
  uint32_t cost_lut_offsets_offset = 0;
  uint32_t cost_lut_values_offset = 0;
  bool sparse_blocked = false;
};

double live_thread_cpu_seconds() {
  timespec ts{};
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
  return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
}

struct LiveHandle {
  PreparedTrace prepared;
  dpu_set_t dpu{};
  bool allocated = false;
  void (*before_dpu_wait)(void*) = nullptr;
  void (*after_dpu_wait)(void*) = nullptr;
  void* dpu_wait_context = nullptr;
  // Host thread-CPU seconds per live phase, reported at destroy when
  // TESSERACT_DPU_LIVE_STATS is set.
  double cpu_input_copy = 0;
  double cpu_prepare = 0;
  double cpu_h2d = 0;
  double cpu_launch = 0;
  double cpu_d2h = 0;
  uint64_t batches = 0;
  uint64_t candidates = 0;

  ~LiveHandle() {
    if (std::getenv("TESSERACT_DPU_LIVE_STATS") != nullptr) {
      std::cerr << "Tesseract DPU bridge: batches=" << batches << " candidates=" << candidates
                << " cpu_input_copy=" << cpu_input_copy << " cpu_prepare=" << cpu_prepare
                << " cpu_h2d=" << cpu_h2d << " cpu_launch=" << cpu_launch
                << " cpu_d2h=" << cpu_d2h << std::endl;
    }
    if (allocated) {
      dpu_free(dpu);
    }
  }
};

PreparedTrace prepare_static(Trace trace) {
  const size_t num_errors = trace.error_costs_q.size();
  PreparedTrace result;
  result.trace = std::move(trace);
  result.d2e_offsets.assign(result.trace.num_detectors + 1, 0);
  for (uint32_t detector : result.trace.e2d_detector_ids) {
    if (detector >= result.trace.num_detectors) {
      throw std::runtime_error("error detector ID is out of range");
    }
    ++result.d2e_offsets[detector + 1];
  }
  for (size_t detector = 0; detector < result.trace.num_detectors; ++detector) {
    result.d2e_offsets[detector + 1] += result.d2e_offsets[detector];
  }
  result.d2e_error_ids.resize(result.trace.e2d_detector_ids.size());
  std::vector<uint32_t> cursors(result.d2e_offsets.begin(), result.d2e_offsets.end() - 1);
  for (size_t error = 0; error < num_errors; ++error) {
    for (uint32_t pos = result.trace.e2d_offsets[error];
         pos < result.trace.e2d_offsets[error + 1]; ++pos) {
      uint32_t detector = result.trace.e2d_detector_ids[pos];
      result.d2e_error_ids[cursors[detector]++] = static_cast<uint32_t>(error);
    }
  }
  // Tesseract's detector heuristic only needs the cheapest incident error.
  // Order each detector's list by the best cost an error could possibly have
  // (all of its detectors overlapping).  The DPU can then stop once this lower
  // bound cannot improve the current minimum.
  for (size_t detector = 0; detector < result.trace.num_detectors; ++detector) {
    auto begin = result.d2e_error_ids.begin() + result.d2e_offsets[detector];
    auto end = result.d2e_error_ids.begin() + result.d2e_offsets[detector + 1];
    std::sort(begin, end, [&result](uint32_t lhs, uint32_t rhs) {
      const uint32_t lhs_degree =
          result.trace.e2d_offsets[lhs + 1] - result.trace.e2d_offsets[lhs];
      const uint32_t rhs_degree =
          result.trace.e2d_offsets[rhs + 1] - result.trace.e2d_offsets[rhs];
      const uint64_t lhs_scaled =
          static_cast<uint64_t>(result.trace.error_costs_q[lhs]) * rhs_degree;
      const uint64_t rhs_scaled =
          static_cast<uint64_t>(result.trace.error_costs_q[rhs]) * lhs_degree;
      return lhs_scaled != rhs_scaled ? lhs_scaled < rhs_scaled : lhs < rhs;
    });
  }

  result.cost_lut_offsets.reserve(num_errors + 1);
  result.cost_lut_offsets.push_back(0);
  for (size_t error = 0; error < num_errors; ++error) {
    uint32_t degree = result.trace.e2d_offsets[error + 1] - result.trace.e2d_offsets[error];
    result.cost_lut_values.push_back(kInfCost);
    for (uint32_t overlap = 1; overlap <= degree; ++overlap) {
      uint64_t cost = (static_cast<uint64_t>(result.trace.error_costs_q[error]) +
                       overlap / 2) /
                      overlap;
      result.cost_lut_values.push_back(
          static_cast<uint32_t>(std::min<uint64_t>(cost, kInfCost - 1)));
    }
    result.cost_lut_offsets.push_back(
        static_cast<uint32_t>(result.cost_lut_values.size()));
  }

  if (!result.trace.candidate_blocked_words.empty()) {
    result.sparse_blocked = num_errors <= 65536;
    const size_t blocked_stride = (num_errors + 63) / 64;
    const size_t num_candidates = result.trace.candidate_offsets.size() - 1;
    for (size_t candidate = 0;
         result.sparse_blocked && candidate < num_candidates;
         ++candidate) {
      size_t count = 0;
      for (size_t word = 0; word < blocked_stride; ++word) {
        count += static_cast<size_t>(__builtin_popcountll(
            result.trace.candidate_blocked_words[candidate * blocked_stride + word]));
      }
      if (count > TESSERACT_MAX_SPARSE_BLOCKED) {
        result.sparse_blocked = false;
        break;
      }
    }
  }

  if (result.trace.candidate_costs_q.empty()) {
    const size_t num_candidates = result.trace.candidate_offsets.size() - 1;
    const size_t blocked_stride = (num_errors + 63) / 64;
    result.trace.candidate_costs_q.resize(num_candidates);
    std::vector<uint8_t> residual(result.trace.num_detectors);
    for (size_t candidate = 0; candidate < num_candidates; ++candidate) {
      std::fill(residual.begin(), residual.end(), 0);
      for (uint32_t pos = result.trace.candidate_offsets[candidate];
           pos < result.trace.candidate_offsets[candidate + 1]; ++pos) {
        residual[result.trace.candidate_detector_ids[pos]] = 1;
      }
      uint64_t total = 0;
      for (uint32_t pos = result.trace.candidate_offsets[candidate];
           pos < result.trace.candidate_offsets[candidate + 1]; ++pos) {
        uint32_t detector = result.trace.candidate_detector_ids[pos];
        uint32_t minimum = kInfCost;
        for (uint32_t error_pos = result.d2e_offsets[detector];
             error_pos < result.d2e_offsets[detector + 1]; ++error_pos) {
          uint32_t error = result.d2e_error_ids[error_pos];
          if (!result.trace.candidate_blocked_words.empty() &&
              ((result.trace.candidate_blocked_words[
                    candidate * blocked_stride + error / 64] >>
                (error & 63)) & 1) != 0) {
            continue;
          }
          uint32_t overlap = 0;
          for (uint32_t detector_pos = result.trace.e2d_offsets[error];
               detector_pos < result.trace.e2d_offsets[error + 1]; ++detector_pos) {
            overlap += residual[result.trace.e2d_detector_ids[detector_pos]];
          }
          if (overlap != 0) {
            minimum = std::min(
                minimum,
                result.cost_lut_values[result.cost_lut_offsets[error] + overlap]);
          }
        }
        if (minimum == kInfCost ||
            static_cast<uint64_t>(minimum) + result.trace.detector_penalty_q >= kInfCost ||
            total + minimum + result.trace.detector_penalty_q >= kInfCost) {
          total = kInfCost;
          break;
        }
        total += minimum + result.trace.detector_penalty_q;
      }
      result.trace.candidate_costs_q[candidate] = static_cast<uint32_t>(total);
    }
  }

  result.d2e_offsets_offset = append_u32(result.static_mram, result.d2e_offsets);
  result.d2e_error_ids_offset = append_u32(result.static_mram, result.d2e_error_ids);
  result.e2d_offsets_offset = append_u32(result.static_mram, result.trace.e2d_offsets);
  result.e2d_detector_ids_offset =
      append_u32(result.static_mram, result.trace.e2d_detector_ids);
  result.cost_lut_offsets_offset =
      append_u32(result.static_mram, result.cost_lut_offsets);
  result.cost_lut_values_offset =
      append_u32(result.static_mram, result.cost_lut_values);
  return result;
}

void align8(std::vector<uint8_t>& bytes) {
  while ((bytes.size() & 7) != 0) {
    bytes.push_back(0);
  }
}

uint32_t append_u32(std::vector<uint8_t>& bytes, const std::vector<uint32_t>& values) {
  align8(bytes);
  const size_t offset = bytes.size();
  if (offset > std::numeric_limits<uint32_t>::max() ||
      values.size() >
          (std::numeric_limits<uint32_t>::max() - offset) / sizeof(uint32_t)) {
    throw std::runtime_error("DPU MRAM layout exceeds 32-bit offsets");
  }
  bytes.resize(offset + values.size() * sizeof(uint32_t));
  uint8_t* destination = bytes.data() + offset;
  for (size_t index = 0; index < values.size(); ++index) {
    uint32_t value = values[index];
    destination[index * 4] = static_cast<uint8_t>(value);
    destination[index * 4 + 1] = static_cast<uint8_t>(value >> 8);
    destination[index * 4 + 2] = static_cast<uint8_t>(value >> 16);
    destination[index * 4 + 3] = static_cast<uint8_t>(value >> 24);
  }
  align8(bytes);
  return static_cast<uint32_t>(offset);
}

struct Batch {
  size_t global_begin;
  size_t count;
  uint32_t static_bytes;
  dpu_arguments_t arguments{};
  std::vector<uint8_t> mram;
  std::vector<uint32_t> expected;
};

Batch prepare_batch(const PreparedTrace& source, size_t begin, size_t end) {
  const Trace& trace = source.trace;
  const size_t num_errors = trace.error_costs_q.size();
  Batch batch;
  batch.global_begin = begin;
  batch.count = end - begin;

  std::vector<uint32_t> candidate_offsets{0};
  std::vector<uint32_t> candidate_ids;
  for (size_t candidate = begin; candidate < end; ++candidate) {
    for (uint32_t pos = trace.candidate_offsets[candidate];
         pos < trace.candidate_offsets[candidate + 1]; ++pos) {
      candidate_ids.push_back(trace.candidate_detector_ids[pos]);
    }
    candidate_offsets.push_back(static_cast<uint32_t>(candidate_ids.size()));
  }

  const size_t residual_words = (trace.num_detectors + 31) / 32;
  const size_t residual_stride = (residual_words + 1) & ~size_t{1};
  std::vector<uint32_t> residual(batch.count * residual_stride, 0);
  for (size_t local = 0; local < batch.count; ++local) {
    for (uint32_t pos = candidate_offsets[local]; pos < candidate_offsets[local + 1]; ++pos) {
      uint32_t detector = candidate_ids[pos];
      residual[local * residual_stride + detector / 32] |= uint32_t{1} << (detector & 31);
    }
  }

  std::vector<uint32_t> blocked;
  std::vector<uint32_t> blocked_record_offsets;
  std::vector<uint32_t> blocked_records;
  if (source.sparse_blocked) {
    const size_t source_stride = (num_errors + 63) / 64;
    blocked_record_offsets.reserve(batch.count);
    for (size_t candidate = begin; candidate < end; ++candidate) {
      blocked_record_offsets.push_back(static_cast<uint32_t>(blocked_records.size()));
      size_t count_index = blocked_records.size();
      blocked_records.push_back(0);
      blocked_records.push_back(0);
      bool low_half = true;
      uint32_t count = 0;
      for (size_t word_index = 0; word_index < source_stride; ++word_index) {
        uint64_t word = trace.candidate_blocked_words[
            candidate * source_stride + word_index];
        while (word != 0) {
          unsigned bit = static_cast<unsigned>(__builtin_ctzll(word));
          size_t error = word_index * 64 + bit;
          if (error < num_errors) {
            if (low_half) {
              blocked_records.push_back(static_cast<uint32_t>(error));
            } else {
              blocked_records.back() |= static_cast<uint32_t>(error) << 16;
            }
            low_half = !low_half;
            ++count;
          }
          word &= word - 1;
        }
      }
      blocked_records[count_index] = count;
      if ((blocked_records.size() & 1) != 0) {
        blocked_records.push_back(0);
      }
    }
  } else if (!trace.candidate_blocked_words.empty()) {
    const size_t source_stride = (num_errors + 63) / 64;
    const size_t destination_stride = (num_errors + 31) / 32;
    blocked.reserve(batch.count * destination_stride);
    for (size_t candidate = begin; candidate < end; ++candidate) {
      for (size_t word32 = 0; word32 < destination_stride; ++word32) {
        uint64_t value = trace.candidate_blocked_words[
            candidate * source_stride + word32 / 2];
        blocked.push_back(static_cast<uint32_t>(value >> ((word32 & 1) * 32)));
      }
    }
  }

  if (!trace.candidate_costs_q.empty()) {
    batch.expected.assign(trace.candidate_costs_q.begin() + begin,
                          trace.candidate_costs_q.begin() + end);
  }
  batch.static_bytes = static_cast<uint32_t>(source.static_mram.size());
  auto append_dynamic = [&batch](const std::vector<uint32_t>& values) {
    uint32_t relative = append_u32(batch.mram, values);
    if (relative > std::numeric_limits<uint32_t>::max() - batch.static_bytes) {
      throw std::runtime_error("DPU MRAM layout exceeds 32-bit offsets");
    }
    return batch.static_bytes + relative;
  };
  uint32_t candidate_offsets_offset = append_dynamic(candidate_offsets);
  uint32_t candidate_ids_offset = append_dynamic(candidate_ids);
  uint32_t residual_offset = append_dynamic(residual);
  uint32_t blocked_offset = kNoBlockedWords;
  if (!blocked.empty()) {
    blocked_offset = append_dynamic(blocked);
  }
  uint32_t blocked_record_offsets_offset = kNoBlockedWords;
  uint32_t blocked_records_offset = kNoBlockedWords;
  if (!blocked_records.empty()) {
    blocked_record_offsets_offset = append_dynamic(blocked_record_offsets);
    blocked_records_offset = append_dynamic(blocked_records);
  }
  align8(batch.mram);
  if (batch.mram.size() >
      std::numeric_limits<uint32_t>::max() - batch.static_bytes) {
    throw std::runtime_error("DPU MRAM layout exceeds 32-bit offsets");
  }
  uint32_t output_offset =
      batch.static_bytes + static_cast<uint32_t>(batch.mram.size());
  if (output_offset > kDpuMramBytes ||
      batch.count > (kDpuMramBytes - output_offset) / sizeof(uint64_t)) {
    throw std::runtime_error("candidate batch exceeds the DPU MRAM capacity");
  }

  batch.arguments.num_detectors = trace.num_detectors;
  batch.arguments.num_errors = static_cast<uint32_t>(num_errors);
  batch.arguments.num_candidates = static_cast<uint32_t>(batch.count);
  batch.arguments.num_d2e_ids = static_cast<uint32_t>(source.d2e_error_ids.size());
  batch.arguments.num_e2d_ids = static_cast<uint32_t>(trace.e2d_detector_ids.size());
  batch.arguments.d2e_offsets_offset = source.d2e_offsets_offset;
  batch.arguments.d2e_error_ids_offset = source.d2e_error_ids_offset;
  batch.arguments.e2d_offsets_offset = source.e2d_offsets_offset;
  batch.arguments.e2d_detector_ids_offset = source.e2d_detector_ids_offset;
  batch.arguments.cost_lut_offsets_offset = source.cost_lut_offsets_offset;
  batch.arguments.cost_lut_values_offset = source.cost_lut_values_offset;
  batch.arguments.candidate_offsets_offset = candidate_offsets_offset;
  batch.arguments.candidate_detector_ids_offset = candidate_ids_offset;
  batch.arguments.candidate_residual_words_offset = residual_offset;
  batch.arguments.blocked_words_offset = blocked_offset;
  batch.arguments.blocked_record_offsets_offset = blocked_record_offsets_offset;
  batch.arguments.blocked_records_offset = blocked_records_offset;
  batch.arguments.output_costs_offset = output_offset;
  batch.arguments.detector_penalty_q = trace.detector_penalty_q;
  batch.arguments.q_bits = trace.q_bits;
  batch.arguments.kernel = dpu_arguments_t::kernel1;
  return batch;
}

struct Timings {
  double packing = 0;
  double h2d = 0;
  double launch = 0;
  double wait = 0;
  double d2h_and_integrate = 0;
  uint64_t h2d_bytes = 0;
  uint64_t d2h_bytes = 0;
  size_t batches = 0;
};

struct Slot {
  dpu_set_t dpu{};
  bool allocated = false;
  bool in_flight = false;
  bool static_model_loaded = false;
  std::unique_ptr<Batch> batch;
  std::vector<uint64_t> output;

  ~Slot() {
    if (allocated) {
      dpu_free(dpu);
    }
  }
};

void finish_slot(Slot& slot, std::vector<uint32_t>& integrated, Timings& timings) {
  if (!slot.in_flight) {
    return;
  }
  auto wait_start = Clock::now();
  check_dpu(dpu_sync(slot.dpu), "dpu_sync");
  timings.wait += seconds_since(wait_start);

  auto read_start = Clock::now();
  slot.output.resize(slot.batch->count);
  check_dpu(dpu_prepare_xfer(slot.dpu, slot.output.data()),
            "prepare DPU-to-host output transfer");
  check_dpu(dpu_push_xfer(slot.dpu,
                          DPU_XFER_FROM_DPU,
                          DPU_MRAM_HEAP_POINTER_NAME,
                          slot.batch->arguments.output_costs_offset,
                          slot.output.size() * sizeof(uint64_t),
                          DPU_XFER_DEFAULT),
            "DPU-to-host output transfer");
  timings.d2h_bytes += slot.output.size() * sizeof(uint64_t);
  for (size_t local = 0; local < slot.output.size(); ++local) {
    uint64_t expected = slot.batch->expected[local];
    if (slot.output[local] != expected) {
      throw std::runtime_error(
          "DPU mismatch at candidate " +
          std::to_string(slot.batch->global_begin + local) + ": expected " +
          std::to_string(expected) + ", got " +
          std::to_string(slot.output[local]));
    }
    integrated[slot.batch->global_begin + local] =
        static_cast<uint32_t>(slot.output[local]);
  }
  timings.d2h_and_integrate += seconds_since(read_start);
  slot.batch.reset();
  slot.in_flight = false;
}

struct RunResult {
  double initialization_seconds;
  double end_to_end_seconds;
  Timings timings;
};

[[maybe_unused]] RunResult run_pipeline(const PreparedTrace& prepared,
                                        const std::string& dpu_binary,
                                        size_t batch_size,
                                        size_t repetitions) {
  auto initialization_start = Clock::now();
  std::vector<std::unique_ptr<Slot>> slots;
  for (size_t index = 0; index < 2; ++index) {
    auto slot = std::make_unique<Slot>();
    check_dpu(dpu_alloc(1, nullptr, &slot->dpu), "dpu_alloc");
    slot->allocated = true;
    check_dpu(dpu_load(slot->dpu, dpu_binary.c_str(), nullptr), "dpu_load");
    slots.push_back(std::move(slot));
  }
  RunResult result{};
  result.initialization_seconds = seconds_since(initialization_start);

  const size_t candidates = prepared.trace.candidate_offsets.size() - 1;
  std::vector<uint32_t> integrated(candidates);
  auto end_to_end_start = Clock::now();
  for (size_t repetition = 0; repetition < repetitions; ++repetition) {
    std::fill(integrated.begin(), integrated.end(), 0);
    size_t sequence = 0;
    for (size_t begin = 0; begin < candidates; begin += batch_size, ++sequence) {
      size_t end = std::min(candidates, begin + batch_size);
      auto packing_start = Clock::now();
      auto batch = std::make_unique<Batch>(prepare_batch(prepared, begin, end));
      result.timings.packing += seconds_since(packing_start);

      Slot& slot = *slots[sequence % slots.size()];
      finish_slot(slot, integrated, result.timings);
      auto h2d_start = Clock::now();
      check_dpu(dpu_copy_to(slot.dpu,
                            "DPU_INPUT_ARGUMENTS",
                            0,
                            &batch->arguments,
                            sizeof(batch->arguments)),
                "host-to-DPU argument transfer");
      if (!slot.static_model_loaded) {
        check_dpu(dpu_copy_to(slot.dpu,
                              DPU_MRAM_HEAP_POINTER_NAME,
                              0,
                              prepared.static_mram.data(),
                              prepared.static_mram.size()),
                  "static model host-to-DPU transfer");
        result.timings.h2d_bytes += prepared.static_mram.size();
      }
      check_dpu(dpu_copy_to(slot.dpu,
                            DPU_MRAM_HEAP_POINTER_NAME,
                            batch->static_bytes,
                            batch->mram.data(),
                            batch->mram.size()),
                "dynamic batch host-to-DPU transfer");
      result.timings.h2d += seconds_since(h2d_start);
      result.timings.h2d_bytes += sizeof(batch->arguments) + batch->mram.size();
      slot.static_model_loaded = true;

      auto launch_start = Clock::now();
      check_dpu(dpu_launch(slot.dpu, DPU_ASYNCHRONOUS), "asynchronous DPU launch");
      result.timings.launch += seconds_since(launch_start);
      slot.batch = std::move(batch);
      slot.in_flight = true;
      ++result.timings.batches;
    }
    for (auto& slot : slots) {
      finish_slot(*slot, integrated, result.timings);
    }
    if (integrated != prepared.trace.candidate_costs_q) {
      throw std::runtime_error("integrated candidate output does not match producer trace");
    }
  }
  result.end_to_end_seconds = seconds_since(end_to_end_start);
  return result;
}

[[maybe_unused]] void write_csv(const std::string& path,
                                const std::string& trace_path,
                                const char* blocked_layout,
                                size_t candidates,
                                size_t batch_size,
                                size_t repetitions,
                                const RunResult& result) {
  std::ofstream output(path, std::ios::trunc);
  if (!output) {
    throw std::runtime_error("cannot write result CSV: " + path);
  }
  output << std::setprecision(12);
  output << "metric,value\n";
  output << "mode,trace_replay_async_dpu_double_buffer\n";
  output << "trace,\"" << trace_path << "\"\n";
  output << "blocked_layout," << blocked_layout << "\n";
  output << "slots,2\n";
  output << "candidates," << candidates << "\n";
  output << "batch_size," << batch_size << "\n";
  output << "repetitions," << repetitions << "\n";
  output << "batches," << result.timings.batches << "\n";
  output << "initialization_seconds," << result.initialization_seconds << "\n";
  output << "end_to_end_seconds," << result.end_to_end_seconds << "\n";
  output << "candidate_throughput_per_second,"
         << (candidates * repetitions / result.end_to_end_seconds) << "\n";
  output << "packing_seconds," << result.timings.packing << "\n";
  output << "h2d_seconds," << result.timings.h2d << "\n";
  output << "async_launch_seconds," << result.timings.launch << "\n";
  output << "blocking_wait_seconds," << result.timings.wait << "\n";
  output << "d2h_and_integration_seconds," << result.timings.d2h_and_integrate << "\n";
  output << "h2d_bytes," << result.timings.h2d_bytes << "\n";
  output << "d2h_bytes," << result.timings.d2h_bytes << "\n";
}

void set_abi_error(char* destination, size_t capacity, const std::string& message) {
  if (destination == nullptr || capacity == 0) {
    return;
  }
  const size_t count = std::min(capacity - 1, message.size());
  std::memcpy(destination, message.data(), count);
  destination[count] = '\0';
}

std::vector<uint32_t> copy_u32_input(const uint32_t* source,
                                     size_t count,
                                     const char* name) {
  if (count != 0 && source == nullptr) {
    throw std::runtime_error(std::string(name) + " pointer is null");
  }
  return count == 0 ? std::vector<uint32_t>{}
                    : std::vector<uint32_t>(source, source + count);
}

std::vector<uint64_t> copy_u64_input(const uint64_t* source,
                                     size_t count,
                                     const char* name) {
  if (count != 0 && source == nullptr) {
    throw std::runtime_error(std::string(name) + " pointer is null");
  }
  return count == 0 ? std::vector<uint64_t>{}
                    : std::vector<uint64_t>(source, source + count);
}

}  // namespace

extern "C" void* tesseract_dpu_live_create(
    const tesseract_dpu_live_config_t* config,
    char* error_message,
    size_t error_message_capacity) {
  try {
    if (config == nullptr) {
      throw std::runtime_error("configuration pointer is null");
    }
    if (config->q_bits > 24) {
      throw std::runtime_error("q_bits must be in [0,24]");
    }
    if (config->num_detectors == 0 || config->num_detectors > TESSERACT_MAX_DETECTORS) {
      throw std::runtime_error("detector count is outside the DPU limit");
    }
    if (config->num_errors == 0) {
      throw std::runtime_error("the DPU model must contain at least one error");
    }
    if (config->num_error_detector_offsets !=
        static_cast<size_t>(config->num_errors) + 1) {
      throw std::runtime_error("error-to-detector offsets have the wrong length");
    }
    if (config->num_error_costs != config->num_errors) {
      throw std::runtime_error("error cost array has the wrong length");
    }
    if (config->dpu_binary == nullptr || config->dpu_binary[0] == '\0') {
      throw std::runtime_error("DPU binary path is empty");
    }

    Trace trace{};
    trace.q_bits = config->q_bits;
    trace.detector_penalty_q = config->detector_penalty_q;
    trace.num_detectors = config->num_detectors;
    trace.e2d_offsets = copy_u32_input(config->error_detector_offsets,
                                       config->num_error_detector_offsets,
                                       "error-to-detector offsets");
    trace.e2d_detector_ids = copy_u32_input(config->error_detector_ids,
                                            config->num_error_detector_ids,
                                            "error-to-detector IDs");
    trace.error_costs_q = copy_u32_input(config->error_costs_q,
                                         config->num_error_costs,
                                         "error costs");
    trace.candidate_offsets = {0};
    validate_offsets("error-to-detector", trace.e2d_offsets, config->num_errors,
                     trace.e2d_detector_ids.size());
    for (uint32_t detector : trace.e2d_detector_ids) {
      if (detector >= trace.num_detectors) {
        throw std::runtime_error("error-to-detector ID is out of range");
      }
    }
    for (uint32_t cost : trace.error_costs_q) {
      if (cost == kInfCost) {
        throw std::runtime_error("error cost uses the reserved INF value");
      }
    }

    auto handle = std::make_unique<LiveHandle>();
    handle->prepared = prepare_static(std::move(trace));
    handle->before_dpu_wait = config->before_dpu_wait;
    handle->after_dpu_wait = config->after_dpu_wait;
    handle->dpu_wait_context = config->dpu_wait_context;
    if (handle->prepared.static_mram.size() > kDpuMramBytes) {
      throw std::runtime_error("static DPU model exceeds MRAM capacity");
    }
    check_dpu(dpu_alloc(1, nullptr, &handle->dpu), "dpu_alloc");
    handle->allocated = true;
    check_dpu(dpu_load(handle->dpu, config->dpu_binary, nullptr), "dpu_load");
    check_dpu(dpu_copy_to(handle->dpu,
                          DPU_MRAM_HEAP_POINTER_NAME,
                          0,
                          handle->prepared.static_mram.data(),
                          handle->prepared.static_mram.size()),
              "static model host-to-DPU transfer");
    set_abi_error(error_message, error_message_capacity, "");
    return handle.release();
  } catch (const std::exception& ex) {
    set_abi_error(error_message, error_message_capacity, ex.what());
    return nullptr;
  }
}

extern "C" int tesseract_dpu_live_evaluate(
    void* opaque_handle,
    const tesseract_dpu_live_batch_t* input,
    uint32_t* output_costs_q,
    size_t output_count,
    char* error_message,
    size_t error_message_capacity) {
  try {
    if (opaque_handle == nullptr || input == nullptr) {
      throw std::runtime_error("live handle or batch pointer is null");
    }
    if (output_count != input->num_candidates) {
      throw std::runtime_error("output array has the wrong length");
    }
    if (output_count != 0 && output_costs_q == nullptr) {
      throw std::runtime_error("output pointer is null");
    }
    if (input->num_candidate_detector_offsets !=
        static_cast<size_t>(input->num_candidates) + 1) {
      throw std::runtime_error("candidate offsets have the wrong length");
    }
    if (input->num_candidates == 0) {
      set_abi_error(error_message, error_message_capacity, "");
      return 0;
    }

    auto* handle = static_cast<LiveHandle*>(opaque_handle);
    Trace& trace = handle->prepared.trace;
    double phase_start = live_thread_cpu_seconds();
    auto end_phase = [&phase_start](double& accumulator) {
      const double now = live_thread_cpu_seconds();
      accumulator += now - phase_start;
      phase_start = now;
    };
    trace.candidate_offsets = copy_u32_input(input->candidate_detector_offsets,
                                             input->num_candidate_detector_offsets,
                                             "candidate offsets");
    trace.candidate_detector_ids = copy_u32_input(input->candidate_detector_ids,
                                                  input->num_candidate_detector_ids,
                                                  "candidate detector IDs");
    trace.candidate_blocked_words = copy_u64_input(
        input->candidate_blocked_error_words,
        input->num_candidate_blocked_error_words,
        "candidate blocked-error words");
    trace.candidate_costs_q.clear();
    validate_offsets("candidate-to-detector", trace.candidate_offsets,
                     input->num_candidates, trace.candidate_detector_ids.size());

    std::vector<size_t> seen(trace.num_detectors, std::numeric_limits<size_t>::max());
    for (size_t candidate = 0; candidate < input->num_candidates; ++candidate) {
      for (uint32_t pos = trace.candidate_offsets[candidate];
           pos < trace.candidate_offsets[candidate + 1]; ++pos) {
        const uint32_t detector = trace.candidate_detector_ids[pos];
        if (detector >= trace.num_detectors) {
          throw std::runtime_error("candidate detector ID is out of range");
        }
        if (seen[detector] == candidate) {
          throw std::runtime_error("candidate repeats a detector ID");
        }
        seen[detector] = candidate;
      }
    }

    const size_t num_errors = trace.error_costs_q.size();
    const size_t blocked_stride = (num_errors + 63) / 64;
    if (input->num_candidates > std::numeric_limits<size_t>::max() / blocked_stride ||
        trace.candidate_blocked_words.size() != input->num_candidates * blocked_stride) {
      throw std::runtime_error("candidate blocked-error bitsets have the wrong length");
    }
    handle->prepared.sparse_blocked = num_errors <= 65536;
    for (size_t candidate = 0;
         handle->prepared.sparse_blocked && candidate < input->num_candidates;
         ++candidate) {
      size_t blocked_count = 0;
      for (size_t word = 0; word < blocked_stride; ++word) {
        blocked_count += static_cast<size_t>(__builtin_popcountll(
            trace.candidate_blocked_words[candidate * blocked_stride + word]));
      }
      if (blocked_count > TESSERACT_MAX_SPARSE_BLOCKED) {
        handle->prepared.sparse_blocked = false;
      }
    }

    end_phase(handle->cpu_input_copy);
    Batch batch = prepare_batch(handle->prepared, 0, input->num_candidates);
    end_phase(handle->cpu_prepare);
    check_dpu(dpu_copy_to(handle->dpu,
                          "DPU_INPUT_ARGUMENTS",
                          0,
                          &batch.arguments,
                          sizeof(batch.arguments)),
              "live argument host-to-DPU transfer");
    check_dpu(dpu_copy_to(handle->dpu,
                          DPU_MRAM_HEAP_POINTER_NAME,
                          batch.static_bytes,
                          batch.mram.data(),
                          batch.mram.size()),
              "live batch host-to-DPU transfer");
    end_phase(handle->cpu_h2d);
    if (handle->before_dpu_wait != nullptr) {
      handle->before_dpu_wait(handle->dpu_wait_context);
    }
    try {
      check_dpu(dpu_launch(handle->dpu, DPU_SYNCHRONOUS), "live DPU launch");
    } catch (...) {
      if (handle->after_dpu_wait != nullptr) {
        handle->after_dpu_wait(handle->dpu_wait_context);
      }
      throw;
    }
    if (handle->after_dpu_wait != nullptr) {
      handle->after_dpu_wait(handle->dpu_wait_context);
    }
    end_phase(handle->cpu_launch);

    std::vector<uint64_t> output64(input->num_candidates);
    check_dpu(dpu_prepare_xfer(handle->dpu, output64.data()),
              "prepare live DPU-to-host transfer");
    check_dpu(dpu_push_xfer(handle->dpu,
                            DPU_XFER_FROM_DPU,
                            DPU_MRAM_HEAP_POINTER_NAME,
                            batch.arguments.output_costs_offset,
                            output64.size() * sizeof(uint64_t),
                            DPU_XFER_DEFAULT),
              "live DPU-to-host transfer");
    for (size_t index = 0; index < output64.size(); ++index) {
      if (output64[index] > kInfCost) {
        throw std::runtime_error("DPU returned a cost outside the 32-bit ABI");
      }
      output_costs_q[index] = static_cast<uint32_t>(output64[index]);
    }
    end_phase(handle->cpu_d2h);
    ++handle->batches;
    handle->candidates += input->num_candidates;
    set_abi_error(error_message, error_message_capacity, "");
    return 0;
  } catch (const std::exception& ex) {
    set_abi_error(error_message, error_message_capacity, ex.what());
    return 1;
  }
}

extern "C" void tesseract_dpu_live_destroy(void* opaque_handle) {
  delete static_cast<LiveHandle*>(opaque_handle);
}

// ---------------------------------------------------------------------------
// DPU-resident A* search (TESSERACT_ASTAR kernel).

namespace {

constexpr uint32_t kAstarStatusFallback = 2;
constexpr size_t kAstarMaxDepth = 256;
constexpr size_t kAstarNodeBytes = 16;
constexpr size_t kAstarArenaBytes = 8;
constexpr size_t kAstarOutputBytes = 48 + kAstarMaxDepth * 2;
constexpr size_t kAstarDefaultCapacity = 1000000;
constexpr size_t kAstarVisitedEntryBytes = 80;
constexpr size_t kAstarDefaultVisitedCapacity = 131072;

// The ABI always carries two group slots, but the production bridge assigns
// exactly one syndrome to each DPU.  Keeping these constants separate avoids
// silently reserving half of MRAM for an idle second group.
constexpr uint32_t kAstarAbiGroups = 2;
constexpr uint32_t kAstarActiveGroups = 1;

struct AstarArguments {
  uint32_t num_detectors;
  uint32_t num_errors;
  uint32_t d2e_offsets_offset;
  uint32_t d2e_error_ids_offset;
  uint32_t e2d_offsets_offset;
  uint32_t e2d_detector_ids_offset;
  uint32_t cost_lut_offsets_offset;
  uint32_t cost_lut_values_offset;
  uint32_t error_records_offset;
  uint32_t eneighbor_offsets_offset;
  uint32_t eneighbor_ids_offset;
  uint32_t detector_penalty_q;
  uint32_t det_beam;
  uint32_t pqlimit;
  uint32_t no_revisit_dets;
  uint32_t max_expansions;
  uint32_t num_syndromes;
  uint32_t heap_capacity;
  uint32_t arena_capacity;
  uint32_t syndrome_words_offset[kAstarAbiGroups];
  uint32_t heap_offset[kAstarAbiGroups];
  uint32_t arena_offset[kAstarAbiGroups];
  uint32_t output_offset[kAstarAbiGroups];
  uint32_t visited_offset[kAstarAbiGroups];
  uint32_t visited_capacity;
  uint32_t visited_generation;
  uint32_t order_positions_offset;
  uint32_t order_index[kAstarAbiGroups];
  uint32_t kernel;
};

struct ResidentHandle {
  PreparedTrace prepared;
  std::vector<uint8_t> static_image;
  AstarArguments arguments{};
  std::vector<uint32_t> syndrome_words;
  dpu_set_t dpu{};
  bool allocated = false;
  void (*before_dpu_wait)(void*) = nullptr;
  void (*after_dpu_wait)(void*) = nullptr;
  void* dpu_wait_context = nullptr;
  double cpu_pack = 0;
  double cpu_h2d = 0;
  double cpu_launch = 0;
  double cpu_d2h = 0;
  uint64_t shots = 0;
  uint64_t fallbacks = 0;
  uint32_t num_orders = 1;
  uint32_t visited_generation = 0;

  ~ResidentHandle() {
    if (std::getenv("TESSERACT_DPU_LIVE_STATS") != nullptr) {
      std::cerr << "Tesseract DPU resident: shots=" << shots << " fallbacks=" << fallbacks
                << " cpu_pack=" << cpu_pack << " cpu_h2d=" << cpu_h2d
                << " cpu_launch=" << cpu_launch << " cpu_d2h=" << cpu_d2h << std::endl;
    }
    if (allocated) {
      dpu_free(dpu);
    }
  }
};

uint32_t align8(size_t value) {
  return static_cast<uint32_t>((value + 7) & ~size_t{7});
}

}  // namespace

extern "C" void* tesseract_dpu_resident_create(
    const tesseract_dpu_live_config_t* config,
    uint32_t det_beam,
    uint32_t pqlimit,
    uint32_t no_revisit_dets,
    uint32_t max_expansions,
    char* error_message,
    size_t error_message_capacity) {
  try {
    if (config == nullptr) {
      throw std::runtime_error("configuration pointer is null");
    }
    Trace trace{};
    trace.q_bits = config->q_bits;
    trace.detector_penalty_q = config->detector_penalty_q;
    trace.num_detectors = config->num_detectors;
    trace.e2d_offsets = copy_u32_input(config->error_detector_offsets,
                                       config->num_error_detector_offsets,
                                       "error-to-detector offsets");
    trace.e2d_detector_ids = copy_u32_input(config->error_detector_ids,
                                            config->num_error_detector_ids,
                                            "error-to-detector IDs");
    trace.error_costs_q = copy_u32_input(config->error_costs_q, config->num_error_costs,
                                         "error costs");
    trace.candidate_offsets = {0};
    validate_offsets("error-to-detector", trace.e2d_offsets, config->num_errors,
                     trace.e2d_detector_ids.size());
    if (config->num_errors > 65535) {
      throw std::runtime_error("resident kernel stores error IDs as 16-bit values");
    }

    auto handle = std::make_unique<ResidentHandle>();
    handle->prepared = prepare_static(std::move(trace));
    handle->before_dpu_wait = config->before_dpu_wait;
    handle->after_dpu_wait = config->after_dpu_wait;
    handle->dpu_wait_context = config->dpu_wait_context;

    const PreparedTrace& prepared = handle->prepared;
    const size_t residual_words = (config->num_detectors + 31) / 32;
    const size_t residual_stride = (residual_words + 1) & ~size_t{1};
    handle->syndrome_words.assign(residual_stride, 0);

    // Static image = shared model regions + per-error 16-byte records.
    handle->static_image = prepared.static_mram;
    // 32-byte record: {lut_begin, cost_lut[e][degree], u16 degree, u16 detectors[11]}.
    constexpr uint32_t kMaxDegree = 11;
    std::vector<uint32_t> records;
    records.reserve(static_cast<size_t>(config->num_errors) * 8);
    for (uint32_t e = 0; e < config->num_errors; ++e) {
      const uint32_t begin = prepared.trace.e2d_offsets[e];
      const uint32_t end = prepared.trace.e2d_offsets[e + 1];
      const uint32_t degree = end - begin;
      if (degree > kMaxDegree) {
        throw std::runtime_error("resident kernel supports error degree <= 11");
      }
      const uint32_t lut_begin = prepared.cost_lut_offsets[e];
      records.push_back(lut_begin);
      records.push_back(prepared.cost_lut_values[lut_begin + degree]);
      uint16_t halves[kMaxDegree + 1] = {};
      halves[0] = static_cast<uint16_t>(degree);
      for (uint32_t k = 0; k < degree; ++k) {
        halves[1 + k] = static_cast<uint16_t>(prepared.trace.e2d_detector_ids[begin + k]);
      }
      for (uint32_t k = 0; k < kMaxDegree + 1; k += 2) {
        records.push_back(static_cast<uint32_t>(halves[k]) |
                          (static_cast<uint32_t>(halves[k + 1]) << 16));
      }
    }
    const uint32_t error_records_offset = append_u32(handle->static_image, records);

    // Tesseract's eneighbors: detectors of every error sharing a detector with
    // e, minus e's own detectors, sorted. Packed as u16 with a u32 offset table.
    std::vector<uint32_t> neighbor_offsets(config->num_errors + 1, 0);
    std::vector<uint16_t> neighbor_ids;
    std::vector<uint32_t> own(config->num_detectors, 0);
    std::vector<uint32_t> seen(config->num_detectors, 0);
    for (uint32_t e = 0; e < config->num_errors; ++e) {
      const uint32_t stamp = e + 1;
      const auto& e2d_offsets = prepared.trace.e2d_offsets;
      const auto& e2d_ids = prepared.trace.e2d_detector_ids;
      for (uint32_t pos = e2d_offsets[e]; pos < e2d_offsets[e + 1]; ++pos) {
        own[e2d_ids[pos]] = stamp;
      }
      const size_t begin = neighbor_ids.size();
      for (uint32_t pos = e2d_offsets[e]; pos < e2d_offsets[e + 1]; ++pos) {
        const uint32_t d = e2d_ids[pos];
        for (uint32_t epos = prepared.d2e_offsets[d]; epos < prepared.d2e_offsets[d + 1]; ++epos) {
          const uint32_t other = prepared.d2e_error_ids[epos];
          for (uint32_t opos = e2d_offsets[other]; opos < e2d_offsets[other + 1]; ++opos) {
            const uint32_t od = e2d_ids[opos];
            if (own[od] == stamp || seen[od] == stamp) {
              continue;
            }
            seen[od] = stamp;
            neighbor_ids.push_back(static_cast<uint16_t>(od));
          }
        }
      }
      std::sort(neighbor_ids.begin() + begin, neighbor_ids.end());
      neighbor_offsets[e + 1] = static_cast<uint32_t>(neighbor_ids.size());
    }
    std::vector<uint32_t> packed_neighbors((neighbor_ids.size() + 1) / 2, 0);
    for (size_t i = 0; i < neighbor_ids.size(); ++i) {
      packed_neighbors[i / 2] |= static_cast<uint32_t>(neighbor_ids[i]) << (16 * (i % 2));
    }
    const uint32_t neighbor_offsets_offset = append_u32(handle->static_image, neighbor_offsets);
    const uint32_t neighbor_ids_offset = append_u32(handle->static_image, packed_neighbors);
    uint32_t order_positions_offset = 0xFFFFFFFFu;
    handle->num_orders = 1;
    if (config->order_positions != nullptr && config->num_orders > 0) {
      handle->num_orders = config->num_orders;
      const size_t table_bytes = (static_cast<size_t>(config->num_detectors) * 2 + 7) & ~size_t{7};
      std::vector<uint32_t> tables(config->num_orders * table_bytes / 4, 0);
      for (uint32_t o = 0; o < config->num_orders; ++o) {
        for (uint32_t d = 0; d < config->num_detectors; ++d) {
          const uint16_t position = config->order_positions[o * config->num_detectors + d];
          const size_t byte = o * table_bytes + d * 2;
          tables[byte / 4] |= static_cast<uint32_t>(position) << (8 * (byte % 4));
        }
      }
      order_positions_offset = append_u32(handle->static_image, tables);
    }

    size_t cursor = handle->static_image.size();
    AstarArguments& args = handle->arguments;
    args.error_records_offset = error_records_offset;
    args.eneighbor_offsets_offset = neighbor_offsets_offset;
    args.eneighbor_ids_offset = neighbor_ids_offset;
    args.order_positions_offset = order_positions_offset;
    args.order_index[0] = 0;
    args.order_index[1] = 0;
    args.num_detectors = config->num_detectors;
    args.num_errors = config->num_errors;
    args.d2e_offsets_offset = prepared.d2e_offsets_offset;
    args.d2e_error_ids_offset = prepared.d2e_error_ids_offset;
    args.e2d_offsets_offset = prepared.e2d_offsets_offset;
    args.e2d_detector_ids_offset = prepared.e2d_detector_ids_offset;
    args.cost_lut_offsets_offset = prepared.cost_lut_offsets_offset;
    args.cost_lut_values_offset = prepared.cost_lut_values_offset;
    args.num_syndromes = 1;
    for (uint32_t g = 0; g < kAstarActiveGroups; ++g) {
      args.syndrome_words_offset[g] = align8(cursor);
      cursor = args.syndrome_words_offset[g] + residual_stride * 4;
    }
    for (uint32_t g = 0; g < kAstarActiveGroups; ++g) {
      args.output_offset[g] = align8(cursor);
      cursor = args.output_offset[g] + kAstarOutputBytes;
    }
    const size_t first_heap = align8(cursor);
    const size_t remaining = kDpuMramBytes - first_heap;
    size_t visited_capacity = 0;
    if (no_revisit_dets != 0) {
      const char* visited_text = std::getenv("TESSERACT_DPU_VISITED_CAPACITY");
      visited_capacity = visited_text ? std::stoull(visited_text) : kAstarDefaultVisitedCapacity;
      if (visited_capacity == 0 || (visited_capacity & (visited_capacity - 1)) != 0) {
        throw std::runtime_error("TESSERACT_DPU_VISITED_CAPACITY must be a power of two");
      }
      while (visited_capacity > 256 &&
             visited_capacity * kAstarVisitedEntryBytes +
                     256 * (kAstarNodeBytes + kAstarArenaBytes) >
                 remaining) {
        visited_capacity >>= 1;
      }
    }
    const size_t visited_bytes = visited_capacity * kAstarVisitedEntryBytes;
    if (visited_bytes >= remaining) {
      throw std::runtime_error("visited-state table leaves no MRAM for the search state");
    }
    size_t capacity =
        (remaining - visited_bytes) /
        (kAstarActiveGroups * (kAstarNodeBytes + kAstarArenaBytes));
    const char* capacity_text = std::getenv("TESSERACT_DPU_RESIDENT_CAPACITY");
    size_t requested = capacity_text ? std::stoull(capacity_text) : kAstarDefaultCapacity;
    if (requested < capacity) {
      capacity = requested;
    }
    if (capacity < 256) {
      throw std::runtime_error("static DPU model leaves too little MRAM for the search state");
    }
    cursor = first_heap;
    for (uint32_t g = 0; g < kAstarActiveGroups; ++g) {
      args.heap_offset[g] = align8(cursor);
      cursor = args.heap_offset[g] + capacity * kAstarNodeBytes;
      args.arena_offset[g] = align8(cursor);
      cursor = args.arena_offset[g] + capacity * kAstarArenaBytes;
      if (visited_capacity != 0) {
        args.visited_offset[g] = align8(cursor);
        cursor = args.visited_offset[g] + visited_bytes;
      }
    }
    args.heap_capacity = static_cast<uint32_t>(capacity);
    args.arena_capacity = static_cast<uint32_t>(capacity);
    args.detector_penalty_q = config->detector_penalty_q;
    args.det_beam = det_beam;
    args.pqlimit = pqlimit;
    args.no_revisit_dets = no_revisit_dets;
    args.max_expansions = max_expansions;
    args.visited_capacity = static_cast<uint32_t>(visited_capacity);
    args.visited_generation = 0;
    args.kernel = 0;

    check_dpu(dpu_alloc(1, nullptr, &handle->dpu), "dpu_alloc");
    handle->allocated = true;
    check_dpu(dpu_load(handle->dpu, config->dpu_binary, nullptr), "dpu_load");
    check_dpu(dpu_copy_to(handle->dpu, DPU_MRAM_HEAP_POINTER_NAME, 0,
                          handle->static_image.data(), handle->static_image.size()),
              "static model host-to-DPU transfer");
    set_abi_error(error_message, error_message_capacity, "");
    return handle.release();
  } catch (const std::exception& ex) {
    set_abi_error(error_message, error_message_capacity, ex.what());
    return nullptr;
  }
}

extern "C" int tesseract_dpu_resident_decode(
    void* opaque_handle,
    uint32_t order_index,
    uint32_t det_beam,
    const uint32_t* active_detectors,
    size_t num_active_detectors,
    uint16_t* output_errors,
    size_t output_capacity,
    tesseract_dpu_resident_result_t* result,
    char* error_message,
    size_t error_message_capacity) {
  try {
    if (opaque_handle == nullptr || result == nullptr) {
      throw std::runtime_error("resident handle or result pointer is null");
    }
    auto* handle = static_cast<ResidentHandle*>(opaque_handle);
    if (order_index >= handle->num_orders) {
      throw std::runtime_error("detector order index is out of range");
    }
    handle->arguments.order_index[0] = order_index;
    handle->arguments.det_beam = det_beam;
    if (handle->arguments.no_revisit_dets != 0) {
      ++handle->visited_generation;
      if (handle->visited_generation == 0) {
        throw std::runtime_error("resident visited-state generation counter wrapped");
      }
      handle->arguments.visited_generation = handle->visited_generation;
    }
    double phase_start = live_thread_cpu_seconds();
    auto end_phase = [&phase_start](double& accumulator) {
      const double now = live_thread_cpu_seconds();
      accumulator += now - phase_start;
      phase_start = now;
    };

    std::fill(handle->syndrome_words.begin(), handle->syndrome_words.end(), 0u);
    for (size_t i = 0; i < num_active_detectors; ++i) {
      const uint32_t detector = active_detectors[i];
      if (detector >= handle->arguments.num_detectors) {
        throw std::runtime_error("active detector ID is out of range");
      }
      handle->syndrome_words[detector / 32] |= uint32_t{1} << (detector & 31);
    }
    end_phase(handle->cpu_pack);

    check_dpu(dpu_copy_to(handle->dpu, "DPU_INPUT_ARGUMENTS", 0, &handle->arguments,
                          sizeof(handle->arguments)),
              "resident argument host-to-DPU transfer");
    check_dpu(dpu_copy_to(handle->dpu, DPU_MRAM_HEAP_POINTER_NAME,
                          handle->arguments.syndrome_words_offset[0],
                          handle->syndrome_words.data(),
                          handle->syndrome_words.size() * sizeof(uint32_t)),
              "resident syndrome host-to-DPU transfer");
    end_phase(handle->cpu_h2d);

    if (handle->before_dpu_wait != nullptr) {
      handle->before_dpu_wait(handle->dpu_wait_context);
    }
    try {
      check_dpu(dpu_launch(handle->dpu, DPU_SYNCHRONOUS), "resident DPU launch");
    } catch (...) {
      if (handle->after_dpu_wait != nullptr) {
        handle->after_dpu_wait(handle->dpu_wait_context);
      }
      throw;
    }
    if (handle->after_dpu_wait != nullptr) {
      handle->after_dpu_wait(handle->dpu_wait_context);
    }
    end_phase(handle->cpu_launch);

    std::vector<uint8_t> raw(kAstarOutputBytes);
    check_dpu(dpu_prepare_xfer(handle->dpu, raw.data()), "prepare resident DPU-to-host transfer");
    check_dpu(dpu_push_xfer(handle->dpu, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME,
                            handle->arguments.output_offset[0], raw.size(), DPU_XFER_DEFAULT),
              "resident DPU-to-host transfer");
    uint32_t header[12];
    std::memcpy(header, raw.data(), sizeof(header));
    result->status = header[0];
    result->depth = header[1];
    result->expansions = header[2];
    result->candidates = header[3];
    result->heap_peak = header[4];
    result->arena_used = header[5];
    result->pushed = header[6];
    result->heap_size = header[8];
    result->min_num_dets = header[9];
    result->max_num_dets = header[10];
    if (result->status == 0) {
      if (result->depth > output_capacity || result->depth > kAstarMaxDepth) {
        result->status = kAstarStatusFallback;
      } else {
        std::memcpy(output_errors, raw.data() + 48, result->depth * sizeof(uint16_t));
      }
    }
    if (header[7] != 0 && std::getenv("TESSERACT_DPU_LIVE_STATS") != nullptr) {
      uint16_t debug[8];
      std::memcpy(debug, raw.data() + 48 + (kAstarMaxDepth - 8) * 2, sizeof(debug));
      std::cerr << "Tesseract DPU resident VERIFY mismatch: shot=" << handle->shots
                << " expansion=" << debug[0] << " j=" << debug[1] << " error=" << debug[2]
                << " incremental=" << (debug[3] | (uint32_t{debug[4]} << 16))
                << " reference=" << (debug[5] | (uint32_t{debug[6]} << 16))
                << " min_detector=" << debug[7] << std::endl;
    }
    ++handle->shots;
    if (result->status == kAstarStatusFallback) {
      ++handle->fallbacks;
    }
    end_phase(handle->cpu_d2h);
    set_abi_error(error_message, error_message_capacity, "");
    return 0;
  } catch (const std::exception& ex) {
    set_abi_error(error_message, error_message_capacity, ex.what());
    return 1;
  }
}

extern "C" int tesseract_dpu_resident_export(
    void* opaque_handle,
    uint32_t* heap_nodes,
    size_t heap_capacity_entries,
    uint32_t* arena,
    size_t arena_capacity_entries,
    char* error_message,
    size_t error_message_capacity) {
  try {
    if (opaque_handle == nullptr) {
      throw std::runtime_error("resident handle is null");
    }
    auto* handle = static_cast<ResidentHandle*>(opaque_handle);
    if (heap_capacity_entries > handle->arguments.heap_capacity ||
        arena_capacity_entries > handle->arguments.arena_capacity) {
      throw std::runtime_error("export request exceeds the DPU search capacity");
    }
    // Both regions are read with 8-byte granularity; entry sizes are multiples of 8.
    if (heap_capacity_entries != 0) {
      check_dpu(dpu_prepare_xfer(handle->dpu, heap_nodes), "prepare heap export");
      check_dpu(dpu_push_xfer(handle->dpu, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME,
                              handle->arguments.heap_offset[0],
                              heap_capacity_entries * kAstarNodeBytes, DPU_XFER_DEFAULT),
                "heap export");
    }
    if (arena_capacity_entries != 0) {
      check_dpu(dpu_prepare_xfer(handle->dpu, arena), "prepare arena export");
      check_dpu(dpu_push_xfer(handle->dpu, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME,
                              handle->arguments.arena_offset[0],
                              arena_capacity_entries * kAstarArenaBytes, DPU_XFER_DEFAULT),
                "arena export");
    }
    set_abi_error(error_message, error_message_capacity, "");
    return 0;
  } catch (const std::exception& ex) {
    set_abi_error(error_message, error_message_capacity, ex.what());
    return 1;
  }
}

extern "C" void tesseract_dpu_resident_destroy(void* opaque_handle) {
  delete static_cast<ResidentHandle*>(opaque_handle);
}

#ifndef TESSERACT_DPU_LIVE_LIBRARY
int main(int argc, char** argv) {
  try {
    if (argc < 4 || argc > 6) {
      std::cerr << "usage: " << argv[0]
                << " TRACE_JSON DPU_BINARY RESULT_CSV [BATCH_SIZE] [REPETITIONS]\n";
      return 2;
    }
    size_t batch_size = argc >= 5 ? std::stoul(argv[4]) : 64;
    size_t repetitions = argc >= 6 ? std::stoul(argv[5]) : 1;
    if (batch_size == 0 || repetitions == 0) {
      throw std::runtime_error("batch size and repetitions must be positive");
    }
    PreparedTrace prepared = prepare_static(load_trace(argv[1]));
    const size_t candidates = prepared.trace.candidate_offsets.size() - 1;
    RunResult result = run_pipeline(prepared, argv[2], batch_size, repetitions);
    const char* blocked_layout = prepared.trace.candidate_blocked_words.empty()
                                     ? "none"
                                     : (prepared.sparse_blocked ? "sparse_wram" : "dense_mram");
    write_csv(argv[3], argv[1], blocked_layout, candidates, batch_size, repetitions, result);
    std::cout << "async DPU double-buffer passed: " << candidates
              << " candidates, " << result.timings.batches << " batches, "
              << candidates * repetitions / result.end_to_end_seconds
              << " candidates/s\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "async DPU host failed: " << ex.what() << '\n';
    return 1;
  }
}
#endif
