// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tesseract.h"

#define TESSERACT_DPU_WEAK_IMPORT
#include "tesseract_dpu_live.h"
#undef TESSERACT_DPU_WEAK_IMPORT

#include <algorithm>
#include <array>
#include <boost/functional/hash.hpp>  // For boost::hash_range
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>  // For std::hash (though not strictly necessary here, but good practice)
#include <iostream>
#include <limits>
#include <mutex>
#include <nlohmann/json.hpp>
#include <numeric>
#include <string>
#include <utility>

#if defined(__linux__)
#include <time.h>
#endif

namespace {

// CPU time consumed by the calling thread. Unlike wall time it excludes blocked
// waits and is unaffected by other processes or clock adjustments.
double thread_cpu_seconds() {
#if defined(__linux__)
  timespec ts{};
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
  return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
#else
  return 0.0;
#endif
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
  os << "[";
  bool is_first = true;
  for (auto& x : vec) {
    if (!is_first) {
      os << ", ";
    }
    is_first = false;
    os << x;
  }
  os << "]";
  return os;
}

int suggest_sparsify_reactivate_limit_capped(size_t num_detectors, int sparsify_base_degree,
                                             int max_limit) {
  if (sparsify_base_degree < 0) {
    throw std::invalid_argument("sparsify_base_degree must be >= 0.");
  }
  if (num_detectors == 0 || max_limit <= 0) {
    return 0;
  }
  double exponent = static_cast<double>(sparsify_base_degree) - 2.0;
  double max_result = static_cast<double>(max_limit);
  double log_result =
      exponent * std::log(4.5) - std::log(3.0) + std::log(static_cast<double>(num_detectors));
  if (!std::isfinite(log_result) || log_result >= std::log(max_result)) {
    return max_limit;
  }
  double result = std::exp(log_result);
  if (!std::isfinite(result)) {
    return max_limit;
  }
  double rounded = std::round(result);
  if (rounded >= max_result) {
    return max_limit;
  }
  return static_cast<int>(rounded);
}

void validate_detector_orders(const std::vector<std::vector<size_t>>& detector_orders,
                              size_t num_detectors) {
  for (size_t order_index = 0; order_index < detector_orders.size(); ++order_index) {
    const std::vector<size_t>& detector_at_position = detector_orders[order_index];
    if (detector_at_position.size() != num_detectors) {
      throw std::invalid_argument("Detector order " + std::to_string(order_index) + " has size " +
                                  std::to_string(detector_at_position.size()) +
                                  ", but the detector error model has " +
                                  std::to_string(num_detectors) + " detectors.");
    }

    std::vector<char> seen(num_detectors, false);
    for (size_t position = 0; position < detector_at_position.size(); ++position) {
      const size_t detector = detector_at_position[position];
      if (detector >= num_detectors) {
        throw std::invalid_argument("Detector order " + std::to_string(order_index) +
                                    " contains out-of-range detector ID " +
                                    std::to_string(detector) + " at position " +
                                    std::to_string(position) + ".");
      }
      if (seen[detector]) {
        throw std::invalid_argument("Detector order " + std::to_string(order_index) +
                                    " contains detector ID " + std::to_string(detector) +
                                    " more than once.");
      }
      seen[detector] = true;
    }
  }
}

};  // namespace

namespace std {
template <>
struct hash<boost::dynamic_bitset<>> {
  size_t operator()(const boost::dynamic_bitset<>& bs) const {
    // Delegate to Boost's internal hash_value for dynamic_bitset
    // This is the correct and most efficient way.
    return boost::hash_value(bs);
  }
};
}  // namespace std

namespace tesseract_decoder {

struct DpuLiveState {
  void* handle = nullptr;
  uint32_t q_bits = 16;
  bool print_stats = false;
  uint64_t batches = 0;
  uint64_t candidates = 0;
  double seconds = 0;
  // Time spent blocked in the synchronous DPU launch only; `seconds` minus this
  // is host-side bridge work (validation, packing, H2D, D2H).
  double dpu_wait_seconds = 0;
  // Time spent waiting to re-acquire a CPU permit after the DPU returned.
  double permit_wait_seconds = 0;
  std::vector<uint32_t> batch_sizes;

  ~DpuLiveState();
  std::vector<uint32_t> evaluate(
      const std::vector<uint32_t>& candidate_offsets,
      const std::vector<uint32_t>& candidate_detector_ids,
      const std::vector<uint64_t>& candidate_blocked_error_words);
};

namespace {
class ResidentOracle;
}

struct DpuResidentState {
  void* handle = nullptr;
  bool identity_order = false;
  bool print_stats = false;
  bool resume = true;
  bool no_revisit_dets = false;
  uint32_t q_bits = 16;
  std::unique_ptr<ResidentOracle> oracle;  // TESSERACT_DPU_RESIDENT_ORACLE=1: host port, no DPU
  uint64_t shots = 0;
  uint64_t solved = 0;
  uint64_t low_confidence = 0;
  uint64_t fallbacks = 0;
  uint64_t resumed = 0;
  uint64_t expansions = 0;
  uint64_t candidates = 0;
  double seconds = 0;
  ~DpuResidentState();
};

namespace {

constexpr uint32_t kDpuTraceInfCost = std::numeric_limits<uint32_t>::max();
std::mutex dpu_live_stats_mutex;

uint32_t quantize_dpu_trace_cost(double value, int q_bits, const char* name) {
  const double scaled = std::round(value * static_cast<double>(uint64_t{1} << q_bits));
  if (!std::isfinite(scaled) || scaled < 0) {
    throw std::runtime_error(std::string(name) + " cannot be represented as an unsigned DPU cost.");
  }
  if (scaled >= static_cast<double>(kDpuTraceInfCost)) {
    return kDpuTraceInfCost - 1;
  }
  return static_cast<uint32_t>(scaled);
}

bool dpu_live_requested() {
  const char* value = std::getenv("TESSERACT_DPU_LIVE");
  if (value == nullptr || value[0] == '\0') {
    return false;
  }
  const std::string text(value);
  return text != "0" && text != "false" && text != "FALSE" && text != "off" &&
         text != "OFF";
}

thread_local std::chrono::steady_clock::time_point dpu_live_wait_start;
thread_local double dpu_live_wait_cpu_start = 0;
thread_local double dpu_live_wait_accumulated = 0;
thread_local double dpu_live_wait_cpu_accumulated = 0;
thread_local double dpu_live_permit_accumulated = 0;

void before_dpu_live_wait(void*) {
  dpu_live_wait_start = std::chrono::steady_clock::now();
  dpu_live_wait_cpu_start = thread_cpu_seconds();
  release_cpu_permit_for_dpu_wait();
}

void after_dpu_live_wait(void*) {
  const auto dpu_done = std::chrono::steady_clock::now();
  dpu_live_wait_accumulated +=
      std::chrono::duration<double>(dpu_done - dpu_live_wait_start).count();
  dpu_live_wait_cpu_accumulated += thread_cpu_seconds() - dpu_live_wait_cpu_start;
  acquire_cpu_permit_after_dpu_wait();
  dpu_live_permit_accumulated +=
      std::chrono::duration<double>(std::chrono::steady_clock::now() - dpu_done).count();
}

// Optional per-decode search statistics, one CSV line per decode request,
// appended to TESSERACT_SEARCH_STATS_PATH. With --threads 1 the line order is
// the shot order.
class SearchStatsScope {
 public:
  SearchStatsScope() {
    const char* path = std::getenv("TESSERACT_SEARCH_STATS_PATH");
    if (path == nullptr || path[0] == '\0') {
      return;
    }
    path_ = path;
    wall_start_ = std::chrono::steady_clock::now();
    cpu_start_ = thread_cpu_seconds();
    wait_wall_start_ = dpu_live_wait_accumulated;
    wait_cpu_start_ = dpu_live_wait_cpu_accumulated;
    permit_start_ = dpu_live_permit_accumulated;
  }

  bool enabled() const {
    return !path_.empty();
  }

  void count_expansion() {
    ++expansions_;
  }

  void count_candidates(size_t n) {
    candidates_ += n;
  }

  void count_dpu_batch() {
    ++dpu_batches_;
  }

  void add_counts(uint64_t expansions, uint64_t candidates) {
    expansions_ += expansions;
    candidates_ += candidates;
  }

  void set_resident_result(uint32_t status, std::string fallback_reason,
                           uint32_t heap_size, uint32_t arena_used) {
    kind_ = "resident_dpu";
    status_ = static_cast<int64_t>(status);
    fallback_reason_ = std::move(fallback_reason);
    heap_size_ = heap_size;
    arena_used_ = arena_used;
  }

  void set_solution_cost(double solution_cost) {
    solution_cost_ = solution_cost;
  }

  void set_solution_depth(size_t solution_depth) {
    solution_depth_ = solution_depth;
  }

  void set_resident_trial(size_t order_index, size_t detector_beam, bool reverse_order) {
    order_index_ = static_cast<int64_t>(order_index);
    detector_beam_ = static_cast<int64_t>(detector_beam);
    reverse_order_ = reverse_order ? 1 : 0;
  }

  ~SearchStatsScope() {
    if (!enabled()) {
      return;
    }
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start_).count();
    const double cpu = thread_cpu_seconds() - cpu_start_;
    const double wait_wall = dpu_live_wait_accumulated - wait_wall_start_;
    const double wait_cpu = dpu_live_wait_cpu_accumulated - wait_cpu_start_;
    const double permit = dpu_live_permit_accumulated - permit_start_;
    static std::mutex mutex;
    static std::ofstream out;
    std::lock_guard<std::mutex> lock(mutex);
    if (!out.is_open()) {
      const bool existed = std::ifstream(path_).good();
      out.open(path_, std::ios::out | std::ios::app);
      if (!existed) {
        out << "wall_seconds,cpu_seconds,dpu_wait_wall_seconds,dpu_wait_cpu_seconds,"
               "permit_wait_wall_seconds,expansions,candidates,dpu_batches,kind,status,"
               "fallback_reason,heap_size,arena_used,export_bytes,solution_cost,solution_depth,"
               "order_index,detector_beam,reverse_order\n";
      }
    }
    out << wall << ',' << cpu << ',' << wait_wall << ',' << wait_cpu << ',' << permit << ','
        << expansions_ << ',' << candidates_ << ',' << dpu_batches_ << ',' << kind_ << ','
        << status_ << ',' << fallback_reason_ << ',' << heap_size_ << ',' << arena_used_ << ','
        << (heap_size_ * uint64_t{16} + arena_used_ * uint64_t{8}) << ',';
    if (std::isnan(solution_cost_)) {
      out << "nan";
    } else {
      out.precision(17);
      out << solution_cost_;
    }
    out << ',' << solution_depth_ << ',' << order_index_ << ',' << detector_beam_ << ','
        << reverse_order_ << '\n';
  }

 private:
  std::string path_;
  std::chrono::steady_clock::time_point wall_start_;
  double cpu_start_ = 0;
  double wait_wall_start_ = 0;
  double wait_cpu_start_ = 0;
  double permit_start_ = 0;
  uint64_t expansions_ = 0;
  uint64_t candidates_ = 0;
  uint64_t dpu_batches_ = 0;
  std::string kind_ = "cpu_search";
  int64_t status_ = -1;
  std::string fallback_reason_ = "none";
  uint32_t heap_size_ = 0;
  uint32_t arena_used_ = 0;
  double solution_cost_ = std::numeric_limits<double>::quiet_NaN();
  int64_t solution_depth_ = -1;
  int64_t order_index_ = -1;
  int64_t detector_beam_ = -1;
  int64_t reverse_order_ = -1;
};

bool dpu_env_enabled(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return false;
  }
  const std::string text(value);
  return text != "0" && text != "false" && text != "FALSE" && text != "off" &&
         text != "OFF";
}

// Error-to-detector CSR and Q-format costs shared by the live and resident
// DPU bridges. The vectors must outlive `config`.
struct DpuModelBuffers {
  std::vector<uint32_t> offsets{0};
  std::vector<uint32_t> detector_ids;
  std::vector<uint32_t> costs;
  tesseract_dpu_live_config_t config{};

  DpuModelBuffers(const TesseractDecoder& decoder, int q_bits, const char* binary) {
    offsets.reserve(decoder.num_errors + 1);
    costs.reserve(decoder.num_errors);
    for (size_t error = 0; error < decoder.num_errors; ++error) {
      for (int detector : decoder.edets[error]) {
        detector_ids.push_back(static_cast<uint32_t>(detector));
      }
      if (detector_ids.size() > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("Tesseract error-to-detector CSR is too large for the DPU.");
      }
      offsets.push_back(static_cast<uint32_t>(detector_ids.size()));
      costs.push_back(quantize_dpu_trace_cost(
          decoder.error_costs[error].likelihood_cost, q_bits, "error likelihood cost"));
    }
    config.q_bits = static_cast<uint32_t>(q_bits);
    config.detector_penalty_q =
        quantize_dpu_trace_cost(decoder.config.det_penalty, q_bits, "detector penalty");
    config.num_detectors = static_cast<uint32_t>(decoder.num_detectors);
    config.num_errors = static_cast<uint32_t>(decoder.num_errors);
    config.error_detector_offsets = offsets.data();
    config.num_error_detector_offsets = offsets.size();
    config.error_detector_ids = detector_ids.data();
    config.num_error_detector_ids = detector_ids.size();
    config.error_costs_q = costs.data();
    config.num_error_costs = costs.size();
    config.dpu_binary = binary;
    config.before_dpu_wait = before_dpu_live_wait;
    config.after_dpu_wait = after_dpu_live_wait;
    config.dpu_wait_context = nullptr;
    // Detector orders as positions, one table per order.
    const size_t num_detectors = decoder.num_detectors;
    positions.assign(decoder.config.det_orders.size() * num_detectors, 0);
    for (size_t o = 0; o < decoder.config.det_orders.size(); ++o) {
      for (size_t pos = 0; pos < num_detectors; ++pos) {
        positions[o * num_detectors + decoder.config.det_orders[o][pos]] =
            static_cast<uint16_t>(pos);
      }
    }
    config.order_positions = positions.data();
    config.num_orders = static_cast<uint32_t>(decoder.config.det_orders.size());
  }
  std::vector<uint16_t> positions;
};


// Host port of the DPU-resident A* kernel (integer-exact, same order, same
// tie handling). It reproduces the DPU's result without any UPMEM runtime so
// large streams can be checked against CPU-only Tesseract.
class ResidentOracle {
 public:
  struct Result {
    uint32_t status = 0;  // 0 solved, 1 low confidence, 2 fallback
    std::vector<uint16_t> chain;
    uint32_t expansions = 0;
    uint32_t candidates = 0;
    uint32_t min_num_dets = 0;
    uint32_t max_num_dets = 0;
    std::string fallback_reason = "none";
    std::vector<uint32_t> heap_words;   // 4 per node
    std::vector<uint32_t> arena_words;  // 2 per link
  };

  ResidentOracle(const TesseractDecoder& decoder, int q_bits, uint32_t pqlimit,
                 bool no_revisit_dets, uint32_t capacity)
      : decoder_(decoder), pqlimit_(pqlimit), no_revisit_dets_(no_revisit_dets),
        capacity_(capacity) {
    const size_t num_errors = decoder.num_errors;
    cost_q_.resize(num_errors);
    lut_offsets_.assign(num_errors + 1, 0);
    for (size_t e = 0; e < num_errors; ++e) {
      cost_q_[e] = quantize_dpu_trace_cost(decoder.errors[e].likelihood_cost, q_bits,
                                           "error likelihood cost");
      lut_values_.push_back(kInf);
      const uint64_t degree = decoder.edets[e].size();
      for (uint64_t k = 1; k <= degree; ++k) {
        lut_values_.push_back(static_cast<uint32_t>(
            std::min<uint64_t>((static_cast<uint64_t>(cost_q_[e]) + k / 2) / k, kInf - 1)));
      }
      lut_offsets_[e + 1] = static_cast<uint32_t>(lut_values_.size());
    }
    penalty_ = quantize_dpu_trace_cost(decoder.config.det_penalty, q_bits, "detector penalty");
    residual_words_ = (decoder.num_detectors + 31) / 32;
    position_.assign(num_errors, kNoPosition);
    blocked_.assign((num_errors + 31) / 32, 0);
    parent_cost_.assign(decoder.num_detectors, 0);
  }

  Result run(const std::vector<uint64_t>& detections, size_t order_index,
             uint32_t det_beam) {
    Result result;
    const std::vector<size_t>& order = decoder_.config.det_orders.at(order_index);
    std::vector<uint32_t> position(decoder_.num_detectors);
    for (size_t pos = 0; pos < order.size(); ++pos) {
      position[order[pos]] = static_cast<uint32_t>(pos);
    }
    std::vector<uint32_t> syndrome(residual_words_, 0);
    uint32_t num_dets = 0;
    for (uint64_t d : detections) {
      syndrome[d / 32] |= uint32_t{1} << (d % 32);
    }
    for (uint32_t word : syndrome) {
      num_dets += static_cast<uint32_t>(__builtin_popcount(word));
    }
    for (uint64_t d : detections) {
      if (decoder_.d2e[d].empty()) {
        result.status = 1;
        return result;
      }
    }
    heap_.clear();
    arena_.clear();
    uint32_t min_num_dets = num_dets;
    uint32_t max_num_dets = det_beam == kInf ? kInf : num_dets + det_beam;
    heap_push({0, 0, kInf, static_cast<uint16_t>(num_dets), 0});
    uint64_t num_pushed = 1;
    std::vector<std::unordered_set<std::string>> visited(decoder_.num_detectors + 1);
    auto residual_key = [&](const std::vector<uint32_t>& words) {
      return std::string(reinterpret_cast<const char*>(words.data()),
                         words.size() * sizeof(uint32_t));
    };

    std::vector<uint32_t> residual(residual_words_);
    std::vector<uint32_t> candidate(residual_words_);
    auto finish = [&](uint32_t status) {
      result.status = status;
      result.min_num_dets = min_num_dets;
      result.max_num_dets = max_num_dets;
      result.heap_words.reserve(heap_.size() * 4);
      for (const Node& n : heap_) {
        result.heap_words.push_back(n.cost);
        result.heap_words.push_back(n.path);
        result.heap_words.push_back(n.chain);
        result.heap_words.push_back(static_cast<uint32_t>(n.num_dets) |
                                    (static_cast<uint32_t>(n.depth) << 16));
      }
      result.arena_words.reserve(arena_.size() * 2);
      for (const Link& l : arena_) {
        result.arena_words.push_back(l.parent);
        result.arena_words.push_back(static_cast<uint32_t>(l.error) |
                                     (static_cast<uint32_t>(l.min_detector) << 16));
      }
      return result;
    };

    while (true) {
      Node node;
      while (true) {
        if (heap_.empty()) {
          return finish(1);
        }
        node = heap_pop();
        if (node.num_dets <= max_num_dets) {
          break;
        }
      }
      residual = syndrome;
      std::fill(blocked_.begin(), blocked_.end(), 0);
      uint32_t depth = 0;
      for (uint32_t walker = node.chain; walker != kInf; walker = arena_[walker].parent) {
        const Link& link = arena_[walker];
        for (int other : decoder_.d2e[link.min_detector]) {
          blocked_[other / 32] |= uint32_t{1} << (other % 32);
          if (static_cast<uint32_t>(other) == link.error) {
            break;
          }
        }
        flip(residual, link.error);
        ++depth;
      }
      if (node.num_dets == 0) {
        if (depth > 256) {
          // Preserve the solved node in the exported queue.  Otherwise a CPU
          // resume would lose the solution that triggered the depth limit.
          heap_push(node);
          result.fallback_reason = "solution_depth";
          return finish(2);
        }
        result.chain.assign(depth, 0);
        uint32_t index = depth;
        for (uint32_t walker = node.chain; walker != kInf; walker = arena_[walker].parent) {
          result.chain[--index] = arena_[walker].error;
        }
        return finish(0);
      }
      if (no_revisit_dets_ &&
          !visited[node.num_dets].insert(residual_key(residual)).second) {
        continue;
      }
      if (node.num_dets < min_num_dets) {
        min_num_dets = node.num_dets;
        if (det_beam != kInf && min_num_dets + det_beam < max_num_dets) {
          max_num_dets = min_num_dets + det_beam;
        }
      }
      uint32_t min_detector = 0;
      uint32_t min_position = kInf;
      for (uint32_t word = 0; word < residual_words_; ++word) {
        uint32_t bits = residual[word];
        while (bits != 0) {
          const uint32_t d = word * 32 + static_cast<uint32_t>(__builtin_ctz(bits));
          bits &= bits - 1;
          if (position[d] < min_position) {
            min_position = position[d];
            min_detector = d;
          }
        }
      }
      const std::vector<int>& siblings = decoder_.d2e[min_detector];
      const uint32_t count = static_cast<uint32_t>(siblings.size());
      if (count > kMaxSiblings || heap_.size() + count > capacity_ ||
          arena_.size() + count > capacity_) {
        if (count > kMaxSiblings) {
          result.fallback_reason = "sibling_limit";
        } else if (heap_.size() + count > capacity_) {
          result.fallback_reason = "heap_capacity";
        } else {
          result.fallback_reason = "arena_capacity";
        }
        heap_push(node);
        return finish(2);
      }
      ++result.expansions;
      uint32_t parent_total = 0;
      bool parent_inf = false;
      for (uint32_t word = 0; word < residual_words_; ++word) {
        uint32_t bits = residual[word];
        while (bits != 0) {
          const uint32_t d = word * 32 + static_cast<uint32_t>(__builtin_ctz(bits));
          bits &= bits - 1;
          const uint32_t cost = evaluate_detector(residual, d, kNoPrefix);
          parent_cost_[d] = cost;
          if (cost == kInf) {
            parent_inf = true;
          } else {
            parent_total += cost;
          }
        }
      }
      for (uint32_t j = 0; j < count; ++j) {
        position_[siblings[j]] = static_cast<uint16_t>(j);
      }
      std::vector<uint32_t> costs(count, kInf);
      std::vector<uint16_t> dets(count, 0);
      std::vector<uint8_t> evaluated(count, 0);
      for (uint32_t j = 0; j < count; ++j) {
        const uint32_t e = static_cast<uint32_t>(siblings[j]);
        if ((blocked_[e / 32] >> (e % 32)) & 1) {
          continue;
        }
        candidate = residual;
        flip(candidate, e);
        uint32_t total = 0;
        for (uint32_t word : candidate) {
          total += static_cast<uint32_t>(__builtin_popcount(word));
        }
        if (total > max_num_dets) {
          continue;
        }
        if (no_revisit_dets_ &&
            visited[total].find(residual_key(candidate)) != visited[total].end()) {
          continue;
        }
        dets[j] = static_cast<uint16_t>(total);
        evaluated[j] = 1;
        if (parent_inf) {
          continue;
        }
        uint32_t h = parent_total;
        bool infinite = false;
        for (int d : decoder_.edets[e]) {
          if (infinite) break;
          if ((residual[d / 32] >> (d % 32)) & 1) {
            h -= parent_cost_[d];
          } else {
            const uint32_t cost = evaluate_detector(candidate, d, j);
            if (cost == kInf || cost >= kInf - h) {
              infinite = true;
            } else {
              h += cost;
            }
          }
        }
        for (int d : decoder_.eneighbors[e]) {
          if (infinite) break;
          if (((residual[d / 32] >> (d % 32)) & 1) == 0) {
            continue;
          }
          h -= parent_cost_[d];
          const uint32_t cost = evaluate_detector(candidate, d, j);
          if (cost == kInf || cost >= kInf - h) {
            infinite = true;
          } else {
            h += cost;
          }
        }
        if (!infinite) {
          costs[j] = h;
        }
      }
      for (uint32_t j = 0; j < count; ++j) {
        const uint32_t e = static_cast<uint32_t>(siblings[j]);
        position_[e] = kNoPosition;
        result.candidates += evaluated[j];
        if (costs[j] == kInf) {
          continue;
        }
        const uint32_t step = lut_values_[lut_offsets_[e] + 1];
        if (step >= kInf - node.path) continue;
        const uint32_t path = node.path + step;
        if (costs[j] >= kInf - path) continue;
        arena_.push_back({node.chain, static_cast<uint16_t>(e), static_cast<uint16_t>(min_detector)});
        heap_push({path + costs[j], path, static_cast<uint32_t>(arena_.size() - 1), dets[j],
                   static_cast<uint16_t>(node.depth + 1)});
        ++num_pushed;
        if (num_pushed > pqlimit_) {
          return finish(1);
        }
      }
    }
  }

 private:
  static constexpr uint32_t kInf = std::numeric_limits<uint32_t>::max();
  static constexpr uint32_t kNoPrefix = std::numeric_limits<uint32_t>::max();
  static constexpr uint16_t kNoPosition = 0xFF;
  static constexpr uint32_t kMaxSiblings = 255;

  struct Node {
    uint32_t cost;
    uint32_t path;
    uint32_t chain;
    uint16_t num_dets;
    uint16_t depth;
  };
  struct Link {
    uint32_t parent;
    uint16_t error;
    uint16_t min_detector;
  };

  static bool less(const Node& a, const Node& b) {
    return a.cost < b.cost || (a.cost == b.cost && a.num_dets > b.num_dets);
  }

  // Binary heap with the kernel's exact sift order so equal keys pop identically.
  void heap_push(Node node) {
    heap_.push_back(Node{});
    size_t index = heap_.size() - 1;
    while (index > 0) {
      const size_t parent = (index - 1) >> 1;
      if (!less(node, heap_[parent])) break;
      heap_[index] = heap_[parent];
      index = parent;
    }
    heap_[index] = node;
  }

  Node heap_pop() {
    Node out = heap_[0];
    const size_t size = heap_.size() - 1;
    if (size == 0) {
      heap_.clear();
      return out;
    }
    Node last = heap_[size];
    heap_.resize(size);
    size_t index = 0;
    while (true) {
      const size_t left = index * 2 + 1;
      if (left >= size) break;
      size_t best = left;
      Node child = heap_[left];
      if (left + 1 < size && less(heap_[left + 1], child)) {
        best = left + 1;
        child = heap_[left + 1];
      }
      if (!less(child, last)) break;
      heap_[index] = child;
      index = best;
    }
    heap_[index] = last;
    return out;
  }

  void flip(std::vector<uint32_t>& residual, uint32_t e) const {
    for (int d : decoder_.edets[e]) {
      residual[d / 32] ^= uint32_t{1} << (d % 32);
    }
  }

  uint32_t evaluate_detector(const std::vector<uint32_t>& residual, uint32_t d,
                             uint32_t sibling_index) const {
    uint32_t minimum = kInf;
    for (int ei : decoder_.d2e[d]) {
      const uint32_t e = static_cast<uint32_t>(ei);
      if ((blocked_[e / 32] >> (e % 32)) & 1) continue;
      if (sibling_index != kNoPrefix && position_[e] <= sibling_index) continue;
      const uint32_t degree = static_cast<uint32_t>(decoder_.edets[e].size());
      const uint32_t lut_begin = lut_offsets_[e];
      if (minimum != kInf && lut_values_[lut_begin + degree] >= minimum) break;
      uint32_t overlap = 0;
      for (int od : decoder_.edets[e]) {
        overlap += (residual[od / 32] >> (od % 32)) & 1;
      }
      if (overlap == 0) continue;
      const uint32_t cost = lut_values_[lut_begin + overlap];
      if (cost < minimum) minimum = cost;
    }
    if (minimum == kInf || minimum >= kInf - penalty_) return kInf;
    return minimum + penalty_;
  }

  const TesseractDecoder& decoder_;
  uint32_t pqlimit_;
  bool no_revisit_dets_;
  uint32_t capacity_;
  uint32_t penalty_ = 0;
  uint32_t residual_words_ = 0;
  std::vector<uint32_t> cost_q_;
  std::vector<uint32_t> lut_offsets_;
  std::vector<uint32_t> lut_values_;
  std::vector<uint16_t> position_;
  std::vector<uint32_t> blocked_;
  std::vector<uint32_t> parent_cost_;
  std::vector<Node> heap_;
  std::vector<Link> arena_;
};

std::shared_ptr<DpuResidentState> create_dpu_resident_state(const TesseractDecoder& decoder) {
  if (!dpu_env_enabled("TESSERACT_DPU_RESIDENT")) {
    return nullptr;
  }
#if defined(__GNUC__) && !defined(_WIN32)
  const bool oracle_mode = dpu_env_enabled("TESSERACT_DPU_RESIDENT_ORACLE");
  if (!oracle_mode &&
      (tesseract_dpu_resident_create == nullptr || tesseract_dpu_resident_decode == nullptr ||
       tesseract_dpu_resident_destroy == nullptr)) {
    throw std::runtime_error(
        "TESSERACT_DPU_RESIDENT is enabled, but libtesseract_dpu_live.so is not loaded.");
  }
  const char* binary = std::getenv("TESSERACT_DPU_BINARY");
  if (!oracle_mode && (binary == nullptr || binary[0] == '\0')) {
    throw std::runtime_error(
        "TESSERACT_DPU_BINARY must name the compiled resident DPU program.");
  }
  const char* q_bits_text = std::getenv("TESSERACT_DPU_LIVE_Q_BITS");
  const int q_bits = q_bits_text == nullptr ? 16 : std::stoi(q_bits_text);
  if (q_bits < 0 || q_bits > 24) {
    throw std::runtime_error("TESSERACT_DPU_LIVE_Q_BITS must be in [0,24].");
  }
  if (decoder.config.sparsify_errors || decoder.num_detectors > 65535) {
    throw std::runtime_error(
        "TESSERACT_DPU_RESIDENT does not support sparsification or more than 65535 detectors.");
  }
  DpuModelBuffers model(decoder, q_bits, binary);
  const uint32_t det_beam = decoder.config.det_beam >= static_cast<int>(INF_DET_BEAM)
                                ? std::numeric_limits<uint32_t>::max()
                                : static_cast<uint32_t>(decoder.config.det_beam);
  const uint32_t pqlimit = decoder.config.pqlimit >= std::numeric_limits<uint32_t>::max()
                               ? std::numeric_limits<uint32_t>::max()
                               : static_cast<uint32_t>(decoder.config.pqlimit);
  const char* max_expansions_text = std::getenv("TESSERACT_DPU_RESIDENT_MAX_EXPANSIONS");
  const uint32_t max_expansions =
      max_expansions_text == nullptr ? 0 : static_cast<uint32_t>(std::stoul(max_expansions_text));

  auto state = std::make_shared<DpuResidentState>();
  if (oracle_mode) {
    const char* capacity_text = std::getenv("TESSERACT_DPU_RESIDENT_CAPACITY");
    const uint32_t capacity =
        capacity_text == nullptr ? 1000000u : static_cast<uint32_t>(std::stoul(capacity_text));
    state->oracle = std::make_unique<ResidentOracle>(
        decoder, q_bits, pqlimit, decoder.config.no_revisit_dets, capacity);
  } else {
    std::array<char, 512> error{};
    void* handle = tesseract_dpu_resident_create(
        &model.config, det_beam, pqlimit, decoder.config.no_revisit_dets ? 1u : 0u,
        max_expansions, error.data(), error.size());
    if (handle == nullptr) {
      throw std::runtime_error(std::string("failed to initialize DPU resident search: ") +
                               error.data());
    }
    state->handle = handle;
  }
  state->identity_order = decoder.config.det_orders.size() == 1;
  state->print_stats = std::getenv("TESSERACT_DPU_LIVE_STATS") != nullptr;
  state->q_bits = static_cast<uint32_t>(q_bits);
  state->no_revisit_dets = decoder.config.no_revisit_dets;
  state->resume = !decoder.config.no_revisit_dets &&
                  !dpu_env_enabled("TESSERACT_DPU_RESIDENT_NO_RESUME");
  return state;
#else
  throw std::runtime_error("TESSERACT_DPU_RESIDENT requires ELF weak-symbol support.");
#endif
}

std::shared_ptr<DpuLiveState> create_dpu_live_state(const TesseractDecoder& decoder) {
  if (!dpu_live_requested()) {
    return nullptr;
  }
#if defined(__GNUC__) && !defined(_WIN32)
  if (tesseract_dpu_live_create == nullptr || tesseract_dpu_live_evaluate == nullptr ||
      tesseract_dpu_live_destroy == nullptr) {
    throw std::runtime_error(
        "TESSERACT_DPU_LIVE is enabled, but libtesseract_dpu_live.so is not loaded. "
        "Run through scripts/run_live_dem_comparison.sh or set LD_PRELOAD.");
  }

  const char* binary = std::getenv("TESSERACT_DPU_BINARY");
  if (binary == nullptr || binary[0] == '\0') {
    throw std::runtime_error(
        "TESSERACT_DPU_BINARY must name the compiled DPU program in live mode.");
  }
  const char* q_bits_text = std::getenv("TESSERACT_DPU_LIVE_Q_BITS");
  const int q_bits = q_bits_text == nullptr ? 16 : std::stoi(q_bits_text);
  if (q_bits < 0 || q_bits > 24) {
    throw std::runtime_error("TESSERACT_DPU_LIVE_Q_BITS must be in [0,24].");
  }
  if (decoder.num_detectors > std::numeric_limits<uint32_t>::max() ||
      decoder.num_errors > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error("Tesseract graph is too large for the DPU live ABI.");
  }

  DpuModelBuffers model(decoder, q_bits, binary);

  std::array<char, 512> error{};
  void* handle = tesseract_dpu_live_create(&model.config, error.data(), error.size());
  if (handle == nullptr) {
    throw std::runtime_error(std::string("failed to initialize DPU live heuristic: ") +
                             error.data());
  }
  auto state = std::make_shared<DpuLiveState>();
  state->handle = handle;
  state->q_bits = static_cast<uint32_t>(q_bits);
  state->print_stats = std::getenv("TESSERACT_DPU_LIVE_STATS") != nullptr;
  if (state->print_stats) {
    state->batch_sizes.reserve(64);
  }
  return state;
#else
  throw std::runtime_error("TESSERACT_DPU_LIVE requires ELF weak-symbol support.");
#endif
}

class DpuLiveBatch {
 public:
  DpuLiveBatch(const std::shared_ptr<DpuLiveState>& state,
               const std::vector<uint64_t>& base_blocked,
               size_t num_errors,
               size_t candidate_capacity)
      : state_(state), base_blocked_(base_blocked), num_errors_(num_errors) {
    candidate_offsets_.reserve(candidate_capacity + 1);
    candidate_offsets_.push_back(0);
    if (state_ == nullptr) {
      return;
    }
    const size_t expected_words = (num_errors + 63) / 64;
    if (base_blocked_.size() != expected_words) {
      throw std::runtime_error("DPU live inactive-error mask has the wrong length.");
    }
    if (candidate_capacity <=
        std::numeric_limits<size_t>::max() / std::max<size_t>(expected_words, 1)) {
      candidate_blocked_words_.reserve(candidate_capacity * expected_words);
    }
  }

  bool enabled() const {
    return state_ != nullptr;
  }

  // Snapshot the expanded node's blocked errors once. Sibling candidates then
  // differ only by the cumulative sibling prefix, which add_candidate appends
  // one bit at a time instead of rescanning every error per candidate.
  void begin_expansion(const std::vector<DetectorCostTuple>& node_cost_tuples) {
    if (!enabled()) {
      return;
    }
    current_blocked_ = base_blocked_;
    for (size_t error = 0; error < num_errors_; ++error) {
      if (node_cost_tuples[error].error_blocked) {
        current_blocked_[error / 64] |= uint64_t{1} << (error & 63);
      }
    }
  }

  void add_candidate(const boost::dynamic_bitset<>& detectors, size_t blocked_error) {
    if (!enabled()) {
      return;
    }
    for (size_t detector = detectors.find_first(); detector != boost::dynamic_bitset<>::npos;
         detector = detectors.find_next(detector)) {
      candidate_detector_ids_.push_back(static_cast<uint32_t>(detector));
    }
    candidate_offsets_.push_back(static_cast<uint32_t>(candidate_detector_ids_.size()));

    current_blocked_[blocked_error / 64] |= uint64_t{1} << (blocked_error & 63);
    candidate_blocked_words_.insert(candidate_blocked_words_.end(), current_blocked_.begin(),
                                    current_blocked_.end());
  }

  std::vector<uint32_t> evaluate() {
    if (!enabled() || candidate_offsets_.size() == 1) {
      return {};
    }
    return state_->evaluate(candidate_offsets_, candidate_detector_ids_,
                            candidate_blocked_words_);
  }

 private:
  std::shared_ptr<DpuLiveState> state_;
  const std::vector<uint64_t>& base_blocked_;
  size_t num_errors_;
  std::vector<uint32_t> candidate_offsets_;
  std::vector<uint32_t> candidate_detector_ids_;
  std::vector<uint64_t> candidate_blocked_words_;
  std::vector<uint64_t> current_blocked_;
};

std::vector<uint64_t> make_dpu_inactive_error_words(
    const std::shared_ptr<DpuLiveState>& state,
    const std::vector<std::vector<int>>& active_d2e,
    size_t num_errors) {
  if (state == nullptr) {
    return {};
  }
  std::vector<uint8_t> active(num_errors, 0);
  for (const auto& incident_errors : active_d2e) {
    for (int error : incident_errors) {
      active.at(static_cast<size_t>(error)) = 1;
    }
  }
  std::vector<uint64_t> blocked((num_errors + 63) / 64, 0);
  for (size_t error = 0; error < num_errors; ++error) {
    if (!active[error]) {
      blocked[error / 64] |= uint64_t{1} << (error & 63);
    }
  }
  return blocked;
}

class DpuTraceWriter {
 public:
  DpuTraceWriter(const TesseractDecoder& decoder,
                 const std::vector<std::vector<int>>& active_d2e)
      : decoder_(decoder),
        active_d2e_(active_d2e),
        active_errors_(decoder.num_errors, false),
        syndrome_only_(dpu_env_enabled("TESSERACT_DPU_TRACE_SYNDROME_ONLY")) {
    const char* path = std::getenv("TESSERACT_DPU_TRACE_PATH");
    if (path == nullptr || path[0] == '\0') {
      return;
    }
    path_ = path;

    const char* q_bits_text = std::getenv("TESSERACT_DPU_TRACE_Q_BITS");
    q_bits_ = q_bits_text == nullptr ? 16 : std::stoi(q_bits_text);
    if (q_bits_ < 0 || q_bits_ > 24) {
      throw std::runtime_error("TESSERACT_DPU_TRACE_Q_BITS must be in [0,24].");
    }

    for (const auto& error_ids : active_d2e) {
      for (int error_id : error_ids) {
        active_errors_.at(static_cast<size_t>(error_id)) = true;
      }
    }

    std::vector<uint32_t> error_detector_offsets{0};
    std::vector<uint32_t> error_detector_ids;
    error_cost_q_.reserve(decoder.num_errors);
    for (size_t error_id = 0; error_id < decoder.num_errors; ++error_id) {
      for (int detector_id : decoder.edets[error_id]) {
        error_detector_ids.push_back(static_cast<uint32_t>(detector_id));
      }
      error_detector_offsets.push_back(static_cast<uint32_t>(error_detector_ids.size()));
      error_cost_q_.push_back(quantize_dpu_trace_cost(
          decoder.error_costs[error_id].likelihood_cost, q_bits_, "error likelihood cost"));
    }

    trace_ = {
        {"format", "tesseract-dpu-trace-v1"},
        {"q_bits", q_bits_},
        {"detector_penalty_q",
         quantize_dpu_trace_cost(decoder.config.det_penalty, q_bits_, "detector penalty")},
        {"num_detectors", decoder.num_detectors},
        {"error_detector_offsets", error_detector_offsets},
        {"error_detector_ids", error_detector_ids},
        {"error_cost_q", error_cost_q_},
        {"candidate_detector_offsets", std::vector<uint32_t>{0}},
        {"candidate_detector_ids", std::vector<uint32_t>{}},
        {"candidate_blocked_error_words", std::vector<uint64_t>{}},
        {"candidate_cost_q", std::vector<uint32_t>{}},
    };
  }

  ~DpuTraceWriter() {
    if (!enabled()) {
      return;
    }
    try {
      std::ofstream out(path_, std::ios::out | std::ios::trunc);
      if (!out) {
        std::cerr << "Failed to open TESSERACT DPU trace: " << path_ << std::endl;
        return;
      }
      out << trace_.dump(2) << '\n';
    } catch (const std::exception& ex) {
      std::cerr << "Failed to write TESSERACT DPU trace: " << ex.what() << std::endl;
    }
  }

  void add_candidate(const boost::dynamic_bitset<>& detectors,
                     const std::vector<DetectorCostTuple>& detector_cost_tuples) {
    if (!enabled()) {
      return;
    }
    // The resident A* kernel reconstructs the full search from candidate 0,
    // which is the input syndrome.  Avoid serializing every later CPU search
    // candidate when a trace is intended only for cycle-level A* replay.
    if (syndrome_only_ && trace_["candidate_detector_offsets"].size() > 1) {
      return;
    }
    auto& detector_ids = trace_["candidate_detector_ids"];
    for (size_t detector_id = 0; detector_id < detectors.size(); ++detector_id) {
      if (detectors[detector_id]) {
        detector_ids.push_back(static_cast<uint32_t>(detector_id));
      }
    }
    trace_["candidate_detector_offsets"].push_back(detector_ids.size());

    const size_t words_per_candidate = (decoder_.num_errors + 63) / 64;
    std::vector<uint64_t> blocked_words(words_per_candidate, 0);
    for (size_t error_id = 0; error_id < decoder_.num_errors; ++error_id) {
      if (detector_cost_tuples[error_id].error_blocked || !active_errors_[error_id]) {
        blocked_words[error_id / 64] |= uint64_t{1} << (error_id & 63);
      }
    }
    for (uint64_t word : blocked_words) {
      trace_["candidate_blocked_error_words"].push_back(word);
    }

    uint64_t candidate_cost = 0;
    for (size_t detector_id = 0; detector_id < detectors.size(); ++detector_id) {
      if (!detectors[detector_id]) {
        continue;
      }
      uint32_t minimum = kDpuTraceInfCost;
      for (int error_id : active_d2e_[detector_id]) {
        const auto& tuple = detector_cost_tuples[error_id];
        if (tuple.error_blocked || tuple.detectors_count == 0) {
          continue;
        }
        const uint64_t overlap = tuple.detectors_count;
        const uint32_t cost = static_cast<uint32_t>(
            (static_cast<uint64_t>(error_cost_q_[error_id]) + overlap / 2) / overlap);
        minimum = std::min(minimum, cost);
      }
      if (minimum != kDpuTraceInfCost) {
        const uint64_t with_penalty = static_cast<uint64_t>(minimum) +
                                      trace_["detector_penalty_q"].get<uint32_t>();
        minimum = with_penalty >= kDpuTraceInfCost
                      ? kDpuTraceInfCost
                      : static_cast<uint32_t>(with_penalty);
      }
      if (minimum == kDpuTraceInfCost ||
          candidate_cost + minimum >= kDpuTraceInfCost) {
        candidate_cost = kDpuTraceInfCost;
        break;
      }
      candidate_cost += minimum;
    }
    trace_["candidate_cost_q"].push_back(static_cast<uint32_t>(candidate_cost));
  }

 private:
  bool enabled() const {
    return !path_.empty();
  }

  const TesseractDecoder& decoder_;
  const std::vector<std::vector<int>>& active_d2e_;
  std::vector<bool> active_errors_;
  std::vector<uint32_t> error_cost_q_;
  std::string path_;
  int q_bits_ = 16;
  bool syndrome_only_ = false;
  nlohmann::json trace_;
};

}  // namespace

DpuResidentState::~DpuResidentState() {
#if defined(__GNUC__) && !defined(_WIN32)
  if (handle != nullptr && tesseract_dpu_resident_destroy != nullptr) {
    tesseract_dpu_resident_destroy(handle);
  }
#endif
  if (print_stats) {
    std::lock_guard<std::mutex> lock(dpu_live_stats_mutex);
    std::cerr << "Tesseract DPU resident search: shots=" << shots << " solved=" << solved
              << " low_confidence=" << low_confidence << " fallbacks=" << fallbacks
              << " resumed=" << resumed
              << " expansions=" << expansions << " candidates=" << candidates
              << " seconds=" << seconds << std::endl;
  }
}

DpuLiveState::~DpuLiveState() {
#if defined(__GNUC__) && !defined(_WIN32)
  if (handle != nullptr && tesseract_dpu_live_destroy != nullptr) {
    tesseract_dpu_live_destroy(handle);
  }
#endif
  if (print_stats) {
    const double throughput = seconds == 0 ? 0 : static_cast<double>(candidates) / seconds;
    std::lock_guard<std::mutex> lock(dpu_live_stats_mutex);
    std::cerr << "Tesseract DPU live: batches=" << batches << " candidates=" << candidates
              << " seconds=" << seconds << " dpu_wait_seconds=" << dpu_wait_seconds
              << " permit_wait_seconds=" << permit_wait_seconds
              << " candidates/s=" << throughput << " batch_sizes=";
    for (size_t index = 0; index < batch_sizes.size(); ++index) {
      if (index != 0) {
        std::cerr << ',';
      }
      std::cerr << batch_sizes[index];
    }
    std::cerr << std::endl;
  }
}

std::vector<uint32_t> DpuLiveState::evaluate(
    const std::vector<uint32_t>& candidate_offsets,
    const std::vector<uint32_t>& candidate_detector_ids,
    const std::vector<uint64_t>& candidate_blocked_error_words) {
#if defined(__GNUC__) && !defined(_WIN32)
  if (candidate_offsets.empty()) {
    throw std::runtime_error("DPU live candidate offsets cannot be empty.");
  }
  const size_t candidate_count = candidate_offsets.size() - 1;
  if (candidate_count > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error("DPU live batch has too many candidates.");
  }
  std::vector<uint32_t> output(candidate_count);
  tesseract_dpu_live_batch_t batch{};
  batch.num_candidates = static_cast<uint32_t>(candidate_count);
  batch.candidate_detector_offsets = candidate_offsets.data();
  batch.num_candidate_detector_offsets = candidate_offsets.size();
  batch.candidate_detector_ids = candidate_detector_ids.data();
  batch.num_candidate_detector_ids = candidate_detector_ids.size();
  batch.candidate_blocked_error_words = candidate_blocked_error_words.data();
  batch.num_candidate_blocked_error_words = candidate_blocked_error_words.size();

  std::array<char, 512> error{};
  const double wait_before = dpu_live_wait_accumulated;
  const double permit_before = dpu_live_permit_accumulated;
  const auto start = std::chrono::steady_clock::now();
  const int status = tesseract_dpu_live_evaluate(handle, &batch, output.data(), output.size(),
                                                  error.data(), error.size());
  seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  dpu_wait_seconds += dpu_live_wait_accumulated - wait_before;
  permit_wait_seconds += dpu_live_permit_accumulated - permit_before;
  ++batches;
  candidates += candidate_count;
  if (print_stats) {
    batch_sizes.push_back(static_cast<uint32_t>(candidate_count));
  }
  if (status != 0) {
    throw std::runtime_error(std::string("DPU live heuristic evaluation failed: ") + error.data());
  }
  return output;
#else
  (void)candidate_offsets;
  (void)candidate_detector_ids;
  (void)candidate_blocked_error_words;
  throw std::runtime_error("TESSERACT_DPU_LIVE requires ELF weak-symbol support.");
#endif
}

std::string TesseractConfig::str() {
  auto& config = *this;
  std::stringstream ss;
  ss << "TesseractConfig(";
  ss << "dem=DetectorErrorModel_Object" << ", ";
  ss << "det_beam=" << config.det_beam << ", ";
  ss << "no_revisit_dets=" << config.no_revisit_dets << ", ";

  ss << "verbose=" << config.verbose << ", ";
  ss << "merge_errors=" << config.merge_errors << ", ";
  ss << "pqlimit=" << config.pqlimit << ", ";
  ss << "det_orders=" << config.det_orders << ", ";
  ss << "det_penalty=" << config.det_penalty << ", ";
  ss << "create_visualization=" << config.create_visualization;
  ss << ")";
  return ss.str();
}

int suggest_sparsify_reactivate_limit(size_t num_detectors, int sparsify_base_degree) {
  return suggest_sparsify_reactivate_limit_capped(num_detectors, sparsify_base_degree,
                                                  std::numeric_limits<int>::max());
}

std::string Node::str() {
  std::stringstream ss;
  auto& self = *this;
  ss << "Node(";
  ss << "error_chain_idx=" << self.error_chain_idx << ", ";
  ss << "cost=" << self.cost << ", ";
  ss << "path_cost=" << self.path_cost << ", ";
  ss << "num_dets=" << self.num_dets << ", ";
  return ss.str();
}

bool Node::operator>(const Node& other) const {
  return cost > other.cost || (cost == other.cost && num_dets < other.num_dets);
}

double TesseractDecoder::get_detcost(
    size_t d, const std::vector<DetectorCostTuple>& detector_cost_tuples) const {
  return get_detcost(d, detector_cost_tuples, d2e);
}

double TesseractDecoder::get_detcost(size_t d,
                                     const std::vector<DetectorCostTuple>& detector_cost_tuples,
                                     const std::vector<std::vector<int>>& active_d2e) const {
  double min_cost = INF;
  uint32_t min_det_cost_det_count = std::numeric_limits<uint32_t>::max();
  double error_cost;
  ErrorCost ec;
  DetectorCostTuple dct;

  for (int ei : active_d2e[d]) {
    ec = error_costs[ei];
    if (ec.likelihood_cost * min_det_cost_det_count >=
        min_cost * errors[ei].symptom.detectors.size())
      break;

    dct = detector_cost_tuples[ei];
    if (!dct.error_blocked) {
      error_cost = ec.likelihood_cost;
      if (error_cost * min_det_cost_det_count < min_cost * dct.detectors_count) {
        min_cost = error_cost;
        min_det_cost_det_count = dct.detectors_count;
      }
    }
  }

  return (min_cost / min_det_cost_det_count) + config.det_penalty;
}

TesseractDecoder::TesseractDecoder(TesseractConfig config_) : config(std::move(config_)) {
  config.dem = common::flatten(config.dem);

  std::vector<size_t> dem_error_map(config.dem.count_errors());
  std::iota(dem_error_map.begin(), dem_error_map.end(), 0);

  if (config.merge_errors) {
    std::vector<size_t> merge_map;
    config.dem = common::merge_indistinguishable_errors(config.dem, merge_map);
    common::chain_error_maps(dem_error_map, merge_map);
  }

  std::vector<size_t> nonzero_map;
  config.dem = common::remove_zero_probability_errors(config.dem, nonzero_map);
  common::chain_error_maps(dem_error_map, nonzero_map);

  dem_error_to_error = std::move(dem_error_map);
  error_to_dem_error = common::invert_error_map(dem_error_to_error, config.dem.count_errors());

  if (config.det_orders.empty()) {
    config.det_orders.emplace_back(config.dem.count_detectors());
    std::iota(config.det_orders[0].begin(), config.det_orders[0].end(), 0);
  }
  validate_detector_orders(config.det_orders, config.dem.count_detectors());
  errors = get_errors_from_dem(config.dem);
  if (config.verbose) {
    for (auto& error : errors) {
      std::cout << error.str() << "\n";
    }
    std::cout << std::flush;
  }
  num_detectors = config.dem.count_detectors();
  num_errors = config.dem.count_errors();
  num_observables = config.dem.count_observables();
  initialize_structures(config.dem.count_detectors());
  dpu_live_state = create_dpu_live_state(*this);
  dpu_resident_state = create_dpu_resident_state(*this);
  if (config.create_visualization) {
    auto detectors = get_detector_coords(config.dem);
    visualizer.add_detector_coords(detectors);
    visualizer.add_errors(errors);
  }
}

void TesseractDecoder::initialize_structures(size_t num_detectors) {
  d2e.resize(num_detectors);
  edets.resize(num_errors);

  for (size_t ei = 0; ei < num_errors; ++ei) {
    edets[ei] = errors[ei].symptom.detectors;
    for (int d : edets[ei]) {
      d2e[d].push_back(ei);
    }
  }

  for (size_t i = 0; i < errors.size(); ++i) {
    error_costs.push_back({errors[i].likelihood_cost,
                           errors[i].likelihood_cost / errors[i].symptom.detectors.size()});
  }

  if (dpu_env_enabled("TESSERACT_DPU_RESIDENT") &&
      !dpu_env_enabled("TESSERACT_DPU_RESIDENT_CPU_D2E_ORDER")) {
    // Match the DPU bridge's order exactly (quantized cost / degree as an
    // exact rational, ties by error index) so a search resumed on the CPU
    // blocks the same sibling prefixes as the DPU did.
    const char* q_bits_text = std::getenv("TESSERACT_DPU_LIVE_Q_BITS");
    const int q_bits = q_bits_text == nullptr ? 16 : std::stoi(q_bits_text);
    std::vector<uint64_t> cost_q(num_errors);
    for (size_t ei = 0; ei < num_errors; ++ei) {
      cost_q[ei] = quantize_dpu_trace_cost(errors[ei].likelihood_cost, q_bits, "error likelihood cost");
    }
    for (size_t d = 0; d < num_detectors; ++d) {
      std::sort(d2e[d].begin(), d2e[d].end(), [&](size_t a, size_t b) {
        const uint64_t degree_a = errors[a].symptom.detectors.size();
        const uint64_t degree_b = errors[b].symptom.detectors.size();
        const uint64_t lhs = cost_q[a] * degree_b;
        const uint64_t rhs = cost_q[b] * degree_a;
        return lhs != rhs ? lhs < rhs : a < b;
      });
    }
  } else {
    for (size_t d = 0; d < num_detectors; ++d) {
      std::sort(d2e[d].begin(), d2e[d].end(), [this](size_t idx_a, size_t idx_b) {
        return error_costs[idx_a].min_cost < error_costs[idx_b].min_cost;
      });
    }
  }

  eneighbors.resize(num_errors);

  std::vector<boost::dynamic_bitset<>> edets_bitsets(num_errors,
                                                     boost::dynamic_bitset<>(num_detectors));
  for (size_t ei = 0; ei < num_errors; ++ei) {
    for (int d : edets[ei]) {
      edets_bitsets[ei][d] = 1;
    }
  }

  for (size_t ei = 0; ei < num_errors; ++ei) {
    boost::dynamic_bitset<> neighbor_set(num_detectors, false);
    for (int d : edets[ei]) {
      for (int oei : d2e[d]) {
        // Unify detectors from neighboring errors
        neighbor_set |= edets_bitsets[oei];
      }
    }
    // Remove detectors from error's own set
    neighbor_set &= ~edets_bitsets[ei];

    for (size_t d = neighbor_set.find_first(); d != boost::dynamic_bitset<>::npos;
         d = neighbor_set.find_next(d)) {
      eneighbors[ei].push_back(d);
    }
  }

  if (config.sparsify_errors) {
    if (config.sparsify_base_degree <= 0) {
      throw std::invalid_argument(
          "sparsify_base_degree must be > 0 when sparsify_errors is enabled.");
    }
    if (config.sparsify_max_degree < -1) {
      throw std::invalid_argument("sparsify_max_degree must be >= -1.");
    }
    if (config.sparsify_reactivate_limit < -1) {
      throw std::invalid_argument("sparsify_reactivate_limit must be >= -1.");
    }
    if (config.sparsify_max_degree >= 0 &&
        config.sparsify_max_degree < config.sparsify_base_degree) {
      throw std::invalid_argument("sparsify_max_degree must be >= sparsify_base_degree.");
    }

    if (config.sparsify_reactivate_limit == -1) {
      int error_count_limit = static_cast<int>(
          std::min(num_errors, static_cast<size_t>(std::numeric_limits<int>::max())));
      config.sparsify_reactivate_limit = suggest_sparsify_reactivate_limit_capped(
          config.dem.count_detectors(), config.sparsify_base_degree, error_count_limit);
    }

    sparsify_mandatory_errors.clear();
    sparsify_optional_errors.clear();
    for (size_t ei = 0; ei < num_errors; ++ei) {
      int degree = errors[ei].symptom.detectors.size();
      if (degree <= config.sparsify_base_degree) {
        sparsify_mandatory_errors.push_back(ei);
      } else if (degree > config.sparsify_base_degree &&
                 (config.sparsify_max_degree == -1 || degree <= config.sparsify_max_degree)) {
        sparsify_optional_errors.push_back(ei);
      }
    }
    sparse_error_active.assign(num_errors, 0);
    sparse_d2e.resize(num_detectors);
  }
}

bool TesseractDecoder::decode_on_resident_dpu(const std::vector<uint64_t>& detections,
                                              size_t order_index,
                                              size_t detector_beam) {
#if defined(__GNUC__) && !defined(_WIN32)
  SearchStatsScope search_stats;
  const std::vector<size_t>& detector_order = config.det_orders.at(order_index);
  const bool reverse_order = detector_order.size() > 1 && detector_order.front() > detector_order.back();
  search_stats.set_resident_trial(order_index, detector_beam, reverse_order);
  std::vector<uint32_t> active;
  active.reserve(detections.size());
  for (uint64_t d : detections) {
    if (d >= num_detectors) {
      throw std::runtime_error("Symptom references a detector >= num_detectors.");
    }
    active.push_back(static_cast<uint32_t>(d));
  }
  std::array<uint16_t, 256> chain{};
  tesseract_dpu_resident_result_t result{};
  std::array<char, 512> error{};
  std::vector<uint32_t> oracle_heap;
  std::vector<uint32_t> oracle_arena;
  std::string fallback_reason = "none";
  const auto start = std::chrono::steady_clock::now();
  const uint32_t det_beam = detector_beam >= INF_DET_BEAM
                                ? std::numeric_limits<uint32_t>::max()
                                : static_cast<uint32_t>(detector_beam);
  if (dpu_resident_state->oracle) {
    ResidentOracle::Result r =
        dpu_resident_state->oracle->run(detections, order_index, det_beam);
    result.status = r.status;
    result.depth = static_cast<uint32_t>(r.chain.size());
    result.expansions = r.expansions;
    result.candidates = r.candidates;
    result.heap_size = static_cast<uint32_t>(r.heap_words.size() / 4);
    result.arena_used = static_cast<uint32_t>(r.arena_words.size() / 2);
    result.min_num_dets = r.min_num_dets;
    result.max_num_dets = r.max_num_dets;
    if (r.status == 0 && r.chain.size() > chain.size()) {
      result.status = 2;
      fallback_reason = "solution_depth";
    } else {
      std::copy(r.chain.begin(), r.chain.end(), chain.begin());
    }
    if (result.status == 2 && fallback_reason == "none") {
      fallback_reason = r.fallback_reason;
    }
    oracle_heap = std::move(r.heap_words);
    oracle_arena = std::move(r.arena_words);
  } else {
    const int status = tesseract_dpu_resident_decode(
        dpu_resident_state->handle, static_cast<uint32_t>(order_index), det_beam, active.data(),
        active.size(), chain.data(), chain.size(), &result, error.data(), error.size());
    if (status != 0) {
      throw std::runtime_error(std::string("DPU resident decode failed: ") + error.data());
    }
  }
  dpu_resident_state->seconds +=
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  ++dpu_resident_state->shots;
  dpu_resident_state->expansions += result.expansions;
  dpu_resident_state->candidates += result.candidates;
  search_stats.add_counts(result.expansions, result.candidates);
  search_stats.count_dpu_batch();
  if (result.status == 2 && fallback_reason == "none") {
    fallback_reason = "kernel_unspecified";
  }
  search_stats.set_resident_result(
      result.status, fallback_reason, result.heap_size, result.arena_used);
  if (result.status == 0) {
    double solution_cost = 0;
    for (uint32_t i = 0; i < result.depth; ++i) {
      solution_cost += errors[chain[i]].likelihood_cost;
    }
    search_stats.set_solution_cost(solution_cost);
    search_stats.set_solution_depth(result.depth);
  }
  if (dpu_env_enabled("TESSERACT_DPU_RESIDENT_DEBUG")) {
    double actual_path_cost = 0;
    std::cerr << "Tesseract resident result: status=" << result.status
              << " depth=" << result.depth << " fixed_expansions=" << result.expansions
              << " fixed_candidates=" << result.candidates << " chain=";
    for (uint32_t i = 0; i < result.depth; ++i) {
      actual_path_cost += errors[chain[i]].likelihood_cost;
      if (i != 0) std::cerr << ',';
      std::cerr << chain[i];
    }
    std::cerr << " actual_path_cost=" << actual_path_cost << std::endl;
  }
  if (result.status == 2) {
    ++dpu_resident_state->fallbacks;
    if (!dpu_resident_state->resume || result.heap_size == 0 ||
        (!dpu_resident_state->oracle && tesseract_dpu_resident_export == nullptr)) {
      return false;
    }
    // Continue the DPU's search on the CPU from its remaining queue.
    std::vector<uint32_t> heap_words(static_cast<size_t>(result.heap_size) * 4);
    std::vector<uint32_t> arena_words(static_cast<size_t>(result.arena_used) * 2);
    if (dpu_resident_state->oracle) {
      heap_words = std::move(oracle_heap);
      arena_words = std::move(oracle_arena);
    } else if (tesseract_dpu_resident_export(dpu_resident_state->handle, heap_words.data(),
                                             result.heap_size, arena_words.data(),
                                             result.arena_used, error.data(),
                                             error.size()) != 0) {
      throw std::runtime_error(std::string("DPU resident export failed: ") + error.data());
    }
    DpuResumeState state;
    state.arena.resize(result.arena_used);
    for (uint32_t i = 0; i < result.arena_used; ++i) {
      const uint32_t parent = arena_words[2 * i];
      const uint32_t packed = arena_words[2 * i + 1];
      state.arena[i].error_index = packed & 0xFFFFu;
      state.arena[i].min_detector = packed >> 16;
      state.arena[i].parent_idx = parent == 0xFFFFFFFFu ? -1 : static_cast<int64_t>(parent);
    }
    const double q_scale = std::ldexp(1.0, -static_cast<int>(dpu_resident_state->q_bits));
    state.nodes.reserve(result.heap_size);
    for (uint32_t i = 0; i < result.heap_size; ++i) {
      const uint32_t* w = &heap_words[4 * i];
      Node node;
      node.cost = static_cast<double>(w[0]) * q_scale;
      node.path_cost = static_cast<double>(w[1]) * q_scale;
      node.error_chain_idx = w[2] == 0xFFFFFFFFu ? -1 : static_cast<int64_t>(w[2]);
      node.num_dets = w[3] & 0xFFFFu;
      node.depth = w[3] >> 16;
      state.nodes.push_back(node);
    }
    state.min_num_dets = result.min_num_dets;
    state.max_num_dets = result.max_num_dets == 0xFFFFFFFFu
                             ? std::numeric_limits<size_t>::max()
                             : result.max_num_dets;
    ++dpu_resident_state->resumed;
    decode_to_errors_with_graph(detections, order_index, detector_beam, d2e, &state);
    return true;
  }
  predicted_errors_buffer.clear();
  if (result.status == 1) {
    ++dpu_resident_state->low_confidence;
    low_confidence_flag = true;
    return true;
  }
  ++dpu_resident_state->solved;
  low_confidence_flag = false;
  predicted_errors_buffer.reserve(result.depth);
  for (uint32_t i = 0; i < result.depth; ++i) {
    predicted_errors_buffer.push_back(error_to_dem_error[chain[i]]);
  }
  return true;
#else
  (void)detections;
  return false;
#endif
}

void TesseractDecoder::decode_to_errors(const std::vector<uint64_t>& detections) {
  if (config.sparsify_errors) {
    build_sparse_d2e(detections);
  }
  const auto& active_d2e = config.sparsify_errors ? sparse_d2e : d2e;

  std::vector<size_t> best_errors;
  double best_cost = std::numeric_limits<double>::max();
  if (config.det_orders.empty()) {
    throw std::runtime_error("Detector orders list must not be empty before decoding.");
  }

  if (dpu_resident_state != nullptr) {
    auto run_resident_trial = [&](size_t order_index, size_t detector_beam) {
      if (!decode_on_resident_dpu(detections, order_index, detector_beam)) {
        // A resident DPU fallback without an exportable/resumable queue must
        // restart this order on the CPU.  In particular, do not score
        // predicted_errors_buffer here: decode_on_resident_dpu deliberately
        // leaves it untouched on this path, so it may still contain the
        // previous shot's result.
        decode_to_errors_with_graph(detections, order_index, detector_beam, active_d2e);
      }
      double local_cost = cost_from_errors(predicted_errors_buffer);
      if (!low_confidence_flag && local_cost < best_cost) {
        best_errors = predicted_errors_buffer;
        best_cost = local_cost;
      }
    };
    if (config.beam_climbing) {
      int beam = 0;
      int order_index = 0;
      const int trials = std::max(config.det_beam + 1, int(config.det_orders.size()));
      for (int trial = 0; trial < trials; ++trial) {
        run_resident_trial(static_cast<size_t>(order_index), static_cast<size_t>(beam));
        beam = (beam + 1) % (config.det_beam + 1);
        order_index = (order_index + 1) % config.det_orders.size();
      }
    } else {
      for (size_t order_index = 0; order_index < config.det_orders.size(); ++order_index) {
        run_resident_trial(order_index, config.det_beam);
      }
    }
    predicted_errors_buffer = best_errors;
    low_confidence_flag = best_cost == std::numeric_limits<double>::max();
    return;
  }

  if (config.beam_climbing) {
    int beam = 0;
    int order_index = 0;
    for (int trial = 0; trial < std::max(config.det_beam + 1, int(config.det_orders.size()));
         ++trial) {
      decode_to_errors_with_graph(detections, order_index, beam, active_d2e);
      double local_cost = cost_from_errors(predicted_errors_buffer);
      if (!low_confidence_flag && local_cost < best_cost) {
        best_errors = predicted_errors_buffer;
        best_cost = local_cost;
      }
      if (config.verbose) {
        std::cout << "for detector_order " << order_index << " beam " << beam
                  << " got low confidence " << low_confidence_flag << " and cost " << local_cost
                  << " and obs_mask " << get_flipped_observables(predicted_errors_buffer)
                  << ". Best cost so far: " << best_cost << std::endl;
      }
      beam += 1;
      order_index += 1;
      beam %= (config.det_beam + 1);
      order_index %= config.det_orders.size();
    }
  } else {
    for (size_t order_index = 0; order_index < config.det_orders.size(); ++order_index) {
      decode_to_errors_with_graph(detections, order_index, config.det_beam, active_d2e);
      double local_cost = cost_from_errors(predicted_errors_buffer);
      if (!low_confidence_flag && local_cost < best_cost) {
        best_errors = predicted_errors_buffer;
        best_cost = local_cost;
      }
      if (config.verbose) {
        std::cout << "for detector_order " << order_index << " beam " << config.det_beam
                  << " got low confidence " << low_confidence_flag << " and cost " << local_cost
                  << " and obs_mask " << get_flipped_observables(predicted_errors_buffer)
                  << ". Best cost so far: " << best_cost << std::endl;
      }
    }
  }
  predicted_errors_buffer = best_errors;
  low_confidence_flag = best_cost == std::numeric_limits<double>::max();
}

void TesseractDecoder::flip_detectors_and_block_errors(
    size_t detector_order_index, int64_t error_chain_idx, boost::dynamic_bitset<>& detectors,
    std::vector<DetectorCostTuple>& detector_cost_tuples,
    const std::vector<std::vector<int>>& active_d2e) const {
  int64_t walker_idx = error_chain_idx;
  while (walker_idx != -1) {
    const auto& node = error_chain_arena[walker_idx];
    size_t ei = node.error_index;
    size_t min_detector = node.min_detector;

    for (int oei : active_d2e[min_detector]) {
      detector_cost_tuples[oei].error_blocked = 1;
      if (oei == ei) break;
    }

    for (int d : edets[ei]) {
      detectors[d] = !detectors[d];
    }
    walker_idx = node.parent_idx;
  }
}

void TesseractDecoder::decode_to_errors(const std::vector<uint64_t>& detections,
                                        size_t detector_order_index, size_t detector_beam) {
  if (config.sparsify_errors) {
    build_sparse_d2e(detections);
  }
  const auto& active_d2e = config.sparsify_errors ? sparse_d2e : d2e;
  decode_to_errors_with_graph(detections, detector_order_index, detector_beam, active_d2e);
}

void TesseractDecoder::decode_to_errors_with_graph(
    const std::vector<uint64_t>& detections, size_t detector_order_index, size_t detector_beam,
    const std::vector<std::vector<int>>& active_d2e, const DpuResumeState* resume) {
  const std::vector<size_t>& detector_at_position = config.det_orders.at(detector_order_index);

  predicted_errors_buffer.clear();
  low_confidence_flag = false;
  error_chain_arena.clear();
  DpuTraceWriter dpu_trace(*this, active_d2e);
  SearchStatsScope search_stats;
  const std::vector<uint64_t> dpu_inactive_error_words =
      make_dpu_inactive_error_words(dpu_live_state, active_d2e, num_errors);
  // Can technically be larger than pqlimit, but we need an initial guess on how many nodes we
  // will process from the queue. Only reserve if pqlimit is a reasonable (finite) value;
  // reserving SIZE_MAX bytes would throw std::length_error.
  if (config.pqlimit != std::numeric_limits<size_t>::max()) {
    error_chain_arena.reserve(config.pqlimit);
  }

  std::priority_queue<Node, std::vector<Node>, std::greater<Node>> pq;
  std::unordered_map<size_t, std::unordered_set<boost::dynamic_bitset<>>> visited_detectors;

  boost::dynamic_bitset<> initial_detectors(num_detectors, false);
  std::vector<DetectorCostTuple> initial_detector_cost_tuples(num_errors);

  for (size_t d : detections) {
    if (d >= num_detectors) {
      throw std::runtime_error(
          "Symptom " + std::to_string(d) +
          " references a detector >= num_detectors (= " + std::to_string(num_detectors) + ").");
    }
    initial_detectors[d] = true;
    for (int ei : active_d2e[d]) {
      ++initial_detector_cost_tuples[ei].detectors_count;
    }
  }
  dpu_trace.add_candidate(initial_detectors, initial_detector_cost_tuples);

  double initial_cost = 0;
  if (dpu_live_state != nullptr) {
    // The root is the only queue entry, so its heuristic cannot affect A* ordering.
    // Avoid a one-candidate DPU launch, but retain the CPU path's early rejection
    // when an active detector has no incident active error.
    for (size_t d : detections) {
      if (active_d2e[d].empty()) {
        initial_cost = INF;
        break;
      }
    }
  } else {
    for (size_t d : detections) {
      initial_cost += get_detcost(d, initial_detector_cost_tuples, active_d2e);
    }
  }

  if (initial_cost == INF) {
    low_confidence_flag = true;
    return;
  }

  size_t min_num_dets = detections.size();
  size_t max_num_dets = min_num_dets + detector_beam;

  boost::dynamic_bitset<> next_detectors;
  std::vector<DetectorCostTuple> next_detector_cost_tuples;

  size_t num_pq_pushed = 1;
  if (resume != nullptr) {
    error_chain_arena = resume->arena;
    for (const Node& node : resume->nodes) {
      pq.push(node);
    }
    min_num_dets = resume->min_num_dets;
    max_num_dets = resume->max_num_dets;
    num_pq_pushed = resume->nodes.size();
  } else {
    pq.push({initial_cost, 0, min_num_dets, 0, -1});
  }

  struct PendingDpuCandidate {
    size_t error_index;
    size_t min_detector;
    size_t num_dets;
    size_t depth;
    int64_t parent_idx;
    double path_cost;
  };

  while (!pq.empty()) {
    const Node node = pq.top();
    pq.pop();

    if (node.num_dets > max_num_dets) continue;

    boost::dynamic_bitset<> detectors = initial_detectors;
    std::vector<DetectorCostTuple> detector_cost_tuples(num_errors);
    flip_detectors_and_block_errors(detector_order_index, node.error_chain_idx, detectors,
                                    detector_cost_tuples, active_d2e);

    if (node.num_dets == 0) {
      search_stats.set_solution_cost(node.path_cost);
      search_stats.set_solution_depth(node.depth);
      if (config.create_visualization) {
        visualizer.add_activated_errors(node.error_chain_idx, error_chain_arena);
        visualizer.add_activated_detectors(detectors, num_detectors);
      }
      if (config.verbose) {
        std::cout << "activated_errors = ";
        int64_t walker_idx = node.error_chain_idx;
        while (walker_idx != -1) {
          std::cout << error_chain_arena[walker_idx].error_index << ", ";
          walker_idx = error_chain_arena[walker_idx].parent_idx;
        }
        std::cout << std::endl;
        std::cout << "activated_detectors = ";
        for (size_t d = 0; d < num_detectors; ++d) {
          if (detectors[d]) {
            std::cout << d << ", ";
          }
        }
        std::cout << std::endl;
        std::cout.precision(13);
        std::cout << "Decoding complete. Cost: " << node.cost
                  << " num_pq_pushed = " << num_pq_pushed << std::endl;
      }
      predicted_errors_buffer.resize(node.depth);
      int64_t walker_idx = node.error_chain_idx;
      for (size_t i = 0; i < node.depth; ++i) {
        predicted_errors_buffer[node.depth - 1 - i] =
            error_to_dem_error[error_chain_arena[walker_idx].error_index];
        walker_idx = error_chain_arena[walker_idx].parent_idx;
      }
      return;
    }

    if (config.no_revisit_dets && !visited_detectors[node.num_dets].insert(detectors).second)
      continue;

    if (config.create_visualization) {
      visualizer.add_activated_errors(node.error_chain_idx, error_chain_arena);
      visualizer.add_activated_detectors(detectors, num_detectors);
    }
    if (config.verbose) {
      std::cout.precision(13);
      std::cout << "len(pq) = " << pq.size() << " num_pq_pushed = " << num_pq_pushed << std::endl;
      std::cout << "num_dets = " << node.num_dets << " max_num_dets = " << max_num_dets
                << " cost = " << node.cost << std::endl;
      std::cout << "activated_errors = ";
      int64_t walker_idx = node.error_chain_idx;
      while (walker_idx != -1) {
        std::cout << error_chain_arena[walker_idx].error_index << ", ";
        walker_idx = error_chain_arena[walker_idx].parent_idx;
      }
      std::cout << std::endl;
      std::cout << "activated_detectors = ";
      for (size_t d = 0; d < num_detectors; ++d) {
        if (detectors[d]) {
          std::cout << d << ", ";
        }
      }
      std::cout << std::endl;
    }

    if (node.num_dets < min_num_dets) {
      min_num_dets = node.num_dets;
      if (config.no_revisit_dets) {
        for (size_t i = min_num_dets + detector_beam + 1; i <= max_num_dets; ++i) {
          visited_detectors[i].clear();
        }
      }
      max_num_dets = std::min(max_num_dets, min_num_dets + detector_beam);
    }

    search_stats.count_expansion();
    for (size_t d = 0; d < num_detectors; ++d) {
      if (!detectors[d]) continue;
      for (int ei : active_d2e[d]) {
        ++detector_cost_tuples[ei].detectors_count;
      }
    }

    next_detector_cost_tuples = detector_cost_tuples;

    size_t min_detector = std::numeric_limits<size_t>::max();
    for (size_t position = 0; position < num_detectors; ++position) {
      const size_t detector = detector_at_position[position];
      if (detectors[detector]) {
        min_detector = detector;
        break;
      }
    }

    size_t prev_ei = std::numeric_limits<size_t>::max();
    std::vector<double> detector_cost_cache;
    if (dpu_live_state == nullptr) {
      detector_cost_cache.assign(num_detectors, -1);
    }
    DpuLiveBatch dpu_batch(dpu_live_state, dpu_inactive_error_words, num_errors,
                           active_d2e[min_detector].size());
    dpu_batch.begin_expansion(detector_cost_tuples);
    std::vector<PendingDpuCandidate> pending_dpu_candidates;
    pending_dpu_candidates.reserve(active_d2e[min_detector].size());

    for (int ei : active_d2e[min_detector]) {
      if (detector_cost_tuples[ei].error_blocked) continue;

      if (prev_ei != std::numeric_limits<size_t>::max()) {
        for (int d : edets[prev_ei]) {
          int fired = detectors[d] ? 1 : -1;
          for (int oei : active_d2e[d]) {
            next_detector_cost_tuples[oei].detectors_count += fired;
          }
        }
      }
      prev_ei = ei;

      next_detectors = detectors;
      next_detector_cost_tuples[ei].error_blocked = 1;

      const double next_path_cost = node.path_cost + errors[ei].likelihood_cost;
      double next_cost = node.cost + errors[ei].likelihood_cost;
      size_t next_num_dets = node.num_dets;

      for (int d : edets[ei]) {
        next_detectors[d] = !next_detectors[d];
        int fired = next_detectors[d] ? 1 : -1;
        next_num_dets += fired;
        for (int oei : active_d2e[d]) {
          next_detector_cost_tuples[oei].detectors_count += fired;
        }
      }

      if (next_num_dets > max_num_dets) continue;

      if (config.no_revisit_dets && visited_detectors[next_num_dets].find(next_detectors) !=
                                        visited_detectors[next_num_dets].end())
        continue;

      dpu_trace.add_candidate(next_detectors, next_detector_cost_tuples);
      search_stats.count_candidates(1);

      if (dpu_batch.enabled()) {
        dpu_batch.add_candidate(next_detectors, static_cast<size_t>(ei));
        pending_dpu_candidates.push_back({static_cast<size_t>(ei), min_detector, next_num_dets,
                                          node.depth + 1, node.error_chain_idx, next_path_cost});
        continue;
      }

      for (int d : edets[ei]) {
        if (detectors[d]) {
          if (detector_cost_cache[d] == -1) {
            detector_cost_cache[d] = get_detcost(d, detector_cost_tuples, active_d2e);
          }
          next_cost -= detector_cost_cache[d];
        } else {
          next_cost += get_detcost(d, next_detector_cost_tuples, active_d2e);
        }
      }

      for (int od : eneighbors[ei]) {
        if (!detectors[od] || !next_detectors[od]) continue;
        if (detector_cost_cache[od] == -1) {
          detector_cost_cache[od] = get_detcost(od, detector_cost_tuples, active_d2e);
        }
        next_cost -= detector_cost_cache[od];
        next_cost += get_detcost(od, next_detector_cost_tuples, active_d2e);
      }

      if (next_cost == INF) continue;

      // Create the error chain node for this candidate.
      error_chain_arena.emplace_back();
      auto& next_node = error_chain_arena.back();
      next_node.error_index = ei;
      next_node.min_detector = min_detector;
      next_node.parent_idx = node.error_chain_idx;

      pq.push({next_cost, next_path_cost, next_num_dets, node.depth + 1,
               (int64_t)(error_chain_arena.size() - 1)});
      ++num_pq_pushed;

      if (num_pq_pushed > config.pqlimit) {
        if (config.verbose) {
          std::cout << "setting low confidence flag" << std::endl;
        }
        low_confidence_flag = true;
        return;
      }
    }

    if (dpu_batch.enabled() && !pending_dpu_candidates.empty()) {
      search_stats.count_dpu_batch();
      const std::vector<uint32_t> costs_q = dpu_batch.evaluate();
      if (costs_q.size() != pending_dpu_candidates.size()) {
        throw std::runtime_error("DPU live batch returned the wrong output count.");
      }
      const double q_scale = std::ldexp(1.0, -static_cast<int>(dpu_live_state->q_bits));
      for (size_t index = 0; index < pending_dpu_candidates.size(); ++index) {
        if (costs_q[index] == kDpuTraceInfCost) {
          continue;
        }
        const PendingDpuCandidate& candidate = pending_dpu_candidates[index];
        const double next_cost = candidate.path_cost + static_cast<double>(costs_q[index]) * q_scale;

        error_chain_arena.emplace_back();
        auto& next_node = error_chain_arena.back();
        next_node.error_index = candidate.error_index;
        next_node.min_detector = candidate.min_detector;
        next_node.parent_idx = candidate.parent_idx;

        pq.push({next_cost, candidate.path_cost, candidate.num_dets, candidate.depth,
                 static_cast<int64_t>(error_chain_arena.size() - 1)});
        ++num_pq_pushed;
        if (num_pq_pushed > config.pqlimit) {
          if (config.verbose) {
            std::cout << "setting low confidence flag" << std::endl;
          }
          low_confidence_flag = true;
          return;
        }
      }
    }
  }

  if (!pq.empty()) {
    throw std::runtime_error("Priority queue should be empty after decoding failure.");
  }
  if (config.verbose) {
    std::cout << "Decoding failed to converge within beam limit." << std::endl;
  }
  low_confidence_flag = true;
}

double TesseractDecoder::cost_from_errors(const std::vector<size_t>& predicted_errors) const {
  double total_cost = 0;
  for (size_t dem_error_index : predicted_errors) {
    size_t error_index = dem_error_to_error.at(dem_error_index);
    if (error_index == std::numeric_limits<size_t>::max()) {
      throw std::invalid_argument("error index does not map to a retained decoder error");
    }
    total_cost += errors[error_index].likelihood_cost;
  }
  return total_cost;
}

std::vector<int> TesseractDecoder::get_flipped_observables(
    const std::vector<size_t>& predicted_errors) const {
  std::unordered_set<int> flipped_observables_set;

  // Iterate over all errors and compute the mask.
  // We use a set to perform an XOR-like sum.
  // If an observable is already in the set, we remove it (XORing with itself).
  // If it's not, we add it.
  for (size_t dem_error_index : predicted_errors) {
    size_t error_index = dem_error_to_error.at(dem_error_index);
    if (error_index == std::numeric_limits<size_t>::max()) {
      throw std::invalid_argument("error index does not map to a retained decoder error");
    }
    for (int obs_index : errors[error_index].symptom.observables) {
      if (flipped_observables_set.count(obs_index)) {
        flipped_observables_set.erase(obs_index);
      } else {
        flipped_observables_set.insert(obs_index);
      }
    }
  }

  // Convert the set to a vector and return it.
  std::vector<int> flipped_observables(flipped_observables_set.begin(),
                                       flipped_observables_set.end());
  // Sort observables
  std::sort(flipped_observables.begin(), flipped_observables.end());
  return flipped_observables;
}

std::vector<int> TesseractDecoder::decode(const std::vector<uint64_t>& detections) {
  decode_to_errors(detections);
  return get_flipped_observables(predicted_errors_buffer);
}

void TesseractDecoder::decode_shots(std::vector<stim::SparseShot>& shots,
                                    std::vector<std::vector<int>>& obs_predicted) {
  obs_predicted.resize(shots.size());
  for (size_t i = 0; i < shots.size(); ++i) {
    obs_predicted[i] = decode(shots[i].hits);
  }
}

void TesseractDecoder::build_sparse_d2e(const std::vector<uint64_t>& detections) {
  std::vector<uint8_t> shot_dets(num_detectors, 0);
  for (uint64_t d : detections) {
    if (d < num_detectors) {
      shot_dets[d] = 1;
    }
  }

  std::fill(sparse_error_active.begin(), sparse_error_active.end(), 0);

  for (int ei : sparsify_mandatory_errors) {
    sparse_error_active[ei] = 1;
  }

  struct OptionalErrorCandidate {
    int error_index;
    int overlap;
    int degree;
    double likelihood_cost;
  };

  std::vector<OptionalErrorCandidate> candidates;
  candidates.reserve(sparsify_optional_errors.size());

  for (int ei : sparsify_optional_errors) {
    int overlap = 0;
    for (int d : errors[ei].symptom.detectors) {
      if (shot_dets[d]) {
        overlap++;
      }
    }
    if (overlap > 0) {
      candidates.push_back({ei, overlap, static_cast<int>(errors[ei].symptom.detectors.size()),
                            errors[ei].likelihood_cost});
    }
  }

  std::sort(candidates.begin(), candidates.end(),
            [](const OptionalErrorCandidate& a, const OptionalErrorCandidate& b) {
              if (a.overlap != b.overlap) {
                return a.overlap > b.overlap;
              }
              if (a.degree != b.degree) {
                return a.degree < b.degree;
              }
              if (a.likelihood_cost != b.likelihood_cost) {
                return a.likelihood_cost < b.likelihood_cost;
              }
              return a.error_index < b.error_index;
            });

  size_t limit = std::min(static_cast<size_t>(config.sparsify_reactivate_limit), candidates.size());
  for (size_t i = 0; i < limit; ++i) {
    sparse_error_active[candidates[i].error_index] = 1;
  }

  for (size_t d = 0; d < num_detectors; ++d) {
    sparse_d2e[d].clear();
    for (int ei : d2e[d]) {
      if (sparse_error_active[ei]) {
        sparse_d2e[d].push_back(ei);
      }
    }
  }
}

}  // namespace tesseract_decoder
