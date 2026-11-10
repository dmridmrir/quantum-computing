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

#ifndef __TESSERACT_UTILS_H__
#define __TESSERACT_UTILS_H__

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include "common.h"
#include "stim.h"

namespace tesseract_decoder {

constexpr const double EPSILON = 1e-7;

// Returns detector coordinates keyed by detector ID. If the DEM contains any
// detector coordinate instructions, the returned vector has one entry per
// detector and an empty entry for each detector without declared coordinates.
// Returns an empty vector if the DEM has no detector coordinate instructions.
std::vector<std::vector<double>> get_detector_coords(const stim::DetectorErrorModel& dem);

// Builds an adjacency list graph where each positive-probability error's
// parity-reduced detector symptom induces a clique.
std::vector<std::vector<size_t>> build_detector_graph(const stim::DetectorErrorModel& dem);

enum class DetOrder { DetBFS, DetIndex, DetCoordinate };

// Builds detector traversal orders. Each inner vector uses the convention
// detector_at_position[position] = detector_id and is a permutation of all
// detector IDs in the DEM. Coordinate ordering projects all declared
// coordinate dimensions, treating missing trailing dimensions as zero and
// placing detectors without coordinates last. Seeded randomized orders are
// reproducible only within a fixed C++ standard-library implementation.
std::vector<std::vector<size_t>> build_det_orders(const stim::DetectorErrorModel& dem,
                                                  size_t num_det_orders,
                                                  DetOrder method = DetOrder::DetIndex,
                                                  uint64_t seed = 0);

const double INF = std::numeric_limits<double>::infinity();

bool sampling_from_dem(uint64_t seed, size_t num_shots, stim::DetectorErrorModel dem,
                       std::vector<stim::SparseShot>& shots);

void sample_shots(uint64_t sample_seed, stim::Circuit& circuit, size_t sample_num_shots,
                  std::vector<stim::SparseShot>& shots);

std::vector<common::Error> get_errors_from_dem(const stim::DetectorErrorModel& dem);

std::vector<std::string> get_files_recursive(const std::string& directory_path);

uint64_t vector_to_u64_mask(const std::vector<int>& v);

// Limits host CPU work while allowing more independent shot searches to stay
// in flight.  A worker returns its permit only while it is blocked in a DPU
// launch, so e.g. 16 host CPU workers can keep up to 2560 independent DPUs
// occupied without treating simulator implementation threads as CPU cores.
class CpuPermitPool {
 public:
  explicit CpuPermitPool(size_t permits) : available_(permits) {
    if (permits == 0) {
      throw std::invalid_argument("CPU permit count must be positive.");
    }
  }

  void acquire() {
    std::unique_lock<std::mutex> lock(mutex_);
    ready_.wait(lock, [&]() { return available_ != 0; });
    --available_;
  }

  void release() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      ++available_;
    }
    ready_.notify_one();
  }

 private:
  std::mutex mutex_;
  std::condition_variable ready_;
  size_t available_;
};

inline thread_local CpuPermitPool* active_cpu_permit_pool = nullptr;
inline thread_local bool active_cpu_permit_held = false;

inline void release_cpu_permit_for_dpu_wait() {
  if (active_cpu_permit_pool != nullptr && active_cpu_permit_held) {
    active_cpu_permit_held = false;
    active_cpu_permit_pool->release();
  }
}

inline void acquire_cpu_permit_after_dpu_wait() {
  if (active_cpu_permit_pool != nullptr && !active_cpu_permit_held) {
    active_cpu_permit_pool->acquire();
    active_cpu_permit_held = true;
  }
}

// Applies a shot-wise worker function in parallel while consuming completed
// shots in increasing order.
//
// process_shot(thread_index, shot_index):
//   - Runs on worker threads.
//   - thread_index is stable for each worker and lies in [0, num_threads).
//
// consume_shot(shot_index):
//   - Runs on the caller thread in increasing shot order.
//
// If consume_shot returns false, workers stop claiming new shots but always
// finish any shot they already started.
template <typename ProcessShot, typename ConsumeShot>
size_t parallel_for_shots_in_order_with_cpu_limit(
    size_t num_shots, size_t num_workers, size_t num_cpu_permits, ProcessShot&& process_shot,
    ConsumeShot&& consume_shot) {
  if (num_workers == 0 || num_cpu_permits == 0) {
    throw std::invalid_argument("shot worker and CPU permit counts must be positive.");
  }
  std::atomic<size_t> next_unclaimed_shot = 0;
  std::vector<std::atomic<bool>> finished(num_shots);
  for (auto& flag : finished) {
    flag = false;
  }
  std::atomic<bool> worker_threads_please_terminate = false;
  std::atomic<size_t> num_worker_threads_active = 0;
  std::vector<std::thread> workers;
  num_workers = std::min(num_workers, std::max<size_t>(num_shots, 1));
  workers.reserve(num_workers);
  CpuPermitPool cpu_permits(num_cpu_permits);

  for (size_t t = 0; t < num_workers; ++t) {
    ++num_worker_threads_active;
    workers.emplace_back([&, t]() {
      cpu_permits.acquire();
      active_cpu_permit_pool = &cpu_permits;
      active_cpu_permit_held = true;
      for (size_t shot;
           !worker_threads_please_terminate && ((shot = next_unclaimed_shot++) < num_shots);) {
        process_shot(t, shot);
        finished[shot] = true;
      }
      active_cpu_permit_pool = nullptr;
      active_cpu_permit_held = false;
      cpu_permits.release();
      --num_worker_threads_active;
    });
  }

  size_t shot = 0;
  for (; shot < num_shots; ++shot) {
    while (num_worker_threads_active && !finished[shot]) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (!finished[shot]) {
      assert(num_worker_threads_active == 0);
      break;
    }
    if (!consume_shot(shot)) {
      worker_threads_please_terminate = true;
    }
  }

  for (auto& worker : workers) {
    worker.join();
  }
  return shot;
}

template <typename ProcessShot, typename ConsumeShot>
size_t parallel_for_shots_in_order(size_t num_shots, size_t num_threads, ProcessShot&& process_shot,
                                   ConsumeShot&& consume_shot) {
  return parallel_for_shots_in_order_with_cpu_limit(
      num_shots, num_threads, num_threads, std::forward<ProcessShot>(process_shot),
      std::forward<ConsumeShot>(consume_shot));
}

}  // namespace tesseract_decoder

#endif  // __TESSERACT_UTILS_H__
