#!/usr/bin/env node
// Compute-only, first-come DPU scheduling of the nine completed 100M-validation chunks.
// Input work is measured by the resident C++ oracle; DPU time is a cycle-model estimate.

const fs = require('node:fs');
const path = require('node:path');
const readline = require('node:readline');

const benchRoot = path.resolve(__dirname, '..');
const runRoot = path.join(benchRoot, 'results', 'full_validation_100m', 'chunks');
const modelPath = path.join(
  benchRoot,
  'results',
  'resident',
  'timing_model_verified_1syndrome_actual_fallbacks.json',
);
const tracePath = path.join(
  benchRoot,
  'results',
  'resident',
  'cycles_sample32_astar_final_t16.csv',
);
const dpuCount = 2560;
const expectedShots = 9_000_000;
const model = JSON.parse(fs.readFileSync(modelPath, 'utf8'));
const cyclesPerCandidate = model.dpu_kernel.cycles_per_candidate_ratio_of_sums;
const pessimisticCyclesPerCandidate = model.dpu_kernel.linear_fit.slope_cycles_per_candidate;
const cyclesPerSecond = model.inputs.dpu_mhz * 1_000_000;

const traceLines = fs.readFileSync(tracePath, 'utf8').trim().split(/\r?\n/);
const traceCandidateCounts = traceLines.slice(1).map(line => Number(line.split(',')[1]));
const traceMinCandidates = Math.min(...traceCandidateCounts);
const traceMaxCandidates = Math.max(...traceCandidateCounts);

const heap = Array.from({ length: dpuCount }, (_, index) => index);
const busySeconds = new Float64Array(dpuCount);
const candidatesPerDpu = new Float64Array(dpuCount);
const jobsPerDpu = new Uint32Array(dpuCount);
const lastShotPerDpu = new Int32Array(dpuCount);
const lastJobCandidatesPerDpu = new Uint32Array(dpuCount);
let totalCandidates = 0;
let totalBusySeconds = 0;
let zeroCandidateShots = 0;
let shotsBeyondTraceMax = 0;
let candidatesBeyondTraceMax = 0;
let minShotCandidates = Infinity;
let maxShotCandidates = 0;
let residentShots = 0;
let cpuContinuationRows = 0;
const chunks = [];

function earlier(left, right) {
  return busySeconds[left] < busySeconds[right] ||
    (busySeconds[left] === busySeconds[right] && left < right);
}

function dispatch(candidates) {
  const dpu = heap[0];
  const duration = candidates * cyclesPerCandidate / cyclesPerSecond;
  lastShotPerDpu[dpu] = residentShots + 1;
  lastJobCandidatesPerDpu[dpu] = candidates;
  busySeconds[dpu] += duration;
  candidatesPerDpu[dpu] += candidates;
  jobsPerDpu[dpu]++;
  totalBusySeconds += duration;
  totalCandidates += candidates;
  minShotCandidates = Math.min(minShotCandidates, candidates);
  maxShotCandidates = Math.max(maxShotCandidates, candidates);
  if (candidates === 0) zeroCandidateShots++;
  if (candidates > traceMaxCandidates) {
    shotsBeyondTraceMax++;
    candidatesBeyondTraceMax += candidates;
  }
  residentShots++;

  // The available-at time increased for the root; restore the min-heap.
  let position = 0;
  while (true) {
    const left = position * 2 + 1;
    if (left >= dpuCount) break;
    const right = left + 1;
    const child = right < dpuCount && earlier(heap[right], heap[left]) ? right : left;
    if (earlier(heap[position], heap[child])) break;
    [heap[position], heap[child]] = [heap[child], heap[position]];
    position = child;
  }
}

async function readChunk(directory, start) {
  const chunkRoot = path.join(runRoot, directory);
  if (!fs.existsSync(path.join(chunkRoot, 'complete'))) {
    throw new Error(`Not a completed CPU/oracle validation chunk: ${directory}`);
  }
  const stats = JSON.parse(fs.readFileSync(path.join(chunkRoot, 'oracle', 'stats.json'), 'utf8'));
  if (stats.num_shots !== 1_000_000 || stats.num_det_orders !== 1 ||
      stats.num_inflight_shots !== 1 || stats.num_threads !== 1) {
    throw new Error(`Unexpected oracle workload configuration: ${directory}`);
  }
  const input = path.join(chunkRoot, 'oracle', 'search_stats.csv');
  const stream = readline.createInterface({ input: fs.createReadStream(input) });
  let header = null;
  let chunkShots = 0;
  let chunkContinuations = 0;
  for await (const line of stream) {
    if (header === null) {
      header = line.split(',');
      for (const required of ['candidates', 'kind']) {
        if (!header.includes(required)) throw new Error(`Missing CSV field ${required}: ${input}`);
      }
      continue;
    }
    const fields = line.split(',');
    const kind = fields[header.indexOf('kind')];
    if (kind === 'resident_dpu') {
      const candidates = Number(fields[header.indexOf('candidates')]);
      if (!Number.isSafeInteger(candidates) || candidates < 0) {
        throw new Error(`Bad candidate count in ${input}: ${fields[header.indexOf('candidates')]}`);
      }
      dispatch(candidates);
      chunkShots++;
    } else if (kind === 'cpu_search') {
      chunkContinuations++;
      cpuContinuationRows++;
    } else {
      throw new Error(`Unexpected search-stat kind in ${input}: ${kind}`);
    }
  }
  if (chunkShots !== stats.num_shots) {
    throw new Error(`${directory}: expected ${stats.num_shots} resident shots, found ${chunkShots}`);
  }
  chunks.push({ start_shot: start, resident_shots: chunkShots, cpu_continuation_rows: chunkContinuations });
}

async function main() {
  for (let start = 0; start < expectedShots; start += 1_000_000) {
    const directory = `${String(start).padStart(9, '0')}-${String(start + 1_000_000).padStart(9, '0')}`;
    await readChunk(directory, start);
  }
  if (residentShots !== expectedShots) throw new Error(`Expected ${expectedShots} shots, found ${residentShots}`);

  let least = 0;
  let most = 0;
  for (let dpu = 1; dpu < dpuCount; dpu++) {
    if (busySeconds[dpu] < busySeconds[least]) least = dpu;
    if (busySeconds[dpu] > busySeconds[most]) most = dpu;
  }
  const makespanSeconds = busySeconds[most];
  const scaleToPessimistic = pessimisticCyclesPerCandidate / cyclesPerCandidate;
  const totalAssignedShots = jobsPerDpu.reduce((total, count) => total + count, 0);
  if (totalAssignedShots !== residentShots) throw new Error('DPU assignment count mismatch');
  const result = {
    evidence: 'nine completed one-million-shot resident C++ oracle chunks, not a 2,560-DPU simulator run',
    completed_shots: residentShots,
    cpu_continuation_rows: cpuContinuationRows,
    source_configuration: 'one detector order; one CPU thread; one in-flight shot',
    schedule: 'one shot per DPU; first-come, immediately dispatch to earliest-free DPU; no host/transfer/rank delays',
    modeled_dpus: dpuCount,
    maximum_simultaneously_active_dpus: Math.min(dpuCount, residentShots),
    first_shot_of_each_dpu_starts_at_seconds: 0,
    average_active_dpus: totalBusySeconds / makespanSeconds,
    mean_active_dpu_fraction: totalBusySeconds / (dpuCount * makespanSeconds),
    total_modeled_dpu_cycles: totalCandidates * cyclesPerCandidate,
    total_modeled_dpu_compute_seconds: totalBusySeconds,
    modeled_makespan_seconds: makespanSeconds,
    least_busy_dpu: {
      id: least, busy_seconds: busySeconds[least], assigned_shots: jobsPerDpu[least],
      candidates: candidatesPerDpu[least],
      modeled_cycles: candidatesPerDpu[least] * cyclesPerCandidate,
      final_global_shot_1_based: lastShotPerDpu[least],
      final_job_candidates: lastJobCandidatesPerDpu[least],
    },
    most_busy_dpu: {
      id: most, busy_seconds: busySeconds[most], assigned_shots: jobsPerDpu[most],
      candidates: candidatesPerDpu[most],
      modeled_cycles: candidatesPerDpu[most] * cyclesPerCandidate,
      final_global_shot_1_based: lastShotPerDpu[most],
      final_job_candidates: lastJobCandidatesPerDpu[most],
    },
    work: {
      resident_candidates_total: totalCandidates,
      candidate_min_per_shot: minShotCandidates,
      candidate_max_per_shot: maxShotCandidates,
      zero_candidate_shots: zeroCandidateShots,
      cycle_trace_candidate_min: traceMinCandidates,
      cycle_trace_candidate_max: traceMaxCandidates,
      shots_beyond_cycle_trace_max: shotsBeyondTraceMax,
      fraction_shots_beyond_cycle_trace_max: shotsBeyondTraceMax / residentShots,
      fraction_candidates_beyond_cycle_trace_max: candidatesBeyondTraceMax / totalCandidates,
    },
    cycle_model: {
      source: path.relative(benchRoot, modelPath).replaceAll('\\', '/'),
      traces: model.inputs.cycle_traces,
      dpu_mhz: model.inputs.dpu_mhz,
      primary_cycles_per_candidate: cyclesPerCandidate,
      alternate_cycles_per_candidate: pessimisticCyclesPerCandidate,
      alternate_least_busy_cycles: candidatesPerDpu[least] * pessimisticCyclesPerCandidate,
      alternate_most_busy_cycles: candidatesPerDpu[most] * pessimisticCyclesPerCandidate,
      alternate_least_busy_seconds: busySeconds[least] * scaleToPessimistic,
      alternate_most_busy_seconds: busySeconds[most] * scaleToPessimistic,
    },
    chunks,
  };
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
}

main().catch(error => { console.error(error); process.exitCode = 1; });
