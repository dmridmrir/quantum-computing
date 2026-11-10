#!/usr/bin/env node
// Reconstruct compute-only FCFS schedules for the completed nine-million-shot
// resident-oracle workload at several DPU pool sizes.  This is a cycle-model
// scheduling analysis, not a functional-simulator or hardware-utilization run.

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
const expectedShots = 9_000_000;
const poolSizes = [128, 256, 512, 1024, 2048, 2560];
const model = JSON.parse(fs.readFileSync(modelPath, 'utf8'));
const cyclesPerCandidate = model.dpu_kernel.cycles_per_candidate_ratio_of_sums;

function makePool(size) {
  return {
    size,
    heap: Array.from({ length: size }, (_, index) => index),
    candidates: new Float64Array(size),
    jobs: new Uint32Array(size),
  };
}

const pools = poolSizes.map(makePool);
let totalCandidates = 0;
let residentShots = 0;

function earlier(pool, left, right) {
  return pool.candidates[left] < pool.candidates[right] ||
    (pool.candidates[left] === pool.candidates[right] && left < right);
}

function dispatch(pool, candidates) {
  const dpu = pool.heap[0];
  pool.candidates[dpu] += candidates;
  pool.jobs[dpu]++;

  let position = 0;
  while (true) {
    const left = position * 2 + 1;
    if (left >= pool.size) break;
    const right = left + 1;
    const child = right < pool.size && earlier(pool, pool.heap[right], pool.heap[left])
      ? right
      : left;
    if (earlier(pool, pool.heap[position], pool.heap[child])) break;
    [pool.heap[position], pool.heap[child]] = [pool.heap[child], pool.heap[position]];
    position = child;
  }
}

async function readChunk(directory) {
  const chunkRoot = path.join(runRoot, directory);
  if (!fs.existsSync(path.join(chunkRoot, 'complete'))) {
    throw new Error(`Not a completed validation chunk: ${directory}`);
  }
  const input = path.join(chunkRoot, 'oracle', 'search_stats.csv');
  const stream = readline.createInterface({ input: fs.createReadStream(input) });
  let header = null;
  let candidatesIndex = -1;
  let kindIndex = -1;
  let chunkShots = 0;
  for await (const line of stream) {
    if (header === null) {
      header = line.split(',');
      candidatesIndex = header.indexOf('candidates');
      kindIndex = header.indexOf('kind');
      if (candidatesIndex < 0 || kindIndex < 0) throw new Error(`Unexpected CSV header: ${input}`);
      continue;
    }
    const fields = line.split(',');
    if (fields[kindIndex] !== 'resident_dpu') continue;
    const candidates = Number(fields[candidatesIndex]);
    if (!Number.isSafeInteger(candidates) || candidates < 0) {
      throw new Error(`Invalid candidate count in ${input}: ${fields[candidatesIndex]}`);
    }
    for (const pool of pools) dispatch(pool, candidates);
    totalCandidates += candidates;
    residentShots++;
    chunkShots++;
  }
  if (chunkShots !== 1_000_000) {
    throw new Error(`${directory}: expected 1,000,000 resident shots, got ${chunkShots}`);
  }
}

async function main() {
  for (let start = 0; start < expectedShots; start += 1_000_000) {
    const directory = `${String(start).padStart(9, '0')}-${String(start + 1_000_000).padStart(9, '0')}`;
    await readChunk(directory);
  }
  if (residentShots !== expectedShots) throw new Error(`Expected ${expectedShots} shots, got ${residentShots}`);

  const totalModeledCycles = totalCandidates * cyclesPerCandidate;
  const schedules = pools.map(pool => {
    let least = 0;
    let most = 0;
    for (let dpu = 1; dpu < pool.size; dpu++) {
      if (pool.candidates[dpu] < pool.candidates[least]) least = dpu;
      if (pool.candidates[dpu] > pool.candidates[most]) most = dpu;
    }
    const makespanCycles = pool.candidates[most] * cyclesPerCandidate;
    const averageActiveDpus = totalModeledCycles / makespanCycles;
    return {
      modeled_dpus: pool.size,
      modeled_makespan_cycles: makespanCycles,
      average_active_dpus: averageActiveDpus,
      mean_active_fraction: averageActiveDpus / pool.size,
      least_busy_dpu: {
        id: least,
        modeled_cycles: pool.candidates[least] * cyclesPerCandidate,
        assigned_shots: pool.jobs[least],
        candidates: pool.candidates[least],
      },
      most_busy_dpu: {
        id: most,
        modeled_cycles: pool.candidates[most] * cyclesPerCandidate,
        assigned_shots: pool.jobs[most],
        candidates: pool.candidates[most],
      },
    };
  });

  process.stdout.write(`${JSON.stringify({
    evidence: 'FCFS reconstruction from nine million logged resident attempts; not measured simulator occupancy',
    completed_shots: residentShots,
    pool_sizes: poolSizes,
    cycles_per_candidate: cyclesPerCandidate,
    total_candidates: totalCandidates,
    total_modeled_dpu_cycles: totalModeledCycles,
    schedules,
  }, null, 2)}\n`);
}

main().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
