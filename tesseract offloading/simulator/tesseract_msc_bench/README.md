# MSC Tesseract DPU offloading simulator

> 현재 검증 결론과 처리량은 저장소 최상위 `README.md`를 기준으로 한다.
> 이 문서에 남은 2-group 및 기존 timing-model 설명은 실험 이력이며 현재의
> one-syndrome-per-DPU 최종 모델을 뜻하지 않는다.

This directory adapts uPIMulator for the MSC escape-stage Tesseract decoder.

## Objective

The primary experiment compares end-to-end decoding throughput before and
after offloading Tesseract candidate heuristic evaluation:

```text
CPU-only Tesseract throughput
             vs.
CPU search + DPU candidate evaluation throughput
```

The hybrid measurement must include batch construction, host-to-DPU transfer,
DPU execution, DPU-to-host transfer, and CPU result integration. Candidate
evaluation throughput and uPIMulator cycle counts are supporting metrics; they
must not be reported as end-to-end decode throughput by themselves.

The target is system throughput from 8 or 16 CPU workers plus 2,560 DPUs, not
single-core speedup. The reported speedup is the ratio of full end-to-end
hybrid throughput to full CPU-only throughput for the same syndrome stream and
CPU-worker count; it is not a DPU-core-versus-CPU-core comparison. One
syndrome/search is assigned to one DPU; 16 active tasklets in that DPU
cooperate on each sibling-candidate batch. Independent DPUs process independent
syndromes and never communicate with each other.

## Current implementation

- `uPIMulator/`: cycle-level UPMEM simulator core.
- `upmem_sdk/`: bundled Linux UPMEM SDK used to compile UPMEM-C without Docker.
- `uPIMulator/benchmark/TESSERACT/`: candidate-level DPU kernel.
- `uPIMulator/benchmark/TESSERACT_ASTAR/`: DPU-resident A* kernel; one DPU
  decodes one syndrome end to end (heap, chain arena, sibling evaluation) and
  returns the error chain or a CPU-fallback status. Enabled in Tesseract with
  `TESSERACT_DPU_RESIDENT=1`; cycle-level runs via
  `scripts/build_and_run_astar.sh` (`TESSERACT_ASTAR_GROUPS=1|2`,
  `TESSERACT_TRACE2` for a second syndrome), host-cost runs via
  `scripts/run_resident_search_stats.sh`, keep-if-faster gate via
  `scripts/astar_gate.sh`, model via `scripts/timing_model_resident.py`.
  Env: `TESSERACT_DPU_RESIDENT_CAPACITY` (heap/arena nodes),
  `TESSERACT_DPU_RESIDENT_NO_RESUME=1` (discard on fallback instead of CPU
  resume), `TESSERACT_DPU_RESIDENT_ORACLE=1` (host C++ port of the kernel, no
  UPMEM runtime; used for the 100,000-shot equivalence check).
- `uPIMulator/src/assembler/prim/tesseract.go`: deterministic input builder and
  CPU oracle for uPIMulator output checking.
- `scripts/build_and_run.sh`: Docker-free Linux build and smoke run.
- `scripts/build_tesseract.sh`: Python bindings 없이 공식 C++ decoder 빌드.
- `scripts/export_tesseract_trace.sh`: 공식 search의 고정 seed 1-shot trace 생성.
- `scripts/prepare_msc_workload.py`: MSC `.stim`에서 Tesseract용 `.dem`,
  postselected syndrome, observable 입력을 bounded sampling chunks로 생성.
  Exact survivor targets and bit-packed Stim `b8` output are supported.
- `scripts/export_tesseract_dem_trace.sh`: 저장된 MSC DEM/syndrome 한 shot을
  공식 Tesseract에 입력하고 실제 search batch trace 생성.
- `scripts/run_cpu_baseline.sh`: CPU-only throughput 기준 측정.
- `scripts/run_cpu_dem_baseline.sh`: 동일한 postselected MSC DEM/syndrome
  batch의 8/16-worker CPU-only throughput 측정.
- `scripts/run_async_host.sh`: bundled UPMEM SDK에서 2-slot asynchronous
  host→DPU→host trace replay와 세부 timing 측정.
- `scripts/run_live_dem_comparison.sh`: 동일 MSC `.dem`/syndrome을 CPU-only와
  live DPU A* feedback 경로로 streaming 실행하고 prediction/throughput 비교.
- `scripts/run_scaleout_extrapolation.sh`: 동일한 MSC 입력으로
  4/8/16/32/64-DPU를 측정하고 8/16 CPU별 2,560-DPU throughput 외삽.
- `scripts/run_scaleout_matrix.sh`: 작은 DPU pool의 직접 실행 검증용.
- `scripts/sweep_async_batches.sh`: batch size별 asynchronous candidate
  throughput과 전송량 sweep.
- `scripts/analyze_live_batches.py`: live run이 기록한 launch별 candidate 수의
  평균과 p50/p90/p99 및 크기 구간을 요약.
- `traces/example_v1.json`: minimal versioned trace and blocked-error example.
- `traces/official_color_d3_seed2.json`: upstream color-code search에서 생성한
  producer/consumer integration fixture.

The current assembler produces a deterministic synthetic detector error model
so that the MRAM contract and DPU kernel can be validated independently. The
same adapter can load batches recorded from the official CPU Tesseract
implementation and an MSC grafted-code DEM.

The official decoder is instrumented only when `TESSERACT_DPU_TRACE_PATH` is
set. Normal CPU-only runs do not create traces. Trace capture must use one shot
and one decoder thread because one output path represents one decode request.

## Offload boundary

The CPU keeps:

- A* search and priority queue
- pruning and no-revisit state
- candidate generation and termination

The DPU receives a batch and evaluates:

- detector-to-error CSR scans
- error-to-detector overlap counts
- fixed-point cost-LUT lookups
- per-detector minimum and per-candidate heuristic reduction

The DPU hot path is integer-only: probabilities are quantized on the host,
division is replaced by a precomputed Q-format overlap-cost LUT, residual sets
are packed 32-bit bitsets, and all reductions use saturating unsigned integer
additions. Candidate blocked errors use sorted 16-bit IDs when a candidate has
at most 128 blocked errors and the model has at most 65,536 errors. A 256-bit
WRAM Bloom filter rejects most negative membership checks before an exact
binary search. Larger sets automatically fall back to host-repacked 32-bit
bitsets. There are no floating-point, logarithm, or division instructions in
the DPU kernel.

In live mode the CPU keeps the exact accumulated selected-error likelihood as
a `double`. Only the residual heuristic is evaluated as Q-format integers on
the DPU; the returned value is rescaled and added to that exact path cost
before the node enters the A* priority queue. Siblings from one expansion form
one DPU batch. The root has no competing queue entry, so live mode only checks
that each root detector has an active incident error and avoids an otherwise
useless one-candidate DPU launch.

Candidate residual bitsets are padded to an 8-byte stride by the host and
loaded once per candidate into WRAM. Sequential detector-to-error ID scans use
a 64-byte MRAM block cache; random offset and LUT reads retain 8-byte transfers
to avoid over-fetching. Sparse blocked records use an aligned header read and
one aligned record DMA per candidate. Tasklets obtain candidates from a shared
dynamic work queue instead of a static round-robin assignment.

Measured before/after kernel and transfer results are recorded in
`OPTIMIZATION_RESULTS.md`.

## MRAM layout

All regions are 8-byte aligned and offsets are relative to
`DPU_MRAM_HEAP_POINTER`:

1. `d2e_offsets[]`
2. `d2e_error_ids[]` (each detector's list must be sorted ascending by
   `error_cost_q / degree`, ties by error ID; the kernel's lower-bound early
   exit reads `cost_lut[e][degree(e)]` and assumes this order)
3. `e2d_offsets[]`
4. `e2d_detector_ids[]`
5. `cost_lut_offsets[]`
6. `cost_lut_values[]`
7. `candidate_offsets[]`
8. `candidate_detector_ids[]`
9. `candidate_residual_words[]` (host-packed 32-bit syndrome bitsets with an
   even-word DMA stride per candidate)
10. optional sparse blocked-record offsets and packed 16-bit error IDs, or
    host-repacked 32-bit bitsets for the fallback path
11. 64-bit output slot per candidate (low 32 bits contain the Q-format cost)

## Docker-free build

The bundled SDK and its compiler are Linux executables. Run this from Ubuntu,
WSL, or a Linux server with Go 1.21.5+, CMake, Ninja, and Python 3:

```bash
cd simulator/tesseract_msc_bench
bash scripts/build_and_run.sh 11 64 256 512 42 16
```

Arguments are:

```text
tasklets candidates detectors errors seed q_bits
```

The cycle-level runner no longer hard-codes one DPU. Topology and simulator
implementation parallelism are configured independently, for example:

```bash
TESSERACT_NUM_CHANNELS=20 \
TESSERACT_NUM_RANKS_PER_CHANNEL=2 \
TESSERACT_NUM_DPUS_PER_RANK=64 \
TESSERACT_SIMULATION_THREADS=16 \
TESSERACT_TRACE=traces/official_msc_grafted_d3_d7.json \
  bash scripts/build_and_run.sh 11
```

The trace exporter infers matching `01` or `b8` inputs and defaults to a
single-shot I/O chunk, so exporting one trace from the 100M workload does not
materialize the full bit-packed input.

This is 20 × 2 × 64 = 2,560 simulated DPUs, with one independent trace replica
per DPU. `TESSERACT_SIMULATION_THREADS` only limits the Go simulator's worker
goroutines; it is not the 8/16-thread CPU decoder baseline. The summary
aggregates all DPU cycle counters and labels cycle-derived rates as
`dpu_kernel_only_*`.

uPIMulator's MRAM backing is lazy at 1 KiB wordline granularity. Consequently
a 2,560-DPU topology no longer allocates 2,560 × 64 MiB = 160 GiB before the
first cycle; only touched packed regions consume backing storage. Instruction
simulation work still grows with the number of explicitly simulated DPUs, so
the full run remains much slower than a small topology validation.

To replay a real or recorded candidate batch, set `TESSERACT_TRACE`. The trace
overrides the synthetic workload arguments:

```bash
TESSERACT_TRACE=traces/example_v1.json \
  bash scripts/build_and_run.sh 11
```

To build the pinned official C++ decoder, capture one real search, and replay
it through the simulator:

```bash
bash scripts/build_tesseract.sh
bash scripts/export_tesseract_trace.sh \
  ../../third_party/tesseract-decoder/testdata/colorcodes/r=1,d=3,p=0.2,noise=data_qubit_X,c=color_code_mpp,q=7,gates=mpp.stim \
  traces/official_color_d3_seed2.json 2 16
TESSERACT_TRACE=traces/official_color_d3_seed2.json \
  bash scripts/build_and_run.sh 11
```

## Real MSC grafted-code input

The reproducible fixture under `workloads/msc_grafted_d3_d7/` was generated
from the pinned official MSC repository with:

```text
--circuit_type end2end-inplace-distillation
--noise_strength 0.001 --style css --basis Y
--injection_distance 3 --cultivation_distance 3
--escape_distance 7 --escape_rounds 5
```

### Syndrome provenance

`prepare_msc_workload.py` loads that `.stim` circuit and constructs a DEM with
the same options used by Tesseract. Stim's detector sampler then produces
4,096 raw detector/observable shots with seed 1. The script reproduces the
official cultivation postselection detector set from detector coordinate
annotations plus the two-detector observable-flip rule, and rejects every raw
shot that fires any postselected detector. In the checked-in fixture, 2,256 of
4,096 raw shots survive and the first 64 survivors are retained.

Each line of `syndromes.01` is the resulting 494-bit detector-event vector;
the matching line of `observables.01` is its one-bit actual logical observable.
These are paired records and are never shuffled independently. Their syndrome
weights range from 4 to 29. Shot index 50, weight 4, is the smallest nonzero
syndrome and was selected for the original single-trace kernel correctness
fixture. Consequently, `official_msc_grafted_d3_d7.json` alone is not a
representative end-to-end throughput distribution: the scale-out experiment
uses all 64 `.01` records at every measured DPU count.

From the repository root, prepare a Tesseract-compatible DEM and real sampled
inputs on Windows using the isolated MSC Python environment:

```powershell
./simulator/tesseract_msc_bench/scripts/setup_msc_python.ps1
third_party/magic-state-cultivation/.venv/Scripts/python.exe `
  simulator/tesseract_msc_bench/scripts/prepare_msc_workload.py `
  --circuit simulator/tesseract_msc_bench/workloads/msc_grafted_d3_d7/c=end2end-inplace-distillation,p=0.001,noise=uniform,g=css,q=102,b=Y,r=13,r1=3,d1=3,r2=5,d2=7.stim `
  --out-dir simulator/tesseract_msc_bench/workloads/msc_grafted_d3_d7 `
  --raw-shots 4096 --max-kept 64 --seed 1
```

The checked-in circuit itself can be regenerated from the pinned upstream
source with:

```powershell
third_party/magic-state-cultivation/.venv/Scripts/python.exe `
  third_party/magic-state-cultivation/tools/make_circuits `
  --circuit_type end2end-inplace-distillation `
  --noise_strength 0.001 --gateset css --basis Y `
  --r1 3 --d1 3 --r2 5 --d2 7 `
  --out_dir simulator/tesseract_msc_bench/workloads/msc_grafted_d3_d7
```

`prepare_msc_workload.py` uses the same DEM conversion options as Tesseract
and reproduces the official cultivation postselection rules. The generated
`syndromes.01` contains detector bits only; `observables.01` contains the
separate actual observable flips. `metadata.json` records hashes, accepted raw
shot indices, syndrome weights, and a recommended nonzero shot.

For the production workload, sample until exactly `10^8` postselection
survivors are written. The sampler is compiled once and advances across bounded
chunks; Stim itself is not modified. `b8` reduces the 494-detector payload from
about 49.5 GB in text `01` form to about 6.2 GB:

```powershell
third_party/magic-state-cultivation/.venv/Scripts/python.exe `
  simulator/tesseract_msc_bench/scripts/prepare_msc_workload.py `
  --circuit simulator/tesseract_msc_bench/workloads/msc_grafted_d3_d7/c=end2end-inplace-distillation,p=0.001,noise=uniform,g=css,q=102,b=Y,r=13,r1=3,d1=3,r2=5,d2=7.stim `
  --out-dir simulator/tesseract_msc_bench/workloads/msc_grafted_d3_d7_100m `
  --target-kept 100000000 --sample-chunk-shots 100000 `
  --progress-every-chunks 100 `
  --metadata-sample-limit 100000 --output-format b8 --seed 1
```

The command reports cumulative progress every 10 million raw shots. For large
outputs, metadata retains only the configured prefix sample of raw
indices and per-shot weights. It always records the complete weight histogram,
counts, chunk size, seed, format, and SHA-256 hashes.

Replay that shot through the official Tesseract search and the DPU simulator:

```bash
bash scripts/export_tesseract_dem_trace.sh \
  workloads/msc_grafted_d3_d7/model.dem \
  workloads/msc_grafted_d3_d7/syndromes.01 \
  workloads/msc_grafted_d3_d7/observables.01 \
  traces/official_msc_grafted_d3_d7.json 50 16
TESSERACT_TRACE=traces/official_msc_grafted_d3_d7.json \
  bash scripts/build_and_run.sh 11
```

Measure the CPU-only baseline over the exact same 64 accepted syndromes:

```bash
bash scripts/run_cpu_dem_baseline.sh \
  workloads/msc_grafted_d3_d7/model.dem \
  workloads/msc_grafted_d3_d7/syndromes.01 \
  workloads/msc_grafted_d3_d7/observables.01 \
  results/cpu_msc_grafted_d3_d7_t8 64 8
bash scripts/run_cpu_dem_baseline.sh \
  workloads/msc_grafted_d3_d7/model.dem \
  workloads/msc_grafted_d3_d7/syndromes.01 \
  workloads/msc_grafted_d3_d7/observables.01 \
  results/cpu_msc_grafted_d3_d7_t16 64 16
```

Run the Docker-free asynchronous two-slot UPMEM host pipeline:

```bash
bash scripts/run_async_host.sh \
  traces/official_msc_grafted_d3_d7.json \
  16 3 results/async_host_msc_grafted_d3_d7 16
```

Run the actual live A* feedback path and compare it with CPU-only decoding over
the same MSC inputs (`SHOT_BEGIN`, `NUM_SHOTS`, `TASKLETS`, `Q_BITS`,
`CPU_THREADS`, `DPU_INFLIGHT` follow the result directory):

```bash
bash scripts/run_live_dem_comparison.sh \
  workloads/msc_grafted_d3_d7/model.dem \
  workloads/msc_grafted_d3_d7/syndromes.01 \
  workloads/msc_grafted_d3_d7/observables.01 \
  results/live_msc_grafted_d3_d7_all64 0 64 16 16 8 64
```

The normal Tesseract binary has only weak references to the live C ABI, so an
ordinary CPU run has no UPMEM SDK dependency. The comparison runner preloads
`libtesseract_dpu_live.so`. Each in-flight syndrome owns one DPU, loads the
static DEM/CSR/LUT once, and reuses that DPU for the sequential A* expansions
of the syndrome. Sixteen active tasklets cooperate inside each DPU. A blocking
DPU launch returns its CPU permit, allowing the configured 8/16 CPU workers to
advance other syndrome states while up to `--inflight-shots` DPUs run.
`--io-chunk-shots` bounds resident input and prediction storage while preserving
those decoder instances and their DPU-resident static state across chunks. The
runner defaults to `TESSERACT_IO_CHUNK_SHOTS=65536` and infers `01` or `b8` from
matching input extensions.

The primary system experiment measures 4/8/16/32/64 DPUs and extrapolates the
2,560-DPU point instead of explicitly cycle-simulating all 2,560 DPUs:

```bash
bash scripts/run_scaleout_extrapolation.sh \
  workloads/msc_grafted_d3_d7/model.dem \
  workloads/msc_grafted_d3_d7/syndromes.01 \
  workloads/msc_grafted_d3_d7/observables.01 \
  results/scaleout_extrapolation 64 4,8,16,32,64 16 8,16 2560
```

Every measured point decodes the same 64 paired syndrome/observable records,
so changing DPU count does not change the input distribution. The fit is
`seconds_per_shot = serial + parallel / dpus`, separately for 8 and 16 CPU
workers. The report includes R-squared and labels the 2,560-DPU value as an
extrapolation. The older `run_scaleout_matrix.sh` remains available for direct
small-pool validation but is not the primary 2,560-DPU methodology.

This 64-shot command is a functional pipeline check, not the final statistical
measurement. The formal run targets at least `10^8` postselection-surviving
syndromes, supplied in the same order to CPU-only and every hybrid
configuration. Workload generation and decoding operate in bounded chunks
instead of materializing the complete sample, sparse-shot list, and prediction
arrays in memory. Cycle-level simulation uses a stratified trace sample selected
by syndrome weight and observed A* batch/candidate work; it does not replay all
`10^8` syndromes.

The production sweep uses the bit-packed workload and explicitly fixes the I/O
chunk size in the recorded configuration:

```bash
TESSERACT_IO_CHUNK_SHOTS=65536 \
bash scripts/run_scaleout_extrapolation.sh \
  workloads/msc_grafted_d3_d7_100m/model.dem \
  workloads/msc_grafted_d3_d7_100m/syndromes.b8 \
  workloads/msc_grafted_d3_d7_100m/observables.b8 \
  results/scaleout_extrapolation_100m \
  100000000 4,8,16,32,64 16 8,16 2560
```

The SDK was built against Python 3.10 while current WSL distributions may use
Python 3.12. `setup_upmem_host_runtime.sh` downloads the pinned Canonical
`libpython3.10` package, verifies its SHA-256, and extracts it under the build
directory without installing anything system-wide.

For a batch-size sweep:

```bash
bash scripts/sweep_async_batches.sh \
  traces/official_msc_grafted_d3_d7.json \
  results/async_batch_sweep 3 16 8,16,32,64,128
```

For a tasklet sweep, use a batch of at least as many candidates as active
tasklets so the comparison does not leave tasklets idle by construction:

```bash
bash scripts/sweep_tasklets.sh \
  traces/official_msc_grafted_d3_d7.json \
  results/tasklet_sweep 16 3 11,16,24
```

An UPMEM DPU exposes up to 24 hardware tasklets and has an 11-stage pipeline.
Tasklet count changes only the workers active inside one DPU; this runner does
not require DPU-to-DPU communication. With the exact lower-bound early-exit
kernel, the deterministic 137-candidate sweep measured 6,887,827, 6,685,692,
and 6,672,971 logic cycles for 11, 16, and 24 tasklets. The corresponding
16-shot live medians were 30.75, 29.80, and 30.34 seconds. Therefore the
full-system default is 16 tasklets per DPU; 24 is not retained because its
0.19% cycle advantage over 16 did not survive the end-to-end measurement. The
two replay slots process independent batches and communicate only through the
host.

The asynchronous runner remains a replay microbenchmark whose candidate/s is
not full decoder throughput. The separate live comparison includes packing,
H2D, launch, D2H, and priority-queue integration in decoder and process wall
time. Bundled-SDK functional-simulator timing is a correctness/engineering
measurement, not a performance result.

`build_tesseract.sh` limits the official CMake build to one job. Set
`TESSERACT_BOOST_INCLUDE` only when a local Boost include tree must be used.

Record the corresponding CPU-only baseline with tracing forcibly disabled:

```bash
bash scripts/run_cpu_baseline.sh \
  ../../third_party/tesseract-decoder/testdata/colorcodes/r=1,d=3,p=0.2,noise=data_qubit_X,c=color_code_mpp,q=7,gates=mpp.stim \
  1000 results/cpu_color_d3_seed2 2
```

The summary reports both Tesseract's measured decoder-only time and process
wall time. The latter includes one-time circuit loading and Stim sampling, so
both values are retained instead of silently mixing their measurement scopes.

The `tesseract-dpu-trace-v1` JSON contract contains:

- `q_bits`, `detector_penalty_q`, and `num_detectors`
- error-to-detector CSR in `error_detector_offsets` and `error_detector_ids`
- one fixed-point likelihood value per error in `error_cost_q`
- candidate residual detector CSR in `candidate_detector_offsets` and
  `candidate_detector_ids`
- optional candidate-major 64-bit `candidate_blocked_error_words`
- optional producer-side fixed-point reference values in `candidate_cost_q`

The adapter validates offsets and indices, derives detector-to-error CSR and
the division-free overlap LUT, and computes the CPU oracle before launching the
DPU simulation.

If `candidate_cost_q` is present, the adapter first checks the Go oracle against
the official producer. The smoke run then succeeds only when uPIMulator's DPU
output matches that validated oracle byte-for-byte. Simulation statistics are
written under `uPIMulator/results/`.

## Experiment progression

1. Acquire and characterize the MSC grafted-code DEM.
2. Instrument CPU-only Tesseract decode throughput and batch-size distribution.
3. Replay identical MSC batches in uPIMulator and validate every result.
4. Live sibling batching and A* priority-queue feedback (implemented).
5. Multi-syndrome scheduling with separate CPU-worker and DPU-inflight limits
   (implemented; small functional-simulator validation completed).
6. Bounded Stim sampling/postselection, `b8` output, and bounded Tesseract input
   decoding are implemented; generate the production `10^8`-survivor workload.
7. Measure 4/8/16/32/64 DPUs and extrapolate the 2,560-DPU full-system
   throughput for both 8- and 16-CPU configurations. Compare it with the
   matching CPU-only full-system throughput, never per-core performance.
8. Collect cycle-level results from a statistically representative, stratified
   trace sample instead of the single shot-50 fixture.
9. Sweep batch size and active tasklets, reporting p50/p90/p99/p99.9 latency.

Synthetic smoke-test numbers are engineering checks, not research results.
