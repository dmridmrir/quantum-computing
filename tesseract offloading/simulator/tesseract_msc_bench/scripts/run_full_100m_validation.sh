#!/usr/bin/env bash
# Resumable, single-core validation of the complete accepted-syndrome stream.
#
# For each byte-aligned chunk:
#   1. extract the exact syndrome/observable byte range,
#   2. run the DPU-resident C++ oracle,
#   3. run CPU-only Tesseract with the same DEM and inputs,
#   4. compare every final solution cost against the CPU result. Different
#      observable predictions are accepted only when the cost difference stays
#      within the accumulated Q16 rounding bound; all such ties are counted.
#
# Environment overrides:
#   TESSERACT_VALIDATION_TOTAL_SHOTS (default 100000000)
#   TESSERACT_VALIDATION_CHUNK_SHOTS (default 1000000)
#   TESSERACT_VALIDATION_OUT
#   TESSERACT_BIN
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPOSITORY_ROOT="$(cd "${PROJECT_ROOT}/../.." && pwd)"
WORKLOAD="${PROJECT_ROOT}/workloads/msc_grafted_d3_d7_100m"
TESSERACT_BIN="${TESSERACT_BIN:-${REPOSITORY_ROOT}/third_party/tesseract-decoder/cmake-build/tesseract}"
TOTAL_SHOTS="${TESSERACT_VALIDATION_TOTAL_SHOTS:-100000000}"
CHUNK_SHOTS="${TESSERACT_VALIDATION_CHUNK_SHOTS:-1000000}"
OUT="${TESSERACT_VALIDATION_OUT:-${PROJECT_ROOT}/results/full_validation_100m}"

DEM="${WORKLOAD}/model.dem"
SYNDROMES="${WORKLOAD}/syndromes.b8"
OBSERVABLES="${WORKLOAD}/observables.b8"
SYNDROME_BYTES_PER_SHOT=62
OBSERVABLE_BYTES_PER_SHOT=1

for required in "${TESSERACT_BIN}" "${DEM}" "${SYNDROMES}" "${OBSERVABLES}"; do
    if [[ ! -f "${required}" ]]; then
        echo "missing required input: ${required}" >&2
        exit 1
    fi
done
if (( TOTAL_SHOTS <= 0 || CHUNK_SHOTS <= 0 )); then
    echo "shot counts must be positive" >&2
    exit 1
fi
if (( $(stat -c %s "${SYNDROMES}") < TOTAL_SHOTS * SYNDROME_BYTES_PER_SHOT )); then
    echo "syndrome file is shorter than TOTAL_SHOTS" >&2
    exit 1
fi
if (( $(stat -c %s "${OBSERVABLES}") < TOTAL_SHOTS * OBSERVABLE_BYTES_PER_SHOT )); then
    echo "observable file is shorter than TOTAL_SHOTS" >&2
    exit 1
fi

mkdir -p "${OUT}/chunks" "${OUT}/tmp"
exec 9>"${OUT}/run.lock"
if ! flock -n 9; then
    echo "another full validation process already holds ${OUT}/run.lock" >&2
    exit 1
fi

cleanup() {
    rm -f -- "${OUT}/tmp/syndromes.b8" "${OUT}/tmp/observables.b8"
}
trap cleanup EXIT INT TERM

echo "full validation start: total=${TOTAL_SHOTS} chunk=${CHUNK_SHOTS}"
echo "tesseract=${TESSERACT_BIN}"
echo "output=${OUT}"

for ((begin = 0; begin < TOTAL_SHOTS; begin += CHUNK_SHOTS)); do
    end=$((begin + CHUNK_SHOTS))
    if (( end > TOTAL_SHOTS )); then
        end=${TOTAL_SHOTS}
    fi
    count=$((end - begin))
    tag="$(printf '%09d-%09d' "${begin}" "${end}")"
    chunk_dir="${OUT}/chunks/${tag}"
    if [[ -f "${chunk_dir}/complete" ]]; then
        echo "skip complete chunk ${tag}"
        continue
    fi

    mkdir -p "${chunk_dir}/oracle" "${chunk_dir}/cpu"
    echo "chunk ${tag}: extracting ${count} shots"
    dd if="${SYNDROMES}" of="${OUT}/tmp/syndromes.b8" bs=4M \
        iflag=skip_bytes,count_bytes skip=$((begin * SYNDROME_BYTES_PER_SHOT)) \
        count=$((count * SYNDROME_BYTES_PER_SHOT)) status=none
    dd if="${OBSERVABLES}" of="${OUT}/tmp/observables.b8" bs=4M \
        iflag=skip_bytes,count_bytes skip=$((begin * OBSERVABLE_BYTES_PER_SHOT)) \
        count=$((count * OBSERVABLE_BYTES_PER_SHOT)) status=none

    if [[ ! -s "${chunk_dir}/oracle/stats.json" ]]; then
        rm -f -- "${chunk_dir}/oracle/predictions.b8" \
            "${chunk_dir}/oracle/search_stats.csv" "${chunk_dir}/oracle/run.log"
        echo "chunk ${tag}: resident oracle"
        env \
            TESSERACT_DPU_RESIDENT=1 \
            TESSERACT_DPU_RESIDENT_ORACLE=1 \
            TESSERACT_DPU_RESIDENT_CAPACITY=1000000 \
            TESSERACT_DPU_LIVE_Q_BITS=16 \
            TESSERACT_DPU_LIVE_STATS=1 \
            TESSERACT_SEARCH_STATS_PATH="${chunk_dir}/oracle/search_stats.csv" \
            "${TESSERACT_BIN}" \
            --dem "${DEM}" \
            --in "${OUT}/tmp/syndromes.b8" --in-format b8 \
            --obs-in "${OUT}/tmp/observables.b8" --obs-in-format b8 \
            --io-chunk-shots 65536 \
            --threads 1 --inflight-shots 1 \
            --out "${chunk_dir}/oracle/predictions.b8" --out-format b8 \
            --stats-out "${chunk_dir}/oracle/stats.json" \
            >"${chunk_dir}/oracle/run.log" 2>&1
    fi

    if [[ ! -s "${chunk_dir}/cpu/stats.json" ]]; then
        rm -f -- "${chunk_dir}/cpu/predictions.b8" \
            "${chunk_dir}/cpu/search_stats.csv" "${chunk_dir}/cpu/run.log"
        echo "chunk ${tag}: CPU-only"
        env -u TESSERACT_DPU_RESIDENT \
            -u TESSERACT_DPU_RESIDENT_ORACLE \
            -u TESSERACT_DPU_RESIDENT_CAPACITY \
            -u TESSERACT_DPU_RESIDENT_NO_RESUME \
            TESSERACT_SEARCH_STATS_PATH="${chunk_dir}/cpu/search_stats.csv" \
            "${TESSERACT_BIN}" \
            --dem "${DEM}" \
            --in "${OUT}/tmp/syndromes.b8" --in-format b8 \
            --obs-in "${OUT}/tmp/observables.b8" --obs-in-format b8 \
            --io-chunk-shots 65536 \
            --threads 1 --inflight-shots 1 \
            --out "${chunk_dir}/cpu/predictions.b8" --out-format b8 \
            --stats-out "${chunk_dir}/cpu/stats.json" \
            >"${chunk_dir}/cpu/run.log" 2>&1
    fi

    python3 - "${begin}" "${count}" \
        "${chunk_dir}/oracle/stats.json" "${chunk_dir}/cpu/stats.json" \
        "${chunk_dir}/oracle/search_stats.csv" "${chunk_dir}/cpu/search_stats.csv" \
        "${chunk_dir}/oracle/predictions.b8" "${chunk_dir}/cpu/predictions.b8" \
        "${chunk_dir}/comparison.json" <<'PY'
import csv
import json
import math
import sys

begin = int(sys.argv[1])
expected = int(sys.argv[2])
with open(sys.argv[3]) as f:
    oracle = json.load(f)
with open(sys.argv[4]) as f:
    cpu = json.load(f)
if oracle["num_shots"] != expected or cpu["num_shots"] != expected:
    raise SystemExit(
        f"shot count mismatch: expected={expected} oracle={oracle['num_shots']} cpu={cpu['num_shots']}"
    )

def final_oracle_rows(path):
    pending_resume = None
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["kind"] != "resident_dpu":
                if pending_resume is not None:
                    raise RuntimeError("multiple CPU resume rows before one resident row")
                pending_resume = row
                continue
            status = int(row["status"])
            if status == 2:
                if pending_resume is None:
                    raise RuntimeError("resident fallback has no preceding CPU resume row")
                yield pending_resume, row
                pending_resume = None
            else:
                if pending_resume is not None:
                    raise RuntimeError("CPU resume row precedes a non-fallback resident row")
                yield row, row
    if pending_resume is not None:
        raise RuntimeError("trailing CPU resume row has no resident row")

mismatch_count = 0
quantization_bound_violations = 0
nonzero_cost_deltas = 0
max_abs_cost_delta = 0.0
max_cost_delta_over_bound = 0.0
first_mismatches = []
first_cost_deviations = []
oracle_rows = final_oracle_rows(sys.argv[5])
with open(sys.argv[6], newline="") as cpu_file, open(sys.argv[7], "rb") as oracle_pred, \
        open(sys.argv[8], "rb") as cpu_pred:
    cpu_rows = csv.DictReader(cpu_file)
    row_count = 0
    for row_count, (cpu_row, (oracle_final, oracle_attempt)) in enumerate(
            zip(cpu_rows, oracle_rows, strict=True), start=1):
        cpu_cost = float(cpu_row["solution_cost"])
        oracle_cost = float(oracle_final["solution_cost"])
        if not (math.isfinite(cpu_cost) and math.isfinite(oracle_cost)):
            raise RuntimeError(f"non-finite solution cost at shot {begin + row_count - 1}")
        delta = abs(cpu_cost - oracle_cost)
        max_abs_cost_delta = max(max_abs_cost_delta, delta)
        cpu_depth = int(cpu_row["solution_depth"])
        oracle_depth = int(oracle_final["solution_depth"])
        quantization_bound = (cpu_depth + oracle_depth) * 0.5 / (1 << 16) + 1e-12
        ratio = delta / quantization_bound if quantization_bound else 0.0
        max_cost_delta_over_bound = max(max_cost_delta_over_bound, ratio)
        within_bound = delta <= quantization_bound
        if delta > 1e-12:
            nonzero_cost_deltas += 1
            if len(first_cost_deviations) < 1000:
                first_cost_deviations.append({
                    "global_shot": begin + row_count - 1,
                    "cpu_solution_cost": cpu_cost,
                    "oracle_solution_cost": oracle_cost,
                    "cpu_solution_depth": cpu_depth,
                    "oracle_solution_depth": oracle_depth,
                    "absolute_cost_delta": delta,
                    "q16_quantization_bound": quantization_bound,
                    "delta_over_bound": ratio,
                })
        if not within_bound:
            quantization_bound_violations += 1
        oracle_byte = oracle_pred.read(1)
        cpu_byte = cpu_pred.read(1)
        if len(oracle_byte) != 1 or len(cpu_byte) != 1:
            raise RuntimeError("prediction file ended before the per-shot stats")
        if oracle_byte != cpu_byte:
            mismatch_count += 1
            if len(first_mismatches) < 1000:
                first_mismatches.append({
                    "global_shot": begin + row_count - 1,
                    "cpu_prediction": cpu_byte[0],
                    "oracle_prediction": oracle_byte[0],
                    "cpu_solution_cost": cpu_cost,
                    "oracle_solution_cost": oracle_cost,
                    "absolute_cost_delta": delta,
                    "q16_quantization_bound": quantization_bound,
                    "delta_over_bound": ratio,
                    "resident_status": int(oracle_attempt["status"]),
                    "fallback_reason": oracle_attempt["fallback_reason"],
                })
    if row_count != expected:
        raise RuntimeError(f"expected {expected} compared rows, got {row_count}")
    if oracle_pred.read(1) or cpu_pred.read(1):
        raise RuntimeError("prediction file contains more rows than the per-shot stats")

comparison = {
    "shot_begin": begin,
    "shots": expected,
    "prediction_mismatches": mismatch_count,
    "prediction_mismatch_rate": mismatch_count / expected,
    "all_solution_cost_deltas_within_q16_bound": quantization_bound_violations == 0,
    "quantization_bound_violations": quantization_bound_violations,
    "nonzero_solution_cost_deltas": nonzero_cost_deltas,
    "max_absolute_solution_cost_delta": max_abs_cost_delta,
    "max_solution_cost_delta_over_q16_bound": max_cost_delta_over_bound,
    "cpu_logical_errors": cpu["num_errors"],
    "oracle_logical_errors": oracle["num_errors"],
    "logical_error_delta": oracle["num_errors"] - cpu["num_errors"],
    "first_prediction_mismatches": first_mismatches,
    "first_solution_cost_deviations": first_cost_deviations,
}
with open(sys.argv[9], "w") as f:
    json.dump(comparison, f, indent=2, sort_keys=True)
    f.write("\n")
if quantization_bound_violations:
    raise SystemExit(
        f"solution cost exceeds the Q16 rounding bound in {quantization_bound_violations} shots"
    )
PY
    sha256sum "${chunk_dir}/oracle/predictions.b8" \
        "${chunk_dir}/cpu/predictions.b8" >"${chunk_dir}/prediction_sha256.txt"
    touch "${chunk_dir}/complete"
    cleanup
    echo "chunk ${tag}: PASS"
done

touch "${OUT}/complete"
echo "full validation PASS: ${TOTAL_SHOTS} shots"
