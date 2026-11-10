#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 11 ]]; then
    echo "usage: $0 MODEL_DEM SYNDROMES_01 OBSERVABLES_01 RESULT_DIR [SHOT_BEGIN] [NUM_SHOTS] [TASKLETS] [Q_BITS] [CPU_THREADS] [DPU_INFLIGHT] [IO_CHUNK_SHOTS]" >&2
    exit 2
fi

RUN_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_PROJECT_ROOT="$(cd "${RUN_SCRIPT_DIR}/.." && pwd)"
RUN_REPOSITORY_ROOT="$(cd "${RUN_PROJECT_ROOT}/../.." && pwd)"
RUN_TESSERACT_ROOT="${TESSERACT_ROOT:-${RUN_REPOSITORY_ROOT}/third_party/tesseract-decoder}"
RUN_TESSERACT_BIN="${TESSERACT_BIN:-${RUN_TESSERACT_ROOT}/cmake-build/tesseract}"
RUN_DEM_PATH="$(realpath "$1")"
RUN_SYNDROME_PATH="$(realpath "$2")"
RUN_OBSERVABLE_PATH="$(realpath "$3")"
RUN_RESULT_DIR="$(realpath -m "$4")"
RUN_SHOT_BEGIN="${5:-0}"
RUN_NUM_SHOTS="${6:-1}"
RUN_TASKLETS="${7:-16}"
RUN_Q_BITS="${8:-16}"
RUN_CPU_THREADS="${9:-1}"
RUN_DPU_INFLIGHT="${10:-${RUN_CPU_THREADS}}"
RUN_IO_CHUNK_SHOTS="${11:-${TESSERACT_IO_CHUNK_SHOTS:-65536}}"
RUN_SHOT_FORMAT="${TESSERACT_SHOT_FORMAT:-}"

for path in "${RUN_DEM_PATH}" "${RUN_SYNDROME_PATH}" "${RUN_OBSERVABLE_PATH}"; do
    if [[ ! -f "${path}" ]]; then
        echo "MSC workload input not found: ${path}" >&2
        exit 1
    fi
done
if [[ -z "${RUN_SHOT_FORMAT}" ]]; then
    if [[ "${RUN_SYNDROME_PATH}" == *.b8 && "${RUN_OBSERVABLE_PATH}" == *.b8 ]]; then
        RUN_SHOT_FORMAT=b8
    elif [[ "${RUN_SYNDROME_PATH}" == *.01 && "${RUN_OBSERVABLE_PATH}" == *.01 ]]; then
        RUN_SHOT_FORMAT=01
    else
        echo "cannot infer matching shot format from input extensions; set TESSERACT_SHOT_FORMAT" >&2
        exit 2
    fi
fi
for value in "${RUN_SHOT_BEGIN}" "${RUN_NUM_SHOTS}" "${RUN_TASKLETS}" "${RUN_Q_BITS}" "${RUN_CPU_THREADS}" "${RUN_DPU_INFLIGHT}" "${RUN_IO_CHUNK_SHOTS}"; do
    if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
        echo "shot begin, shot count, tasklets, and Q bits must be non-negative integers" >&2
        exit 2
    fi
done
if (( RUN_NUM_SHOTS == 0 || RUN_TASKLETS == 0 || RUN_CPU_THREADS == 0 || RUN_DPU_INFLIGHT == 0 || RUN_Q_BITS > 24 )); then
    echo "shot count/tasklets/CPU threads/in-flight DPUs must be positive and Q bits must be in [0,24]" >&2
    exit 2
fi
RUN_SHOT_END="$((RUN_SHOT_BEGIN + RUN_NUM_SHOTS))"
RUN_ACTIVE_DPUS="${RUN_DPU_INFLIGHT}"
if (( RUN_ACTIVE_DPUS > RUN_NUM_SHOTS )); then
    RUN_ACTIVE_DPUS="${RUN_NUM_SHOTS}"
fi
if (( RUN_ACTIVE_DPUS > RUN_IO_CHUNK_SHOTS )); then
    RUN_ACTIVE_DPUS="${RUN_IO_CHUNK_SHOTS}"
fi
if [[ ! -x "${RUN_TESSERACT_BIN}" ]]; then
    "${RUN_SCRIPT_DIR}/build_tesseract.sh"
fi
"${RUN_SCRIPT_DIR}/build_async_host.sh" "${RUN_TASKLETS}"
"${RUN_SCRIPT_DIR}/setup_upmem_host_runtime.sh"
RUN_UPMEM_BACKEND="${TESSERACT_UPMEM_BACKEND:-simulator}"
source "${RUN_PROJECT_ROOT}/upmem_sdk/upmem_env.sh" "${RUN_UPMEM_BACKEND}"
if [[ -n "${TESSERACT_UPMEM_LIBRARY_PATH:-}" ]]; then
    export LD_LIBRARY_PATH="${TESSERACT_UPMEM_LIBRARY_PATH}:${LD_LIBRARY_PATH}"
fi

RUN_BUILD_ROOT="${RUN_PROJECT_ROOT}/uPIMulator/benchmark/build/TESSERACT/async_host"
export LD_LIBRARY_PATH="${RUN_BUILD_ROOT}/runtime/python3.10/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"
RUN_DPU_BINARY="${RUN_BUILD_ROOT}/tesseract_dpu_t${RUN_TASKLETS}"
RUN_LIVE_LIBRARY="${RUN_BUILD_ROOT}/libtesseract_dpu_live.so"
RUN_PRELOAD_DIR="$(mktemp -d)"
trap 'rm -f -- "${RUN_PRELOAD_DIR}/libtesseract_dpu_live.so"; rmdir -- "${RUN_PRELOAD_DIR}"' EXIT
ln -s "${RUN_LIVE_LIBRARY}" "${RUN_PRELOAD_DIR}/libtesseract_dpu_live.so"
RUN_PRELOAD_LIBRARY="${RUN_PRELOAD_DIR}/libtesseract_dpu_live.so"
RUN_CPU_DIR="${RUN_RESULT_DIR}/cpu"
RUN_LIVE_DIR="${RUN_RESULT_DIR}/dpu_live"
mkdir -p "${RUN_CPU_DIR}" "${RUN_LIVE_DIR}"

unset TESSERACT_DPU_LIVE TESSERACT_DPU_BINARY TESSERACT_DPU_LIVE_Q_BITS
unset TESSERACT_DPU_LIVE_STATS TESSERACT_DPU_TRACE_PATH TESSERACT_DPU_TRACE_Q_BITS
if [[ ! -x /usr/bin/time ]]; then
    echo "GNU /usr/bin/time is required" >&2
    exit 1
fi
/usr/bin/time -f '%e' -o "${RUN_CPU_DIR}/wall_seconds.txt" \
    "${RUN_TESSERACT_BIN}" \
    --dem "${RUN_DEM_PATH}" \
    --in "${RUN_SYNDROME_PATH}" \
    --in-format "${RUN_SHOT_FORMAT}" \
    --obs-in "${RUN_OBSERVABLE_PATH}" \
    --obs-in-format "${RUN_SHOT_FORMAT}" \
    --shot-range-begin "${RUN_SHOT_BEGIN}" \
    --shot-range-end "${RUN_SHOT_END}" \
    --io-chunk-shots "${RUN_IO_CHUNK_SHOTS}" \
    --threads "${RUN_CPU_THREADS}" \
    --out "${RUN_CPU_DIR}/predictions.01" \
    --out-format 01 \
    --stats-out "${RUN_CPU_DIR}/stats.json"

/usr/bin/time -f '%e' -o "${RUN_LIVE_DIR}/wall_seconds.txt" \
    env \
    TESSERACT_DPU_LIVE=1 \
    TESSERACT_DPU_BINARY="${RUN_DPU_BINARY}" \
    TESSERACT_DPU_LIVE_Q_BITS="${RUN_Q_BITS}" \
    TESSERACT_DPU_LIVE_STATS=1 \
    LD_PRELOAD="${RUN_PRELOAD_LIBRARY}" \
    "${RUN_TESSERACT_BIN}" \
    --dem "${RUN_DEM_PATH}" \
    --in "${RUN_SYNDROME_PATH}" \
    --in-format "${RUN_SHOT_FORMAT}" \
    --obs-in "${RUN_OBSERVABLE_PATH}" \
    --obs-in-format "${RUN_SHOT_FORMAT}" \
    --shot-range-begin "${RUN_SHOT_BEGIN}" \
    --shot-range-end "${RUN_SHOT_END}" \
    --io-chunk-shots "${RUN_IO_CHUNK_SHOTS}" \
    --threads "${RUN_CPU_THREADS}" \
    --inflight-shots "${RUN_DPU_INFLIGHT}" \
    --out "${RUN_LIVE_DIR}/predictions.01" \
    --out-format 01 \
    --stats-out "${RUN_LIVE_DIR}/stats.json" \
    2> "${RUN_LIVE_DIR}/dpu_live.log"

if cmp -s "${RUN_CPU_DIR}/predictions.01" "${RUN_LIVE_DIR}/predictions.01"; then
    RUN_PREDICTIONS_MATCH=true
else
    RUN_PREDICTIONS_MATCH=false
fi

RUN_CPU_SECONDS="$(sed -n 's/.*"total_time_seconds":\([0-9.eE+-]*\).*/\1/p' "${RUN_CPU_DIR}/stats.json")"
RUN_LIVE_SECONDS="$(sed -n 's/.*"total_time_seconds":\([0-9.eE+-]*\).*/\1/p' "${RUN_LIVE_DIR}/stats.json")"
if [[ -z "${RUN_CPU_SECONDS}" || -z "${RUN_LIVE_SECONDS}" ]]; then
    echo "failed to parse decoder timing JSON" >&2
    exit 1
fi
RUN_CPU_WALL="$(<"${RUN_CPU_DIR}/wall_seconds.txt")"
RUN_LIVE_WALL="$(<"${RUN_LIVE_DIR}/wall_seconds.txt")"
RUN_CPU_RATE="$(awk -v shots="${RUN_NUM_SHOTS}" -v seconds="${RUN_CPU_SECONDS}" 'BEGIN { if (seconds == 0) print "inf"; else printf "%.6f", shots/seconds }')"
RUN_LIVE_RATE="$(awk -v shots="${RUN_NUM_SHOTS}" -v seconds="${RUN_LIVE_SECONDS}" 'BEGIN { if (seconds == 0) print "inf"; else printf "%.6f", shots/seconds }')"
read -r RUN_DPU_BATCHES RUN_DPU_CANDIDATES <<< "$(awk '
    /Tesseract DPU live:/ {
        for (i = 1; i <= NF; ++i) {
            split($i, field, "=")
            if (field[1] == "batches") batches += field[2]
            if (field[1] == "candidates") candidates += field[2]
        }
    }
    END { print batches + 0, candidates + 0 }
' "${RUN_LIVE_DIR}/dpu_live.log")"

{
    echo "mode,shots,shot_begin,cpu_threads,configured_dpu_pool,active_dpus,tasklets_per_dpu,q_bits,summed_decoder_seconds,per_cpu_service_rate_per_second,process_wall_seconds,system_throughput_per_second,dpu_batches,dpu_candidates,predictions_match_cpu,io_chunk_shots,shot_format"
    echo "cpu_only,${RUN_NUM_SHOTS},${RUN_SHOT_BEGIN},${RUN_CPU_THREADS},0,0,0,NA,${RUN_CPU_SECONDS},${RUN_CPU_RATE},${RUN_CPU_WALL},$(awk -v shots="${RUN_NUM_SHOTS}" -v seconds="${RUN_CPU_WALL}" 'BEGIN { if (seconds == 0) print "inf"; else printf "%.6f", shots/seconds }'),0,0,true,${RUN_IO_CHUNK_SHOTS},${RUN_SHOT_FORMAT}"
    echo "dpu_live,${RUN_NUM_SHOTS},${RUN_SHOT_BEGIN},${RUN_CPU_THREADS},${RUN_DPU_INFLIGHT},${RUN_ACTIVE_DPUS},${RUN_TASKLETS},${RUN_Q_BITS},${RUN_LIVE_SECONDS},${RUN_LIVE_RATE},${RUN_LIVE_WALL},$(awk -v shots="${RUN_NUM_SHOTS}" -v seconds="${RUN_LIVE_WALL}" 'BEGIN { if (seconds == 0) print "inf"; else printf "%.6f", shots/seconds }'),${RUN_DPU_BATCHES},${RUN_DPU_CANDIDATES},${RUN_PREDICTIONS_MATCH},${RUN_IO_CHUNK_SHOTS},${RUN_SHOT_FORMAT}"
} > "${RUN_RESULT_DIR}/comparison.csv"

if [[ "${RUN_PREDICTIONS_MATCH}" != true ]]; then
    echo "DPU live predictions differ from CPU: ${RUN_RESULT_DIR}/comparison.csv" >&2
    exit 1
fi
echo "MSC CPU vs live-DPU comparison passed: ${RUN_RESULT_DIR}/comparison.csv"
