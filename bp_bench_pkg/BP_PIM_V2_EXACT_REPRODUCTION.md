# BP-PIM cooperative v2 exact reproduction

v2는 EC2에 `BP-PIM/` 디렉터리를 배포하지 않아도 실행되는 독립 재현본이다.
`*_v2_phase1/2.sh`는 기존 일반 layered 포트가 아니라 11개 tasklet이 syndrome
하나를 협력 처리하는 BP-PIM colored cooperative 커널을 실행한다.

## 프로젝트에 포함된 원본 구성요소

- DPU 커널: `uPIMulator/benchmark/BPMS/dpu/task_layered_v2.c`
  - BP-PIM `task_colored_coop.c`의 byte-exact 복사본
- 색상 스케줄: `uPIMulator/benchmark/BPMS/support/bb144_colors.h`
  - BP-PIM 원본의 byte-exact 복사본
- 고정 호스트 입력: `uPIMulator/src/assembler/prim/bpms_colored_coop_host.go.src`
- CPU golden model: `uPIMulator/src/assembler/prim/bpms_layered_v2_verify.go`
- 자체 실행기: `bp_tail_layered_v2_phase1.sh`

tail phase1은 기존 `bpms.go`를 고정 호스트 입력 구현으로 잠시 교체하고 종료 시 원상
복구한다. DPU assembly도 종료 시 기본 `task.c`로 복구한다. 따라서 실행 중에도
`BP-PIM/` 아래 파일을 읽거나 복사하지 않는다.

## 고정 구성

- total tasklets: 11
- tasklets per syndrome (`BP_GROUP_SIZE`): 11
- concurrent syndromes per DPU: 1
- iterations: 1 through 10 sweep
- error rates: BSC p=0.001, 0.003, 0.005, 0.007, 0.009
- prior: 각 p에서 `int(log((1-p)/p)*256)`
- repetitions: 각 (p, iter) 조합마다 기본 15회
- seed base: 기본 42 (`SHARED_SEED`로 변경 가능)
- schedule: three-color horizontal layered, order 1-2-0
- normalization: alpha=1
- damping: 1/2
- modeled clock: 350 MHz
- correctness gate: CPU와 DPU posterior의 byte-exact 비교

커널과 색상 테이블의 SHA-256이 아래 값과 다르면 phase1은 실행을 중단한다.

```text
fa6170defe31d3bb9b40190f7cce80d1e98c1aba53a7b68b637b3ad90a5adaad  task_layered_v2.c
f8549d04585c4842265c256e6ae5dfc796e8355d9c77eedff222d69242cc808a  bb144_colors.h
```

## EC2 실행

Latency 재현:

```bash
/bin/bash bp_tail_layered_v2_phase1.sh
/bin/bash bp_tail_layered_v2_phase2.sh
```

Throughput 및 DPU 수 기준 projection:

```bash
/bin/bash run_bp_increase_layered_v2_phase1.sh
/bin/bash run_bp_increase_layered_v2_phase2.sh
```

increase phase2는 v1과 같은 구성의 `Throughput + LER Summary`와 Wilson 95% CI
상세표를 출력한다. CPU 열은 cooperative colored v2의 정확한 순차 대응 구현을
측정한다. v2는 11개 tasklet이 syndrome 하나를 처리하므로 1 DPU throughput은
동시 decoder 1개로 계산하며, DPU cycle 열은 phase1의 p별 반복 측정 평균이다.

Throughput phase1은 tail phase1에 결과 경로만 별도로 전달한다. 따라서 두 경로
모두 같은 `11 tasklets / 1 syndrome / iter=1..10 / p=0.001..0.009` sweep을 사용한다.

## 측정 범위

phase1 CSV의 latency는 CSS component 하나의 simulator logic latency다. phase2는
기존 tail 측정 정의와 동일하게 이를 두 배하여 직렬 X+Z 전체 latency의 mean, std,
p50, p90, p99, p99.9, p99.99, max를 보고한다. `<1 ms` 판정은 X+Z max를 기준으로
한다. DPU 커널 실행, tasklet scheduling/barrier, 커널 내부 MRAM read/write가
포함되고 host-DPU 전송, runtime launch, program load, controller integration은
포함되지 않는다.
