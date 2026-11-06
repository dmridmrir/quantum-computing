# BP layered decoder v1/v2 구분 (2026-07-27 갱신)

현재 v1과 v2는 단순 최적화 버전 관계가 아니라 서로 다른 tasklet mapping을
측정한다.

| 구분 | v1 | v2 |
|---|---|---|
| DPU 커널 | `task_layered.c` | `task_layered_v2.c` |
| mapping | 11 tasklets가 syndrome 1개를 협력 처리 | 11 tasklets가 syndrome 1개를 협력 처리 |
| 알고리즘 | three-color cooperative layered, v1 수식 | BP-PIM three-color cooperative layered Min-Sum |
| iteration | 기존 스크립트 설정 | 1~10 sweep |
| alpha / damping | v1 구현 설정 | alpha=1, damping=1/2 |
| 동시 syndrome/DPU | 1개 | 1개 |

v1 cooperative 변경에서도 v1의 int32 C2V/lambda, q 비포화, alpha=0.75,
damping 없음은 유지한다. 11개 tasklet이 shared posterior를 안전하게 갱신하도록
v2와 동일한 color order 1-2-0 및 color barrier를 적용한다. 따라서 이전 check
0~71 순차 v1과 schedule은 달라지지만, v1/v2 간 tasklet mapping은 동일해진다.

## v2 재현 기준

`uPIMulator/benchmark/BPMS/dpu/task_layered_v2.c`는 BP-PIM의
`task_colored_coop.c`와 byte 단위로 동일하다. 필요한
`bb144_colors.h`, 고정 호스트 입력 생성기, CPU golden model도 현재 프로젝트에
포함되어 있으므로 EC2에 `BP-PIM/` 디렉터리를 함께 올릴 필요가 없다.

v2의 고정 검증점은 다음과 같다.

- 11 total tasklets, `BP_GROUP_SIZE=11`
- syndrome 1개/DPU
- iter=1 through 10 sweep
- BSC p=0.001, 0.003, 0.005, 0.007, 0.009
- p별 LLR prior와 기본 15개 입력 seed
- color order 1-2-0
- alpha=1, half damping
- CPU-DPU posterior byte-exact 일치 필수

## 실행파일

| 목적 | v1 | v2 |
|---|---|---|
| throughput | `run_bp_increase_layered_phase1/2.sh` | `run_bp_increase_layered_v2_phase1/2.sh` |
| latency | `bp_tail_layered_phase1/2.sh` | `bp_tail_layered_v2_phase1/2.sh` |

v2의 실제 컴파일·검증·측정 로직은 `bp_tail_layered_v2_phase1.sh`에 들어 있다.
throughput phase1도 이 파일에 결과 경로를 전달해 같은 측정점을 재사용한다. 이
실행기는 BP-PIM 경로를 참조하지 않으며, 원본 커널/색상 테이블 해시가 고정값과
다르면 측정을 거부한다.

## latency 해석

phase2는 기존 tail 정의와 동일하게 CSS component logic latency를 두 배하여 X+Z
직렬 전체 latency를 보고한다. `< 1 ms` 판정은 각 p에서 표본 X+Z latency의 max를
기준으로 한다. host 전송과 launch 시간은 포함되지 않는다.
