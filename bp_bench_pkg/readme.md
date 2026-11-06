# BP-PIM: Offloading Quantum BP Decoding to UPMEM PIM

## Overview

This project implements a quantum error correction (QEC) Belief Propagation (BP) decoder on UPMEM Processing-in-Memory (PIM) using Data Processing Units (DPUs).

The goal is to improve decoding throughput by exploiting DPU-level parallelism when processing multiple syndromes. Each DPU uses **11 cooperating tasklets to decode a single syndrome**, allowing multiple syndromes to be processed in parallel across available DPUs.

The implementation supports two evaluation modes:

1. **Throughput Test**: Evaluates decoding throughput when processing multiple syndromes across DPUs.
2. **Tail Latency Test**: Evaluates the tail latency of individual syndrome decoding operations.

All experiments are conducted using the uPIMulator rather than physical UPMEM hardware.

## Third-Party Software

This project uses **uPIMulator**, a cycle-level performance simulator for UPMEM PIM architectures.

- **GitHub Repository:** https://github.com/VIA-Research/uPIMulator
- **Reference Paper:** B. Hyun et al., "Pathfinding Future PIM Architectures by Demystifying a Commercial PIM Technology," *IEEE International Symposium on High-Performance Computer Architecture (HPCA)*, 2024.

The simulator is developed and maintained by its original authors. This project uses it as a third-party framework for evaluating DPU-based quantum BP decoding.

## Prerequisites

### 1. Operating System

Linux or Windows Subsystem for Linux (WSL) is required.

### 2. Install Dependencies

Install the required packages:

```bash
sudo apt update
sudo apt install -y build-essential libgomp1 cmake ninja-build python3 coreutils gawk grep sed
```

### 3. Check Go Installation

Verify that Go 1.22.x is installed:

```bash
go version
```

Expected output:

```text
go version go1.22.x linux/amd64
```

## Getting Started

### Step 1. Open Linux or WSL

Launch a Linux terminal or WSL environment.

### Step 2. Navigate to the Project Directory

```bash
cd /path/to/bp_bench_pkg
```

### Step 3. Run the Setup Script

```bash
bash setup.sh
```

## Running Experiments

### Mode 1. Throughput Test

This mode evaluates decoding throughput when processing multiple syndromes across DPUs.

Run the following scripts sequentially.

**Phase 1**

```bash
bash run_bp_increase_layered_v2_phase1.sh
```

**Phase 2**

```bash
bash run_bp_increase_layered_v2_phase2.sh
```

### Mode 2. Tail Latency Test

This mode evaluates the tail latency of individual syndrome decoding operations.

Run the following scripts sequentially.

**Phase 1**

```bash
bash bp_tail_layered_v2_phase1.sh
```

**Phase 2**

```bash
bash bp_tail_layered_v2_phase2.sh
```

## Notes

- Each DPU employs 11 tasklets cooperatively to decode one syndrome.
- Multiple DPUs enable parallel decoding of independent syndromes.
- Throughput and tail latency are evaluated using separate experiment modes.
- All reported DPU execution measurements are based on the uPIMulator rather than physical UPMEM hardware.

## References

[1] B. Hyun et al., "Pathfinding Future PIM Architectures by Demystifying a Commercial PIM Technology," *2024 IEEE International Symposium on High-Performance Computer Architecture (HPCA)*, 2024.

[2] VIA-Research, "uPIMulator," GitHub. https://github.com/VIA-Research/uPIMulator
