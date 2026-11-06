package prim

import (
	"errors"
	"math/rand"
	"uPIMulator/src/abi/encoding"
	"uPIMulator/src/abi/word"
	"uPIMulator/src/misc"
)

type Uf3d struct {
	num_dpus       int
	num_tasklets   int
	num_executions int

	d     int64
	t_val int64
	nn    int64

	tasklet_slots [][]int64
}

func generateSyndrome3D(d int64, T int64, p float64, rng *rand.Rand) []int64 {
	dd := d * d
	nn := dd * T
	syn := make([]bool, nn)

	for t := int64(0); t < T; t++ {
		for r := int64(0); r < d; r++ {
			for c := int64(0); c < d; c++ {
				rv := rng.Float64()
				i := t*dd + r*d + c
				if rv < p*0.6 {
					syn[i] = !syn[i]
					dir := rng.Intn(6)
					v := i
					switch dir {
					case 0:
						if c+1 < d {
							v = t*dd + r*d + (c + 1)
						}
					case 1:
						if c > 0 {
							v = t*dd + r*d + (c - 1)
						}
					case 2:
						if r+1 < d {
							v = t*dd + (r+1)*d + c
						}
					case 3:
						if r > 0 {
							v = t*dd + (r-1)*d + c
						}
					case 4:
						if t+1 < T {
							v = (t+1)*dd + r*d + c
						}
					case 5:
						if t > 0 {
							v = (t-1)*dd + r*d + c
						}
					}
					syn[v] = !syn[v]
				} else if rv < p {
					syn[i] = !syn[i]
				}
			}
		}
	}

	events := make([]int64, 0)
	for i := int64(0); i < nn; i++ {
		if syn[i] {
			events = append(events, i)
		}
	}
	return events
}

func (this *Uf3d) Init(command_line_parser *misc.CommandLineParser) {
	num_channels := int(command_line_parser.IntParameter("num_channels"))
	num_ranks_per_channel := int(command_line_parser.IntParameter("num_ranks_per_channel"))
	num_dpus_per_rank := int(command_line_parser.IntParameter("num_dpus_per_rank"))

	this.num_dpus = num_channels * num_ranks_per_channel * num_dpus_per_rank
	this.num_tasklets = int(command_line_parser.IntParameter("num_tasklets"))
	this.num_executions = 1

	this.d = int64(command_line_parser.DataPrepParams()[0])
	this.t_val = this.d // 3D: T = d
	this.nn = this.d * this.d * this.t_val

	p := 0.005
	this.tasklet_slots = make([][]int64, this.num_tasklets)
	for t := 0; t < this.num_tasklets; t++ {
		rng := rand.New(rand.NewSource(int64(42 + t*137)))
		events := generateSyndrome3D(this.d, this.t_val, p, rng)

		slot := make([]int64, 64)
		ne := len(events)
		if ne > 63 {
			ne = 63
		}
		slot[0] = int64(ne)
		for i := 0; i < ne; i++ {
			slot[1+i] = events[i]
		}
		this.tasklet_slots[t] = slot
	}
}

func (this *Uf3d) InputDpuHost(execution int, dpu_id int) map[string]*encoding.ByteStream {
	if execution >= this.num_executions {
		panic(errors.New("execution >= num executions"))
	}

	bs := new(encoding.ByteStream)
	bs.Init()

	// dpu_arguments_t: d, T, total_nodes, num_events, size, transfer_size, kernel
	fields := []int64{
		this.d,
		this.t_val,
		this.nn,
		0, 0, 0, 0,
	}
	for _, v := range fields {
		w := new(word.Word)
		w.Init(32)
		w.SetValue(v)
		bs.Merge(w.ToByteStream())
	}

	dpu_host := make(map[string]*encoding.ByteStream)
	dpu_host["DPU_INPUT_ARGUMENTS"] = bs
	return dpu_host
}

func (this *Uf3d) OutputDpuHost(execution int, dpu_id int) map[string]*encoding.ByteStream {
	return make(map[string]*encoding.ByteStream)
}

func (this *Uf3d) InputDpuMramHeapPointerName(execution int, dpu_id int) (int64, *encoding.ByteStream) {
	bs := new(encoding.ByteStream)
	bs.Init()

	for t := 0; t < this.num_tasklets; t++ {
		for i := 0; i < 64; i++ {
			w := new(word.Word)
			w.Init(32)
			w.SetValue(this.tasklet_slots[t][i])
			bs.Merge(w.ToByteStream())
		}
	}

	return 0, bs
}

func (this *Uf3d) OutputDpuMramHeapPointerName(execution int, dpu_id int) (int64, *encoding.ByteStream) {
	bs := new(encoding.ByteStream)
	bs.Init()
	return 0, bs
}

func (this *Uf3d) NumExecutions() int {
	return this.num_executions
}
