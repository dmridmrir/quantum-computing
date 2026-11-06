package prim

import (
	"errors"
	"math/rand"
	"uPIMulator/src/abi/encoding"
	"uPIMulator/src/abi/word"
	"uPIMulator/src/misc"
)

type Uf struct {
	num_dpus       int
	num_tasklets   int
	num_executions int

	d     int64
	t_val int64
	nn    int64

	// Per-tasklet event data: [num_events, ev0, ev1, ..., 0-padded to 64 words]
	tasklet_slots [][]int64
}

// generateSyndrome produces detection events for a 2D surface code
// with physical error rate p using a correlated error model:
//   60% of p → paired error (node + random spatial neighbor)
//   40% of p → isolated error (single node)
func generateSyndrome(d int64, p float64, rng *rand.Rand) []int64 {
	nn := d * d
	syn := make([]bool, nn)

	for i := int64(0); i < nn; i++ {
		rv := rng.Float64()
		if rv < p*0.6 {
			// Paired error
			syn[i] = !syn[i]
			r := i / d
			c := i % d
			dir := rng.Intn(4)
			v := i
			switch dir {
			case 0:
				if c+1 < d {
					v = r*d + (c + 1)
				}
			case 1:
				if c > 0 {
					v = r*d + (c - 1)
				}
			case 2:
				if r+1 < d {
					v = (r+1)*d + c
				}
			case 3:
				if r > 0 {
					v = (r-1)*d + c
				}
			}
			syn[v] = !syn[v]
		} else if rv < p {
			// Isolated error
			syn[i] = !syn[i]
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

func (this *Uf) Init(command_line_parser *misc.CommandLineParser) {
	num_channels := int(command_line_parser.IntParameter("num_channels"))
	num_ranks_per_channel := int(command_line_parser.IntParameter("num_ranks_per_channel"))
	num_dpus_per_rank := int(command_line_parser.IntParameter("num_dpus_per_rank"))

	this.num_dpus = num_channels * num_ranks_per_channel * num_dpus_per_rank
	this.num_tasklets = int(command_line_parser.IntParameter("num_tasklets"))
	this.num_executions = 1

	// data_prep_params[0] = d (code distance)
	this.d = int64(command_line_parser.DataPrepParams()[0])

	// 2D surface code: T=1, nn = d*d
	this.t_val = 1
	this.nn = this.d * this.d

	// Generate p=0.005 syndromes for each tasklet
	// MRAM layout per tasklet: [num_events, ev0, ev1, ..., 0-padded] = 64 words
	p := 0.005
	this.tasklet_slots = make([][]int64, this.num_tasklets)
	for t := 0; t < this.num_tasklets; t++ {
		rng := rand.New(rand.NewSource(int64(42 + t*137)))
		events := generateSyndrome(this.d, p, rng)

		slot := make([]int64, 64)
		ne := len(events)
		if ne > 63 {
			ne = 63 // max events that fit in slot (word 0 = count)
		}
		slot[0] = int64(ne)
		for i := 0; i < ne; i++ {
			slot[1+i] = events[i]
		}
		this.tasklet_slots[t] = slot
	}
}

func (this *Uf) InputDpuHost(execution int, dpu_id int) map[string]*encoding.ByteStream {
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
		0, // num_events (per-tasklet, stored in MRAM)
		0, 0, 0,
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

func (this *Uf) OutputDpuHost(execution int, dpu_id int) map[string]*encoding.ByteStream {
	return make(map[string]*encoding.ByteStream)
}

func (this *Uf) InputDpuMramHeapPointerName(execution int, dpu_id int) (int64, *encoding.ByteStream) {
	bs := new(encoding.ByteStream)
	bs.Init()

	// Write per-tasklet slots: each 64 words (256 bytes)
	// Layout: [num_events, ev0, ev1, ..., 0-padded]
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

func (this *Uf) OutputDpuMramHeapPointerName(execution int, dpu_id int) (int64, *encoding.ByteStream) {
	bs := new(encoding.ByteStream)
	bs.Init()
	return 0, bs
}

func (this *Uf) NumExecutions() int {
	return this.num_executions
}
