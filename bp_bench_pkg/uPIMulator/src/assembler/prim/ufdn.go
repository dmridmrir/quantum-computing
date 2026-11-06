package prim

import (
	"errors"
	"math/rand"
	"uPIMulator/src/abi/encoding"
	"uPIMulator/src/abi/word"
	"uPIMulator/src/misc"
)

type Ufdn struct {
	num_dpus       int
	num_tasklets   int
	num_executions int

	L  int64
	NV int64
	NE int64

	// Per-tasklet error patterns
	tasklet_errors [][]byte
}

func (this *Ufdn) Init(command_line_parser *misc.CommandLineParser) {
	num_channels := int(command_line_parser.IntParameter("num_channels"))
	num_ranks_per_channel := int(command_line_parser.IntParameter("num_ranks_per_channel"))
	num_dpus_per_rank := int(command_line_parser.IntParameter("num_dpus_per_rank"))

	this.num_dpus = num_channels * num_ranks_per_channel * num_dpus_per_rank
	this.num_tasklets = int(command_line_parser.IntParameter("num_tasklets"))
	this.num_executions = 1

	this.L = int64(command_line_parser.DataPrepParams()[0])
	this.NV = this.L * this.L
	this.NE = 2 * this.NV

	// Generate p=0.005 error patterns for each tasklet
	p := 0.005
	this.tasklet_errors = make([][]byte, this.num_tasklets)
	for t := 0; t < this.num_tasklets; t++ {
		rng := rand.New(rand.NewSource(int64(42 + t*137)))
		errs := make([]byte, this.NE)
		for i := int64(0); i < this.NE; i++ {
			if rng.Float64() < p {
				errs[i] = 1
			}
		}
		// Pad to 8-byte alignment
		padded := ((int(this.NE) + 7) / 8) * 8
		for len(errs) < padded {
			errs = append(errs, 0)
		}
		this.tasklet_errors[t] = errs
	}
}

func (this *Ufdn) InputDpuHost(execution int, dpu_id int) map[string]*encoding.ByteStream {
	if execution >= this.num_executions {
		panic(errors.New("execution >= num executions"))
	}

	bs := new(encoding.ByteStream)
	bs.Init()

	// dpu_arguments_t: L, num_vertices, num_edges, num_events, size, transfer_size, kernel
	fields := []int64{
		this.L,
		this.NV,
		this.NE,
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

func (this *Ufdn) OutputDpuHost(execution int, dpu_id int) map[string]*encoding.ByteStream {
	return make(map[string]*encoding.ByteStream)
}

func (this *Ufdn) InputDpuMramHeapPointerName(execution int, dpu_id int) (int64, *encoding.ByteStream) {
	bs := new(encoding.ByteStream)
	bs.Init()

	for t := 0; t < this.num_tasklets; t++ {
		for _, b := range this.tasklet_errors[t] {
			w := new(word.Word)
			w.Init(8)
			w.SetValue(int64(b))
			bs.Merge(w.ToByteStream())
		}
	}

	return 0, bs
}

func (this *Ufdn) OutputDpuMramHeapPointerName(execution int, dpu_id int) (int64, *encoding.ByteStream) {
	bs := new(encoding.ByteStream)
	bs.Init()
	return 0, bs
}

func (this *Ufdn) NumExecutions() int {
	return this.num_executions
}
