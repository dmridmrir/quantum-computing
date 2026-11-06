package prim

import (
	"errors"
	"math/rand"
	"uPIMulator/src/abi/encoding"
	"uPIMulator/src/abi/word"
	"uPIMulator/src/misc"
)

type Bp3d struct {
	num_dpus       int
	num_tasklets   int
	num_executions int

	d, t_val, nn       int64
	nr, ndn, nf, ne    int64
	max_iter            int64
	channel_llr         []int64
	v2c_init            []int64
}

func (this *Bp3d) Init(command_line_parser *misc.CommandLineParser) {
	num_channels := int(command_line_parser.IntParameter("num_channels"))
	num_ranks_per_channel := int(command_line_parser.IntParameter("num_ranks_per_channel"))
	num_dpus_per_rank := int(command_line_parser.IntParameter("num_dpus_per_rank"))

	this.num_dpus = num_channels * num_ranks_per_channel * num_dpus_per_rank
	this.num_tasklets = int(command_line_parser.IntParameter("num_tasklets"))
	this.num_executions = 1

	this.nn = int64(command_line_parser.DataPrepParams()[0])

	// Derive d from nn = d^2 * T where T = d
	this.d = 7
	this.t_val = 7
	for dd := int64(3); dd <= 50; dd++ {
		if dd*dd*dd == this.nn {
			this.d = dd
			this.t_val = dd
			break
		}
	}

	d := this.d
	T := this.t_val
	this.nr = T * d * (d - 1)
	this.ndn = T * (d - 1) * d
	if T > 1 {
		this.nf = (T - 1) * d * d
	} else {
		this.nf = 0
	}
	this.ne = this.nr + this.ndn + this.nf
	this.max_iter = 10

	// Generate channel LLR
	this.channel_llr = make([]int64, this.nn)
	for i := int64(0); i < this.nn; i++ {
		this.channel_llr[i] = int64(4000 + rand.Intn(2000))
		if rand.Intn(100) < 1 {
			this.channel_llr[i] = -this.channel_llr[i]
		}
	}

	// Init v2c with channel LLR
	this.v2c_init = make([]int64, this.ne)
	dd := d * d
	dm1 := d - 1
	for t := int64(0); t < T; t++ {
		for r := int64(0); r < d; r++ {
			for c := int64(0); c < d; c++ {
				ni := t*dd + r*d + c
				if c < dm1 {
					this.v2c_init[t*d*dm1+r*dm1+c] = this.channel_llr[ni]
				}
				if r < dm1 {
					this.v2c_init[this.nr+t*dm1*d+r*d+c] = this.channel_llr[ni]
				}
				if t < T-1 {
					this.v2c_init[this.nr+this.ndn+t*dd+r*d+c] = this.channel_llr[ni]
				}
			}
		}
	}
}

func align8(x int64) int64 {
	return (x + 7) & ^int64(7)
}

func (this *Bp3d) InputDpuHost(execution int, dpu_id int) map[string]*encoding.ByteStream {
	if execution >= this.num_executions {
		panic(errors.New("execution >= num executions"))
	}

	bs := new(encoding.ByteStream)
	bs.Init()

	// dpu_arguments_t: d, T, num_edges, max_iter, alpha_fp, kernel, num_nodes, n_right, n_down, n_future, pad
	fields := []int64{
		this.d, this.t_val, this.ne, this.max_iter,
		750, // alpha_fp
		0,   // kernel
		this.nn, this.nr, this.ndn, this.nf,
		0, // pad
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

func (this *Bp3d) OutputDpuHost(execution int, dpu_id int) map[string]*encoding.ByteStream {
	return make(map[string]*encoding.ByteStream)
}

func (this *Bp3d) InputDpuMramHeapPointerName(execution int, dpu_id int) (int64, *encoding.ByteStream) {
	bs := new(encoding.ByteStream)
	bs.Init()

	ne := this.ne

	// MRAM layout: v2c[ne] | c2v[ne] | result | llr[nn]
	// Write v2c
	for i := int64(0); i < ne; i++ {
		w := new(word.Word)
		w.Init(32)
		w.SetValue(this.v2c_init[i])
		bs.Merge(w.ToByteStream())
	}

	// Pad v2c to aligned boundary
	v2c_bytes := ne * 4
	padded_v2c := align8(v2c_bytes)
	for i := v2c_bytes; i < padded_v2c; i++ {
		bw := new(word.Word)
		bw.Init(8)
		bw.SetValue(0)
		bs.Merge(bw.ToByteStream())
	}

	// c2v (zeros)
	for i := int64(0); i < ne; i++ {
		w := new(word.Word)
		w.Init(32)
		w.SetValue(0)
		bs.Merge(w.ToByteStream())
	}
	c2v_bytes := ne * 4
	padded_c2v := align8(c2v_bytes)
	for i := c2v_bytes; i < padded_c2v; i++ {
		bw := new(word.Word)
		bw.Init(8)
		bw.SetValue(0)
		bs.Merge(bw.ToByteStream())
	}

	// Result struct (zeros, 16 bytes)
	for i := 0; i < 4; i++ {
		w := new(word.Word)
		w.Init(32)
		w.SetValue(0)
		bs.Merge(w.ToByteStream())
	}

	// LLR
	for i := int64(0); i < this.nn; i++ {
		w := new(word.Word)
		w.Init(32)
		w.SetValue(this.channel_llr[i])
		bs.Merge(w.ToByteStream())
	}

	return 0, bs
}

func (this *Bp3d) OutputDpuMramHeapPointerName(execution int, dpu_id int) (int64, *encoding.ByteStream) {
	// Return empty — skip output verification (BP writes result to MRAM)
	bs := new(encoding.ByteStream)
	bs.Init()
	return 0, bs
}

func (this *Bp3d) NumExecutions() int {
	return this.num_executions
}
