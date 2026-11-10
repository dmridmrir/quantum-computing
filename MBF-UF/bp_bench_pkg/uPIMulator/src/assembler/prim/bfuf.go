package prim

import (
	"errors"
	"math/rand"
	"uPIMulator/src/abi/encoding"
	"uPIMulator/src/abi/word"
	"uPIMulator/src/misc"
)

type Bfuf struct {
	numDpus       int
	numTasklets   int
	numExecutions int
	nc            int64
	nv            int64
	syndromes     [][]byte
}

func (this *Bfuf) Init(command_line_parser *misc.CommandLineParser) {
	num_channels := int(command_line_parser.IntParameter("num_channels"))
	num_ranks_per_channel := int(command_line_parser.IntParameter("num_ranks_per_channel"))
	num_dpus_per_rank := int(command_line_parser.IntParameter("num_dpus_per_rank"))
	this.numDpus = num_channels * num_ranks_per_channel * num_dpus_per_rank
	this.numTasklets = int(command_line_parser.IntParameter("num_tasklets"))
	this.numExecutions = 1
	this.nc = bfufNC
	this.nv = bfufNV

	this.syndromes = make([][]byte, this.numTasklets)
	for t := 0; t < this.numTasklets; t++ {
		rng := rand.New(rand.NewSource(int64(12345 + t*137)))
		error := make([]uint8, this.nv)
		for v := int64(0); v < this.nv; v++ {
			if rng.Float64() < 0.03 {
				error[v] = 1
			}
		}
		syn := make([]byte, this.nc)
		for c := int64(0); c < this.nc; c++ {
			s := byte(0)
			for j := 0; j < bfufChkDeg; j++ {
				s ^= error[bfufChkVarsData[c][j]]
			}
			syn[c] = s
		}
		this.syndromes[t] = syn
	}
}

func (this *Bfuf) InputDpuHost(execution int, dpu_id int) map[string]*encoding.ByteStream {
	if execution >= this.numExecutions {
		panic(errors.New("execution >= num executions"))
	}
	bs := new(encoding.ByteStream)
	bs.Init()
	for _, v := range []int64{int64(this.numTasklets), 0, 0} {
		w := new(word.Word)
		w.Init(32)
		w.SetValue(v)
		bs.Merge(w.ToByteStream())
	}
	dpu_host := make(map[string]*encoding.ByteStream)
	dpu_host["DPU_INPUT_ARGUMENTS"] = bs
	return dpu_host
}

func (this *Bfuf) OutputDpuHost(execution int, dpu_id int) map[string]*encoding.ByteStream {
	return make(map[string]*encoding.ByteStream)
}

func (this *Bfuf) InputDpuMramHeapPointerName(execution int, dpu_id int) (int64, *encoding.ByteStream) {
	bs := new(encoding.ByteStream)
	bs.Init()
	for t := 0; t < this.numTasklets; t++ {
		for _, b := range this.syndromes[t] {
			w := new(word.Word)
			w.Init(8)
			w.SetValue(int64(b))
			bs.Merge(w.ToByteStream())
		}
	}
	return 0, bs
}

func (this *Bfuf) OutputDpuMramHeapPointerName(execution int, dpu_id int) (int64, *encoding.ByteStream) {
	bs := new(encoding.ByteStream)
	bs.Init()
	for t := 0; t < this.numTasklets; t++ {
		corr := bfufDecode(this.syndromes[t])
		for i := 0; i < int(this.nv); i++ {
			w := new(word.Word)
			w.Init(8)
			w.SetValue(int64(corr[i]))
			bs.Merge(w.ToByteStream())
		}
	}
	synSlotSize := int64(this.nc)
	return int64(this.numTasklets) * synSlotSize, bs
}

func (this *Bfuf) NumExecutions() int { return this.numExecutions }
