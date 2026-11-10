package dram

import (
	"errors"
	"uPIMulator/src/abi/encoding"
	"uPIMulator/src/misc"
)

type Mram struct {
	address int64
	size    int64

	wordline_size int64
	// Allocate only wordlines that have non-zero contents.  Materializing the
	// full 64 MiB MRAM for every DPU requires 160 GiB before a 2560-DPU run has
	// executed one cycle, even though TESSERACT touches only its packed regions.
	wordlines map[int]*Wordline
}

func (this *Mram) Init(command_line_parser *misc.CommandLineParser) {
	config_loader := new(misc.ConfigLoader)
	config_loader.Init()

	this.address = config_loader.MramOffset()
	this.size = config_loader.MramSize()
	this.wordline_size = command_line_parser.IntParameter("wordline_size")

	if this.wordline_size <= 0 {
		err := errors.New("wordline size <= 0")
		panic(err)
	} else if this.address%this.wordline_size != 0 {
		err := errors.New("address is not aligned with wordline size")
		panic(err)
	} else if this.size%this.wordline_size != 0 {
		err := errors.New("size is not aligned with wordline size")
		panic(err)
	}

	this.wordlines = make(map[int]*Wordline)
}

func (this *Mram) Fini() {
	for _, wordline := range this.wordlines {
		wordline.Fini()
	}
	this.wordlines = nil
}

func (this *Mram) Address() int64 {
	return this.address
}

func (this *Mram) Size() int64 {
	return this.size
}

func (this *Mram) Read(address int64) *encoding.ByteStream {
	index := this.Index(address)
	if wordline, found := this.wordlines[index]; found {
		return wordline.Read()
	}

	byte_stream := new(encoding.ByteStream)
	byte_stream.Init()
	for offset := int64(0); offset < this.wordline_size; offset++ {
		byte_stream.Append(0)
	}
	return byte_stream
}

func (this *Mram) Write(address int64, byte_stream *encoding.ByteStream) {
	index := this.Index(address)
	if byte_stream.Size() != this.wordline_size {
		err := errors.New("MRAM write size != wordline size")
		panic(err)
	}
	all_zero := true
	for offset := int64(0); offset < byte_stream.Size(); offset++ {
		if byte_stream.Get(int(offset)) != 0 {
			all_zero = false
			break
		}
	}
	if all_zero {
		delete(this.wordlines, index)
		return
	}

	wordline, found := this.wordlines[index]
	if !found {
		wordline = new(Wordline)
		wordline.Init(address, this.wordline_size)
		this.wordlines[index] = wordline
	}
	wordline.Write(byte_stream)
}

func (this *Mram) Index(address int64) int {
	if address < this.address {
		err := errors.New("address < MRAM offset")
		panic(err)
	} else if address+this.wordline_size > this.address+this.size {
		err := errors.New("address + wordline size > MRAM offset + MRAM size")
		panic(err)
	} else if address%this.wordline_size != 0 {
		err := errors.New("address is not aligned with wordline size")
		panic(err)
	}

	return int((address - this.address) / this.wordline_size)
}
