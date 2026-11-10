package compiler

import (
	"uPIMulator/src/misc"
)

type Compiler struct {
	command_line_parser *misc.CommandLineParser

	root_dirpath string
	benchmark    string

	num_dpus     int
	num_tasklets int
}

func (this *Compiler) Init(command_line_parser *misc.CommandLineParser) {
	this.command_line_parser = command_line_parser

	this.root_dirpath = command_line_parser.StringParameter("root_dirpath")
	this.benchmark = command_line_parser.StringParameter("benchmark")

	num_channels := int(command_line_parser.IntParameter("num_channels"))
	num_ranks_per_channel := int(command_line_parser.IntParameter("num_ranks_per_channel"))
	num_dpus_per_rank := int(command_line_parser.IntParameter("num_dpus_per_rank"))
	this.num_dpus = num_channels * num_ranks_per_channel * num_dpus_per_rank

	this.num_tasklets = int(command_line_parser.IntParameter("num_tasklets"))

	// Skip Docker build — we compile manually
}

func (this *Compiler) Compile() {
	// Skip Docker compile — benchmark and SDK already built manually
}
