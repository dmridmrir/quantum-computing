package host

import (
	"errors"
	"fmt"
	"os"
	"uPIMulator/src/abi/encoding"
	"uPIMulator/src/simulator/channel"
)

type ChannelTransferReadJob struct {
	channel_message *channel.ChannelMessage

	byte_streams []*encoding.ByteStream

	channel *channel.Channel
}

func (this *ChannelTransferReadJob) Init(
	channel_message *channel.ChannelMessage,
	byte_streams []*encoding.ByteStream,
	channel_ *channel.Channel,
) {
	if channel_message.ChannelOperation() != channel.READ {
		err := errors.New("channel operation is not read")
		panic(err)
	}

	this.channel_message = channel_message
	this.byte_streams = byte_streams
	this.channel = channel_
}

func (this *ChannelTransferReadJob) Execute() {
	this.channel.Lock()

	this.channel.Push(this.channel_message)

	for !this.channel.CanPopChannelMessage(this.channel_message) {
		this.channel.Cycle()
	}

	this.channel.PopChannelMessage(this.channel_message)

	// Dump actual DPU output to file
	actual := this.channel_message.ByteStreams()
	if len(actual) > 0 && actual[0].Size() > 0 {
		f, err := os.Create("/root/uPIMulator/golang/uPIMulator/bin/actual_dpu_output.bin")
		if err == nil {
			for i := 0; i < len(actual); i++ {
				for j := int64(0); j < actual[i].Size(); j++ {
					fmt.Fprintf(f, "%d\n", actual[i].Get(int(j)))
				}
			}
			f.Close()
		}
	}

	this.CompareByteStreams(this.byte_streams, this.channel_message.ByteStreams())

	this.channel.Unlock()
}

func (this *ChannelTransferReadJob) CompareByteStreams(
	byte_streams_1 []*encoding.ByteStream,
	byte_streams_2 []*encoding.ByteStream,
) {
	if len(byte_streams_1) != len(byte_streams_2) {
		err := errors.New("byte streams 1's length != byte streams 2's length")
		panic(err)
	}

	for i := 0; i < len(byte_streams_1); i++ {
		byte_stream_1 := byte_streams_1[i]
		byte_stream_2 := byte_streams_2[i]

		if byte_stream_1.Size() != byte_stream_2.Size() {
			err := errors.New("byte stream 1's size != byte stream 2's size")
			panic(err)
		}

		mismatch_count := 0
		for j := int64(0); j < byte_stream_1.Size(); j++ {
			if byte_stream_1.Get(int(j)) != byte_stream_2.Get(int(j)) {
				if mismatch_count < 30 {
					fmt.Printf("MISMATCH stream=%d byte=%d: expected=0x%02x actual=0x%02x\n",
						i, j, byte_stream_1.Get(int(j)), byte_stream_2.Get(int(j)))
				}
				mismatch_count++
			}
		}
		if mismatch_count > 0 {
			fmt.Printf("Total mismatches: %d / %d bytes (stream %d, sizes: expected=%d actual=%d)\n",
				mismatch_count, byte_stream_1.Size(), i, byte_stream_1.Size(), byte_stream_2.Size())
		}
	}
}
