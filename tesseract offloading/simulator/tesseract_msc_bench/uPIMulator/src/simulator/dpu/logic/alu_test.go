package logic

import "testing"

func TestAluXorUsesBothOperands(t *testing.T) {
	alu := new(Alu)
	alu.Init()

	tests := []struct {
		left  int64
		right int64
		want  int64
	}{
		{0, 0, 0},
		{0, 1, 1},
		{1, 0, 1},
		{1, 1, 0},
		{0x55aa55aa, 0x0f0f0f0f, 0x5aa55aa5},
	}

	for _, test := range tests {
		if got := alu.Xor(test.left, test.right); got != test.want {
			t.Fatalf("Xor(%#x, %#x) = %#x, want %#x", test.left, test.right, got, test.want)
		}
	}
}

func TestAluNxorIsXorComplement(t *testing.T) {
	alu := new(Alu)
	alu.Init()

	const left = int64(0x55aa55aa)
	const right = int64(0x0f0f0f0f)
	const mask = int64(0xffffffff)
	want := (^alu.Xor(left, right)) & mask
	if got := alu.Nxor(left, right); got != want {
		t.Fatalf("Nxor(%#x, %#x) = %#x, want %#x", left, right, got, want)
	}
}
