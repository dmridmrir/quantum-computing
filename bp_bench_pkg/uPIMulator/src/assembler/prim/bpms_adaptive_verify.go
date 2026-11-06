package prim

// Adaptive BP: iter count based on prior LLR
// prior >= 1355 (p <= 0.005) → 3 iterations
// prior <  1355 (p >  0.005) → 5 iterations

func bpDecodeAdaptive(syn []byte, prior int16) ([]int16, bool) {
	nIter := 5
	if prior >= 1355 {
		nIter = 3
	}
	return bpDecode(syn, prior, nIter)
}
