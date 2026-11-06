package prim

// CPU golden model for task_layered_v2.c, the exact local copy of BP-PIM's
// cooperative three-color kernel. Alpha is 1, damping is 1/2, and the fixed
// color traversal order is 1,2,0.

var bpV2CheckColors = [3][24]uint8{
	{1, 4, 6, 9, 14, 17, 19, 22, 24, 27, 32, 35, 37, 40, 42, 45, 50, 53, 55, 58, 60, 63, 68, 71},
	{0, 3, 8, 11, 13, 16, 18, 21, 26, 29, 31, 34, 36, 39, 44, 47, 49, 52, 54, 57, 62, 65, 67, 70},
	{2, 5, 7, 10, 12, 15, 20, 23, 25, 28, 30, 33, 38, 41, 43, 46, 48, 51, 56, 59, 61, 64, 66, 69},
}

func bpDecodeLayeredV2(syn []byte, prior int16, nIter int) ([]int16, bool) {
	var c2v [bpNNZ]int16
	var post [bpNV]int16
	for v := range post {
		post[v] = prior
	}
	colorOrder := [3]uint8{1, 2, 0}

	for iter := 0; iter < nIter; iter++ {
		for _, color := range colorOrder {
			for _, checkIndex := range bpV2CheckColors[color] {
				chk := int(checkIndex)
				b := chk * bpChkDeg
				var q, magnitude [bpChkDeg]int16
				sign := 0
				if syn[chk] != 0 {
					sign = 1
				}
				for j := 0; j < bpChkDeg; j++ {
					v := int(bpChkVarsData[chk][j])
					q[j] = bpSat(int32(post[v]) - int32(c2v[b+j]))
					if q[j] < 0 {
						sign ^= 1
						magnitude[j] = -q[j]
					} else {
						magnitude[j] = q[j]
					}
				}

				mn1, mn2 := magnitude[0], int16(bpFpMax)
				minPos := 0
				for j := 1; j < bpChkDeg; j++ {
					if magnitude[j] <= mn1 {
						mn2, mn1, minPos = mn1, magnitude[j], j
					} else if magnitude[j] < mn2 {
						mn2 = magnitude[j]
					}
				}

				finalIter := iter+1 == nIter
				for j := 0; j < bpChkDeg; j++ {
					outSign := sign
					if q[j] < 0 {
						outSign ^= 1
					}
					raw := mn1
					if j == minPos {
						raw = mn2
					}
					if outSign != 0 {
						raw = -raw
					}
					old := int32(c2v[b+j])
					cv := bpSat(old + ((int32(raw) - old) >> 1))
					v := int(bpChkVarsData[chk][j])
					if !finalIter {
						c2v[b+j] = cv
					}
					post[v] = bpSat(int32(q[j]) + int32(cv))
				}
			}
		}
	}

	converged := true
	for chk := 0; chk < bpNC; chk++ {
		s := uint8(0)
		for j := 0; j < bpChkDeg; j++ {
			if post[bpChkVarsData[chk][j]] <= 0 {
				s ^= 1
			}
		}
		if s != syn[chk] {
			converged = false
			break
		}
	}

	result := make([]int16, bpNV)
	copy(result, post[:])
	return result, converged
}
