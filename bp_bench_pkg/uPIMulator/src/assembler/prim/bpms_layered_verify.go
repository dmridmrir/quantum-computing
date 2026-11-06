package prim

// bpDecodeLayered reproduces cooperative task_layered.c exactly for output
// verification. Three-color layered NMS BP: shared in-place message array, int32
// c2v/lambda, q NOT pre-saturated, internal state kept wide, exported posterior
// clamped to int16 range, alpha=6/8=0.75, no damping, and color order 1,2,0. Checks within a color
// have disjoint variables and are therefore equivalent to the DPU's parallel
// 11-tasklet updates.

var bpV1CheckColors = [3][24]uint8{
	{1, 4, 6, 9, 14, 17, 19, 22, 24, 27, 32, 35, 37, 40, 42, 45, 50, 53, 55, 58, 60, 63, 68, 71},
	{0, 3, 8, 11, 13, 16, 18, 21, 26, 29, 31, 34, 36, 39, 44, 47, 49, 52, 54, 57, 62, 65, 67, 70},
	{2, 5, 7, 10, 12, 15, 20, 23, 25, 28, 30, 33, 38, 41, 43, 46, 48, 51, 56, 59, 61, 64, 66, 69},
}

func bpAlphaScaleLayered32(x int32) int32 {
	return (x<<1 + x) >> 2
}

// bpDecodeLayered runs the layered NMS BP variant. Returns (posterior[144], converged).
func bpDecodeLayered(syn []byte, prior int16, nIter int) ([]int16, bool) {
	var c2v [bpNNZ]int32
	var lambda [bpNV]int32

	p := int32(prior)
	for v := 0; v < bpNV; v++ {
		lambda[v] = p
	}
	for j := 0; j < bpNNZ; j++ {
		c2v[j] = 0
	}

	for iter := 0; iter < nIter; iter++ {
		for _, color := range [3]uint8{1, 2, 0} {
			for _, checkIndex := range bpV1CheckColors[color] {
			chk := int(checkIndex)
			b := chk * bpChkDeg
			var q, a [bpChkDeg]int32
			sign := 0
			if syn[chk] != 0 {
				sign = 1
			}
			for j := 0; j < bpChkDeg; j++ {
				v := bpChkVarsData[chk][j]
				qv := lambda[v] - c2v[b+j]
				q[j] = qv
				if qv < 0 {
					sign ^= 1
				}
				if qv < 0 {
					a[j] = -qv
				} else {
					a[j] = qv
				}
			}

			mn1 := a[0]
			mn2 := a[1]
			mp := 0
			if mn2 <= mn1 {
				mn1, mn2 = mn2, mn1
				mp = 1
			}
			for j := 2; j < bpChkDeg; j++ {
				if a[j] <= mn1 {
					mn2 = mn1
					mn1 = a[j]
					mp = j
				} else if a[j] < mn2 {
					mn2 = a[j]
				}
			}

			for j := 0; j < bpChkDeg; j++ {
				v := bpChkVarsData[chk][j]
				s := sign
				if q[j] < 0 {
					s ^= 1
				}
				mn := mn1
				if j == mp {
					mn = mn2
				}
				cv := bpAlphaScaleLayered32(mn)
				if s != 0 {
					cv = -cv
				}
				c2v[b+j] = cv
				lambda[v] = q[j] + cv
			}
			}
		}
	}

	post := make([]int16, bpNV)
	for v := 0; v < bpNV; v++ {
		post[v] = int16(bpSat32Layered(lambda[v]))
	}

	converged := true
	for chk := 0; chk < bpNC; chk++ {
		s := uint8(0)
		for j := 0; j < bpChkDeg; j++ {
			vi := bpChkVarsData[chk][j]
			if post[vi] <= 0 {
				s ^= 1
			}
		}
		if s != syn[chk] {
			converged = false
			break
		}
	}

	return post, converged
}

func bpSat32Layered(x int32) int32 {
	if x > bpFpMax {
		return bpFpMax
	}
	if x < -bpFpMax {
		return -bpFpMax
	}
	return x
}
