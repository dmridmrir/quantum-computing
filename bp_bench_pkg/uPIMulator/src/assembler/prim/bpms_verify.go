package prim

// BP decoder in Go — reproduces task.c logic exactly for output verification.
// Uses the same hardcoded PCM tables from bb144_pcm.h (via bb144_pcm_data.go).

const (
	bpNC     = 72
	bpNV     = 144
	bpNNZ    = 432
	bpChkDeg = 6
	bpVarDeg = 3
	bpFpMax  = 32000
)

func bpAlphaScale(x int16) int16 {
	return int16((int32(x)<<2 + int32(x)) >> 3)
}

func bpSat(x int32) int16 {
	if x > bpFpMax {
		return int16(bpFpMax)
	}
	if x < -bpFpMax {
		return int16(-bpFpMax)
	}
	return int16(x)
}

// bpDecode runs the same fixed-iteration normalized min-sum BP as task.c.
// Uses bpChkVarsData and bpVarMapData from bb144_pcm_data.go.
// Returns (posterior[144], converged).
func bpDecode(syn []byte, prior int16, nIter int) ([]int16, bool) {
	var v2c, c2v [bpNNZ]int16
	var post [bpNV]int16

	for j := 0; j < bpNNZ; j++ {
		v2c[j] = prior
		c2v[j] = 0
	}

	for iter := 0; iter < nIter; iter++ {
		// C2V
		for chk := 0; chk < bpNC; chk++ {
			b := chk * bpChkDeg
			var m [bpChkDeg]int16
			for j := 0; j < bpChkDeg; j++ {
				m[j] = v2c[b+j]
			}
			sign := 0
			if syn[chk] != 0 {
				sign = 1
			}
			for j := 0; j < bpChkDeg; j++ {
				if m[j] < 0 {
					sign ^= 1
				}
			}
			var a [bpChkDeg]int16
			for j := 0; j < bpChkDeg; j++ {
				if m[j] < 0 {
					a[j] = -m[j]
				} else {
					a[j] = m[j]
				}
			}
			mn1 := int16(bpFpMax)
			mn2 := int16(bpFpMax)
			mp := 0
			for j := 0; j < bpChkDeg; j++ {
				if a[j] <= mn1 {
					mn2 = mn1
					mn1 = a[j]
					mp = j
				} else if a[j] < mn2 {
					mn2 = a[j]
				}
			}
			for j := 0; j < bpChkDeg; j++ {
				s := sign
				if m[j] < 0 {
					s ^= 1
				}
				mn := mn1
				if j == mp {
					mn = mn2
				}
				cv := bpAlphaScale(mn)
				if s != 0 {
					cv = -cv
				}
				c2v[b+j] = cv
			}
		}
		// V2C
		for v := 0; v < bpNV; v++ {
			i0 := bpVarMapData[v][0]
			i1 := bpVarMapData[v][1]
			i2 := bpVarMapData[v][2]
			c0 := c2v[i0]
			c1 := c2v[i1]
			c2 := c2v[i2]
			post[v] = bpSat(int32(prior) + int32(c0) + int32(c1) + int32(c2))
			v2c[i0] = bpSat(int32(prior) + int32(c1) + int32(c2))
			v2c[i1] = bpSat(int32(prior) + int32(c0) + int32(c2))
			v2c[i2] = bpSat(int32(prior) + int32(c0) + int32(c1))
		}
	}

	// Convergence check
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

	result := make([]int16, bpNV)
	copy(result, post[:])
	return result, converged
}
