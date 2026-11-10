package prim

// BF-UF decoder in Go — CPU reference for verifying task_bfuf.c output.
//
// Ref: Löbl et al., "Breadth-first graph traversal union-find decoder",
//      arXiv:2407.15988 (2024)
//
// Uses bfufChkVarsData from bb144_pcm_data.go.

const (
	bfufNC     = 72
	bfufNV     = 144
	bfufChkDeg = 6
	bfufVarDeg = 3
	bfufTotal  = bfufNC + bfufNV // 216: 0..71 = checks, 72..215 = variables
)

// varChks[v][j] = j-th check index adjacent to variable v.
// Built once from bfufChkVarsData by bfufBuildVarChks().
var bfufVarChks [bfufNV][bfufVarDeg]uint8

func init() {
	bfufBuildVarChks()
}

func bfufBuildVarChks() {
	var cnt [bfufNV]int
	for c := 0; c < bfufNC; c++ {
		for j := 0; j < bfufChkDeg; j++ {
			v := int(bfufChkVarsData[c][j])
			bfufVarChks[v][cnt[v]] = uint8(c)
			cnt[v]++
		}
	}
}

// ---- Union-Find ----

type bfufUF struct {
	par [bfufTotal]uint8
	rnk [bfufTotal]uint8
}

func (uf *bfufUF) reset() {
	for i := range uf.par {
		uf.par[i] = uint8(i)
		uf.rnk[i] = 0
	}
}

func (uf *bfufUF) find(x uint8) uint8 {
	for uf.par[x] != x {
		uf.par[x] = uf.par[uf.par[x]]
		x = uf.par[x]
	}
	return x
}

func (uf *bfufUF) unite(a, b uint8) {
	a, b = uf.find(a), uf.find(b)
	if a == b {
		return
	}
	if uf.rnk[a] < uf.rnk[b] {
		a, b = b, a
	}
	uf.par[b] = a
	if uf.rnk[a] == uf.rnk[b] {
		uf.rnk[a]++
	}
}

// uniteKeepA merges b's cluster into a's cluster, always keeping a's root.
func (uf *bfufUF) uniteKeepA(a, b uint8) {
	ra := uf.find(a)
	rb := uf.find(b)
	if ra == rb {
		return
	}
	uf.par[rb] = ra
	if uf.rnk[ra] <= uf.rnk[rb] {
		uf.rnk[ra] = uf.rnk[rb] + 1
	}
}

// ---- GF(2) Gaussian elimination ----
// Solve H_cl * e = s over GF(2).
// H: nc × nv binary matrix, s: nc vector.
// Returns e (nv bits) or nil if no solution.
// Free variables are set to 0.
func gf2SolveBfuf(H [][]uint8, s []uint8, nc, nv int) []uint8 {
	// augmented matrix rows as []uint8 (1 bit per byte, index 0..nv is data+rhs)
	mat := make([][]uint8, nc)
	for r := 0; r < nc; r++ {
		row := make([]uint8, nv+1)
		copy(row, H[r][:nv])
		row[nv] = s[r]
		mat[r] = row
	}

	prow := 0
	pivotCol := make([]int, nc)
	for i := range pivotCol {
		pivotCol[i] = -1
	}

	for col := 0; col < nv && prow < nc; col++ {
		// find pivot
		pivot := -1
		for r := prow; r < nc; r++ {
			if mat[r][col] != 0 {
				pivot = r
				break
			}
		}
		if pivot < 0 {
			continue
		}
		// swap
		mat[prow], mat[pivot] = mat[pivot], mat[prow]
		pivotCol[prow] = col
		// eliminate
		for r := 0; r < nc; r++ {
			if r != prow && mat[r][col] != 0 {
				for c := 0; c <= nv; c++ {
					mat[r][c] ^= mat[prow][c]
				}
			}
		}
		prow++
	}

	// consistency check
	for r := prow; r < nc; r++ {
		if mat[r][nv] != 0 {
			return nil
		}
	}

	// extract solution
	e := make([]uint8, nv)
	for r := 0; r < prow; r++ {
		if pivotCol[r] >= 0 {
			e[pivotCol[r]] = mat[r][nv]
		}
	}
	return e
}

// bfufClusterSolvable checks if the GF(2) system for root's cluster is solvable.
func bfufClusterSolvable(uf *bfufUF, syn []byte, visited []bool, root uint8) bool {
	var clChecks []int
	var clVars []int
	clVarIdx := [bfufNV]int{}
	for i := range clVarIdx {
		clVarIdx[i] = -1
	}
	for c := 0; c < bfufNC; c++ {
		if visited[c] && uf.find(uint8(c)) == root {
			clChecks = append(clChecks, c)
		}
	}
	for v := 0; v < bfufNV; v++ {
		vn := uint8(bfufNC + v)
		if visited[vn] && uf.find(vn) == root {
			clVarIdx[v] = len(clVars)
			clVars = append(clVars, v)
		}
	}
	ncCl := len(clChecks)
	nvCl := len(clVars)
	if ncCl == 0 || nvCl == 0 {
		return false
	}
	HCl := make([][]uint8, ncCl)
	for r := range HCl {
		HCl[r] = make([]uint8, nvCl)
	}
	sCl := make([]uint8, ncCl)
	for r, c := range clChecks {
		sCl[r] = syn[c]
		for j := 0; j < bfufChkDeg; j++ {
			v := int(bfufChkVarsData[c][j])
			if clVarIdx[v] >= 0 {
				HCl[r][clVarIdx[v]] = 1
			}
		}
	}
	return gf2SolveBfuf(HCl, sCl, ncCl, nvCl) != nil
}

// ---- BF-UF decoder (Algorithm 3) ----
// syn: NC syndrome bits (1 = unsatisfied).
// Returns correction vector of NV bits.
func bfufDecode(syn []byte) []uint8 {
	var uf bfufUF
	uf.reset()

	var (
		visited    [bfufTotal]bool
		parity     [bfufTotal]bool // parity[root] = true: cluster invalid (GE fails)
		numInvalid int
	)

	queue := make([]uint8, 0, bfufTotal)

	// seed BFS from unsatisfied check nodes
	for c := 0; c < bfufNC; c++ {
		if syn[c] != 0 {
			visited[c] = true
			parity[c] = true
			numInvalid++
			queue = append(queue, uint8(c))
		}
	}

	// BFS: check nodes only, double-step check→qubit→check (Algorithm 3)
	qh := 0
	for numInvalid > 0 {
		if qh >= len(queue) {
			break
		}
		node := queue[qh]
		qh++

		root := uf.find(node)
		if !parity[root] {
			continue // valid cluster: skip
		}

		// grow invalid cluster: double-step expansion
		c := node
		for j := 0; j < bfufChkDeg; j++ {
			v := bfufChkVarsData[c][j]
			vn := uint8(bfufNC + int(v))

			// merge qubit into growing cluster
			rvn := uf.find(vn)
			rc := uf.find(node)
			if rvn != rc {
				if parity[rvn] {
					numInvalid--
					parity[rvn] = false
				}
				uf.uniteKeepA(node, vn)
			}
			visited[vn] = true

			// second step: checks adjacent to qubit v
			for k := 0; k < bfufVarDeg; k++ {
				cn := bfufVarChks[v][k]
				rcn := uf.find(cn)
				rc = uf.find(node)
				if rcn != rc {
					if parity[rcn] {
						numInvalid--
						parity[rcn] = false
					}
					uf.uniteKeepA(node, cn)
				}
				if !visited[cn] {
					visited[cn] = true
					queue = append(queue, cn)
				}
			}
		}

		// update_cluster_validity: run GE; if solvable, mark cluster valid
		root = uf.find(node)
		if bfufClusterSolvable(&uf, syn, visited[:], root) {
			if parity[root] {
				parity[root] = false
				numInvalid--
			}
		}
	}

	corr := make([]uint8, bfufNV)
	rootDone := [bfufTotal]bool{}

	for _, node := range queue {
		if node >= bfufNC {
			continue
		}
		root := uf.find(node)
		if rootDone[root] {
			continue
		}
		rootDone[root] = true

		var clChecks []int
		var clVars []int
		clVarIdx := [bfufNV]int{}
		for i := range clVarIdx {
			clVarIdx[i] = -1
		}

		for c := 0; c < bfufNC; c++ {
			if visited[c] && uf.find(uint8(c)) == root {
				clChecks = append(clChecks, c)
			}
		}
		for v := 0; v < bfufNV; v++ {
			vn := uint8(bfufNC + v)
			if visited[vn] && uf.find(vn) == root {
				clVarIdx[v] = len(clVars)
				clVars = append(clVars, v)
			}
		}

		ncCl := len(clChecks)
		nvCl := len(clVars)
		if ncCl == 0 || nvCl == 0 {
			continue
		}

		HCl := make([][]uint8, ncCl)
		for r := range HCl {
			HCl[r] = make([]uint8, nvCl)
		}
		sCl := make([]uint8, ncCl)

		for r, c := range clChecks {
			sCl[r] = syn[c]
			for j := 0; j < bfufChkDeg; j++ {
				v := int(bfufChkVarsData[c][j])
				if clVarIdx[v] >= 0 {
					HCl[r][clVarIdx[v]] = 1
				}
			}
		}

		e := gf2SolveBfuf(HCl, sCl, ncCl, nvCl)
		if e != nil {
			for col, v := range clVars {
				corr[v] ^= e[col]
			}
		}
	}
	return corr
}

// bfufDecodeAlg1 is the Algorithm 1 (simple single-hop BFS) reference decoder.
func bfufDecodeAlg1(syn []byte) []uint8 {
	var uf bfufUF
	uf.reset()

	visited := [bfufTotal]bool{}
	queue := make([]uint8, 0, bfufTotal)

	for c := 0; c < bfufNC; c++ {
		if syn[c] != 0 {
			visited[c] = true
			queue = append(queue, uint8(c))
		}
	}

	for qh := 0; qh < len(queue); qh++ {
		node := queue[qh]
		if node < bfufNC {
			c := node
			for j := 0; j < bfufChkDeg; j++ {
				v := bfufChkVarsData[c][j]
				vn := uint8(bfufNC + int(v))
				uf.unite(node, vn)
				if !visited[vn] {
					visited[vn] = true
					queue = append(queue, vn)
				}
			}
		} else {
			v := int(node) - bfufNC
			for j := 0; j < bfufVarDeg; j++ {
				cn := bfufVarChks[v][j]
				uf.unite(node, cn)
				if !visited[cn] {
					visited[cn] = true
					queue = append(queue, cn)
				}
			}
		}
	}

	corr := make([]uint8, bfufNV)
	rootDone := [bfufTotal]bool{}

	for _, node := range queue {
		if node >= bfufNC {
			continue
		}
		root := uf.find(node)
		if rootDone[root] {
			continue
		}
		rootDone[root] = true

		var clChecks []int
		var clVars []int
		clVarIdx := [bfufNV]int{}
		for i := range clVarIdx {
			clVarIdx[i] = -1
		}
		for c := 0; c < bfufNC; c++ {
			if visited[c] && uf.find(uint8(c)) == root {
				clChecks = append(clChecks, c)
			}
		}
		for v := 0; v < bfufNV; v++ {
			vn := uint8(bfufNC + v)
			if visited[vn] && uf.find(vn) == root {
				clVarIdx[v] = len(clVars)
				clVars = append(clVars, v)
			}
		}
		ncCl := len(clChecks)
		nvCl := len(clVars)
		if ncCl == 0 || nvCl == 0 {
			continue
		}
		HCl := make([][]uint8, ncCl)
		for r := range HCl {
			HCl[r] = make([]uint8, nvCl)
		}
		sCl := make([]uint8, ncCl)
		for r, c := range clChecks {
			sCl[r] = syn[c]
			for j := 0; j < bfufChkDeg; j++ {
				v := int(bfufChkVarsData[c][j])
				if clVarIdx[v] >= 0 {
					HCl[r][clVarIdx[v]] = 1
				}
			}
		}
		e := gf2SolveBfuf(HCl, sCl, ncCl, nvCl)
		if e != nil {
			for col, v := range clVars {
				corr[v] ^= e[col]
			}
		}
	}
	return corr
}

// bfufVerifySyndrome checks H * corr == syn (mod 2).
func bfufVerifySyndrome(corr []uint8, syn []byte) bool {
	for c := 0; c < bfufNC; c++ {
		s := uint8(0)
		for j := 0; j < bfufChkDeg; j++ {
			s ^= corr[bfufChkVarsData[c][j]]
		}
		if s != syn[c] {
			return false
		}
	}
	return true
}
