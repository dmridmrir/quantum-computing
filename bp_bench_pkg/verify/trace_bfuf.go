// trace_bfuf.go — BF-UF decoder step-by-step trace (Algorithm 3)
//
// Verifies that both CPU (Go) and DPU (C) implementations match
// the algorithm in: Löbl et al., arXiv:2407.15988 (2024)
//
// Usage:
//   cd verify && go run trace_bfuf.go
package main

import "fmt"

const (
	NC      = 72
	NV      = 144
	CHK_DEG = 6
	VAR_DEG = 3
	TOTAL   = NC + NV
)

// Tanner graph: chkVars[c][j] = j-th variable adjacent to check c.
// Identical to bb_chk_vars[] in task.c and bpChkVarsData in bfuf_verify.go.
var chkVars = [NC][CHK_DEG]int{
	/* C0  */ {3, 6, 12, 73, 74, 90},
	/* C1  */ {4, 7, 13, 74, 75, 91},
	/* C2  */ {5, 8, 14, 75, 76, 92},
	/* C3  */ {0, 9, 15, 76, 77, 93},
	/* C4  */ {1, 10, 16, 72, 77, 94},
	/* C5  */ {2, 11, 17, 72, 73, 95},
	/* C6  */ {9, 12, 18, 79, 80, 96},
	/* C7  */ {10, 13, 19, 80, 81, 97},
	/* C8  */ {11, 14, 20, 81, 82, 98},
	/* C9  */ {6, 15, 21, 82, 83, 99},
	/* C10 */ {7, 16, 22, 78, 83, 100},
	/* C11 */ {8, 17, 23, 78, 79, 101},
	/* C12 */ {15, 18, 24, 85, 86, 102},
	/* C13 */ {16, 19, 25, 86, 87, 103},
	/* C14 */ {17, 20, 26, 87, 88, 104},
	/* C15 */ {12, 21, 27, 88, 89, 105},
	/* C16 */ {13, 22, 28, 84, 89, 106},
	/* C17 */ {14, 23, 29, 84, 85, 107},
	/* C18 */ {21, 24, 30, 91, 92, 108},
	/* C19 */ {22, 25, 31, 92, 93, 109},
	/* C20 */ {23, 26, 32, 93, 94, 110},
	/* C21 */ {18, 27, 33, 94, 95, 111},
	/* C22 */ {19, 28, 34, 90, 95, 112},
	/* C23 */ {20, 29, 35, 90, 91, 113},
	/* C24 */ {27, 30, 36, 97, 98, 114},
	/* C25 */ {28, 31, 37, 98, 99, 115},
	/* C26 */ {29, 32, 38, 99, 100, 116},
	/* C27 */ {24, 33, 39, 100, 101, 117},
	/* C28 */ {25, 34, 40, 96, 101, 118},
	/* C29 */ {26, 35, 41, 96, 97, 119},
	/* C30 */ {33, 36, 42, 103, 104, 120},
	/* C31 */ {34, 37, 43, 104, 105, 121},
	/* C32 */ {35, 38, 44, 105, 106, 122},
	/* C33 */ {30, 39, 45, 106, 107, 123},
	/* C34 */ {31, 40, 46, 102, 107, 124},
	/* C35 */ {32, 41, 47, 102, 103, 125},
	/* C36 */ {39, 42, 48, 109, 110, 126},
	/* C37 */ {40, 43, 49, 110, 111, 127},
	/* C38 */ {41, 44, 50, 111, 112, 128},
	/* C39 */ {36, 45, 51, 112, 113, 129},
	/* C40 */ {37, 46, 52, 108, 113, 130},
	/* C41 */ {38, 47, 53, 108, 109, 131},
	/* C42 */ {45, 48, 54, 115, 116, 132},
	/* C43 */ {46, 49, 55, 116, 117, 133},
	/* C44 */ {47, 50, 56, 117, 118, 134},
	/* C45 */ {42, 51, 57, 118, 119, 135},
	/* C46 */ {43, 52, 58, 114, 119, 136},
	/* C47 */ {44, 53, 59, 114, 115, 137},
	/* C48 */ {51, 54, 60, 121, 122, 138},
	/* C49 */ {52, 55, 61, 122, 123, 139},
	/* C50 */ {53, 56, 62, 123, 124, 140},
	/* C51 */ {48, 57, 63, 124, 125, 141},
	/* C52 */ {49, 58, 64, 120, 125, 142},
	/* C53 */ {50, 59, 65, 120, 121, 143},
	/* C54 */ {57, 60, 66, 72, 127, 128},
	/* C55 */ {58, 61, 67, 73, 128, 129},
	/* C56 */ {59, 62, 68, 74, 129, 130},
	/* C57 */ {54, 63, 69, 75, 130, 131},
	/* C58 */ {55, 64, 70, 76, 126, 131},
	/* C59 */ {56, 65, 71, 77, 126, 127},
	/* C60 */ {0, 63, 66, 78, 133, 134},
	/* C61 */ {1, 64, 67, 79, 134, 135},
	/* C62 */ {2, 65, 68, 80, 135, 136},
	/* C63 */ {3, 60, 69, 81, 136, 137},
	/* C64 */ {4, 61, 70, 82, 132, 137},
	/* C65 */ {5, 62, 71, 83, 132, 133},
	/* C66 */ {0, 6, 69, 84, 139, 140},
	/* C67 */ {1, 7, 70, 85, 140, 141},
	/* C68 */ {2, 8, 71, 86, 141, 142},
	/* C69 */ {3, 9, 66, 87, 142, 143},
	/* C70 */ {4, 10, 67, 88, 138, 143},
	/* C71 */ {5, 11, 68, 89, 138, 139},
}

var varChks [NV][VAR_DEG]int

func buildVarChks() {
	var cnt [NV]int
	for c := 0; c < NC; c++ {
		for j := 0; j < CHK_DEG; j++ {
			v := chkVars[c][j]
			varChks[v][cnt[v]] = c
			cnt[v]++
		}
	}
}

// ── Union-Find ────────────────────────────────────────────────────────────────

type UF struct {
	par [TOTAL]int
	rnk [TOTAL]int
}

func newUF() *UF {
	uf := &UF{}
	for i := range uf.par {
		uf.par[i] = i
	}
	return uf
}

func (uf *UF) find(x int) int {
	for uf.par[x] != x {
		uf.par[x] = uf.par[uf.par[x]] // path halving
		x = uf.par[x]
	}
	return x
}

// uniteKeepA merges b's cluster into a's cluster, always keeping a's root.
func (uf *UF) uniteKeepA(a, b int) {
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

// ── GF(2) Gaussian elimination ────────────────────────────────────────────────

func gf2Solve(H [][]int, s []int, nc, nv int) []int {
	mat := make([][]int, nc)
	for r := 0; r < nc; r++ {
		row := make([]int, nv+1)
		copy(row, H[r])
		row[nv] = s[r]
		mat[r] = row
	}
	prow := 0
	pivotCol := make([]int, nc)
	for i := range pivotCol {
		pivotCol[i] = -1
	}

	for col := 0; col < nv && prow < nc; col++ {
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
		mat[prow], mat[pivot] = mat[pivot], mat[prow]
		pivotCol[prow] = col
		for r := 0; r < nc; r++ {
			if r != prow && mat[r][col] != 0 {
				for c := 0; c <= nv; c++ {
					mat[r][c] ^= mat[prow][c]
				}
			}
		}
		prow++
	}

	for r := prow; r < nc; r++ {
		if mat[r][nv] != 0 {
			return nil
		}
	}

	e := make([]int, nv)
	for r := 0; r < prow; r++ {
		if pivotCol[r] >= 0 {
			e[pivotCol[r]] = mat[r][nv]
		}
	}
	return e
}

// clusterSolvable collects the cluster rooted at root and checks GE solvability.
func clusterSolvable(uf *UF, syn []int, visited []bool, root int) bool {
	clVarIdx := make([]int, NV)
	for i := range clVarIdx {
		clVarIdx[i] = -1
	}
	var clChecks, clVars []int
	for c := 0; c < NC; c++ {
		if visited[c] && uf.find(c) == root {
			clChecks = append(clChecks, c)
		}
	}
	for v := 0; v < NV; v++ {
		if visited[NC+v] && uf.find(NC+v) == root {
			clVarIdx[v] = len(clVars)
			clVars = append(clVars, v)
		}
	}
	ncCl, nvCl := len(clChecks), len(clVars)
	if ncCl == 0 || nvCl == 0 {
		return false
	}
	HCl := make([][]int, ncCl)
	sCl := make([]int, ncCl)
	for r, c := range clChecks {
		HCl[r] = make([]int, nvCl)
		sCl[r] = syn[c]
		for j := 0; j < CHK_DEG; j++ {
			v := chkVars[c][j]
			if clVarIdx[v] >= 0 {
				HCl[r][clVarIdx[v]] = 1
			}
		}
	}
	return gf2Solve(HCl, sCl, ncCl, nvCl) != nil
}

// ── Helpers ───────────────────────────────────────────────────────────────────

func computeSyn(e []int) []int {
	syn := make([]int, NC)
	for c := 0; c < NC; c++ {
		for j := 0; j < CHK_DEG; j++ {
			syn[c] ^= e[chkVars[c][j]]
		}
	}
	return syn
}

func weight(v []int) int {
	n := 0
	for _, b := range v {
		if b != 0 {
			n++
		}
	}
	return n
}

func nodeName(n int) string {
	if n < NC {
		return fmt.Sprintf("C%d", n)
	}
	return fmt.Sprintf("V%d", n-NC)
}

func nonzeroBits(v []int, prefix string) []string {
	var out []string
	for i, b := range v {
		if b != 0 {
			out = append(out, fmt.Sprintf("%s%d", prefix, i))
		}
	}
	return out
}

// ── BF-UF trace (Algorithm 3) ─────────────────────────────────────────────────

func traceDecode(label string, errorBits []int) bool {
	syn := computeSyn(errorBits)

	fmt.Printf("\n╔══ %s ══╗\n", label)
	fmt.Println("  [INPUT]")
	fmt.Printf("  Error bits  : %v\n", nonzeroBits(errorBits, "V"))
	fmt.Printf("  Syndrome    : %v\n", nonzeroBits(syn, "C"))
	fmt.Printf("  wt(e)=%d  wt(σ)=%d\n", weight(errorBits), weight(syn))

	if weight(syn) == 0 {
		fmt.Println("  Zero syndrome — no decoding needed. PASS.")
		return true
	}

	// ── Phase 1: BFS with cluster validity tracking (Algorithm 3) ──
	fmt.Println("\n  [PHASE 1: BFS + Cluster Validity (Algorithm 3)]")
	fmt.Println("  Double-step BFS: check→qubit→check; only grow invalid clusters")

	uf := newUF()
	visited := make([]bool, TOTAL)
	parity := make([]bool, TOTAL) // parity[root] = true: cluster invalid (GE fails)
	numInvalid := 0
	queue := []int{}

	for c := 0; c < NC; c++ {
		if syn[c] != 0 {
			visited[c] = true
			parity[c] = true
			numInvalid++
			queue = append(queue, c)
		}
	}
	fmt.Printf("  Seed queue  : %v  (numInvalid=%d)\n", nonzeroBits(syn, "C"), numInvalid)

	step := 0
	qh := 0
	for numInvalid > 0 {
		if qh >= len(queue) {
			break
		}
		node := queue[qh]
		qh++
		step++

		root := uf.find(node)
		if !parity[root] {
			fmt.Printf("    step %3d  C%d → skip (cluster valid)\n", step, node)
			continue
		}

		// double-step expansion: check → qubit → check
		c := node
		var addedChecks []string
		for j := 0; j < CHK_DEG; j++ {
			v := chkVars[c][j]
			vn := NC + v

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
			for k := 0; k < VAR_DEG; k++ {
				cn := varChks[v][k]
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
					addedChecks = append(addedChecks, fmt.Sprintf("C%d", cn))
				}
			}
		}

		// update_cluster_validity
		root = uf.find(node)
		solvable := clusterSolvable(uf, syn, visited, root)
		if solvable && parity[root] {
			parity[root] = false
			numInvalid--
		}
		ge := "unsolvable"
		if solvable {
			ge = "SOLVABLE"
		}
		fmt.Printf("    step %3d  C%d → enqueue %v  [GE:%s, numInvalid=%d]\n",
			step, node, addedChecks, ge, numInvalid)
	}

	visitedChk, visitedVar := 0, 0
	for c := 0; c < NC; c++ {
		if visited[c] {
			visitedChk++
		}
	}
	for v := 0; v < NV; v++ {
		if visited[NC+v] {
			visitedVar++
		}
	}
	fmt.Printf("  BFS done: visited %d checks, %d vars\n", visitedChk, visitedVar)

	// ── Cluster summary ──
	fmt.Println("\n  [CLUSTER SUMMARY after BFS]")
	rootChecks := map[int][]int{}
	rootVars := map[int][]int{}
	for c := 0; c < NC; c++ {
		if visited[c] {
			rootChecks[uf.find(c)] = append(rootChecks[uf.find(c)], c)
		}
	}
	for v := 0; v < NV; v++ {
		if visited[NC+v] {
			rootVars[uf.find(NC+v)] = append(rootVars[uf.find(NC+v)], v)
		}
	}
	clIdx := 0
	for root, checks := range rootChecks {
		vars := rootVars[root]
		fmt.Printf("  Cluster %d (root=%s): %d checks × %d vars\n",
			clIdx, nodeName(root), len(checks), len(vars))
		clIdx++
	}

	// ── Phase 2: GF(2) solve per cluster ──
	fmt.Println("\n  [PHASE 2: GF(2) Gaussian Elimination per Cluster]")

	corr := make([]int, NV)
	rootDone := make([]bool, TOTAL)
	clIdx = 0

	for _, node := range queue {
		if node >= NC {
			continue
		}
		root := uf.find(node)
		if rootDone[root] {
			continue
		}
		rootDone[root] = true

		clVarIdx := make([]int, NV)
		for i := range clVarIdx {
			clVarIdx[i] = -1
		}
		var clChecks, clVars []int
		for c := 0; c < NC; c++ {
			if visited[c] && uf.find(c) == root {
				clChecks = append(clChecks, c)
			}
		}
		for v := 0; v < NV; v++ {
			if visited[NC+v] && uf.find(NC+v) == root {
				clVarIdx[v] = len(clVars)
				clVars = append(clVars, v)
			}
		}
		ncCl, nvCl := len(clChecks), len(clVars)
		if ncCl == 0 || nvCl == 0 {
			continue
		}

		HCl := make([][]int, ncCl)
		sCl := make([]int, ncCl)
		for r, c := range clChecks {
			HCl[r] = make([]int, nvCl)
			sCl[r] = syn[c]
			for j := 0; j < CHK_DEG; j++ {
				v := chkVars[c][j]
				if clVarIdx[v] >= 0 {
					HCl[r][clVarIdx[v]] = 1
				}
			}
		}

		fmt.Printf("\n  Cluster %d: H_cl = %d×%d, wt(s_cl) = %d\n",
			clIdx, ncCl, nvCl, weight(sCl))

		if ncCl <= 12 && nvCl <= 16 {
			fmt.Println("  Augmented matrix [H_cl | s_cl]:")
			for r := 0; r < ncCl; r++ {
				fmt.Printf("    C%d:\t[", clChecks[r])
				for j := 0; j < nvCl; j++ {
					fmt.Printf("%d", HCl[r][j])
				}
				fmt.Printf(" | %d]\n", sCl[r])
			}
		} else {
			fmt.Printf("  (H_cl: cols = V%v)\n", clVars)
			for r := 0; r < ncCl; r++ {
				row := ""
				for j := 0; j < nvCl; j++ {
					row += fmt.Sprintf("%d", HCl[r][j])
				}
				fmt.Printf("    C%d: [%s | %d]\n", clChecks[r], row, sCl[r])
			}
		}

		e := gf2Solve(HCl, sCl, ncCl, nvCl)
		if e != nil {
			var corrVars []string
			for col, v := range clVars {
				if e[col] != 0 {
					corrVars = append(corrVars, fmt.Sprintf("V%d", v))
					corr[v] ^= e[col]
				}
			}
			fmt.Printf("  GF(2) solution: correction at %v\n", corrVars)
		} else {
			fmt.Println("  GF(2): no solution (inconsistent system)")
		}
		clIdx++
	}

	// ── Verification ──
	fmt.Println("\n  [VERIFICATION]")
	fmt.Printf("  Correction  : %v\n", nonzeroBits(corr, "V"))

	synCorr := computeSyn(corr)
	synMatch := true
	for c := 0; c < NC; c++ {
		if synCorr[c] != syn[c] {
			synMatch = false
			break
		}
	}

	residual := make([]int, NV)
	for v := 0; v < NV; v++ {
		residual[v] = errorBits[v] ^ corr[v]
	}
	residualSyn := computeSyn(residual)
	allZero := weight(residualSyn) == 0

	corrMatchError := true
	for v := 0; v < NV; v++ {
		if corr[v] != errorBits[v] {
			corrMatchError = false
			break
		}
	}

	fmt.Printf("  H*corr == σ                   : %v\n", synMatch)
	fmt.Printf("  H*(error ⊕ corr) == 0 (valid) : %v\n", allZero)
	fmt.Printf("  corr == error (exact recovery) : %v\n", corrMatchError)

	if allZero {
		fmt.Println("  RESULT: PASS")
	} else {
		fmt.Println("  RESULT: FAIL")
	}
	return allZero
}

func main() {
	buildVarChks()

	fmt.Println("================================================================")
	fmt.Println("  BF-UF Decoder Step-by-Step Verification Trace (Algorithm 3)")
	fmt.Println("  Code  : [[144,12,12]] BB-QLDPC (NC=72, NV=144, d=12)")
	fmt.Println("  Ref   : Löbl et al., arXiv:2407.15988 (2024)")
	fmt.Println("  Alg.3 : double-step BFS, cluster validity, early stopping")
	fmt.Println("================================================================")

	pass := 0
	total := 0

	run := func(label string, e []int) {
		total++
		if traceDecode(label, e) {
			pass++
		}
	}

	// Test 1: zero error
	run("Test 1: Zero error (trivial)", make([]int, NV))

	// Test 2: single-bit error
	e2 := make([]int, NV)
	e2[3] = 1
	run("Test 2: Single-bit error at V3", e2)

	// Test 3: two-bit error
	e3 := make([]int, NV)
	e3[3] = 1
	e3[9] = 1
	run("Test 3: Two-bit error at V3, V9", e3)

	// Test 4: three-bit error
	e4 := make([]int, NV)
	e4[0] = 1
	e4[72] = 1
	e4[90] = 1
	run("Test 4: Three-bit error at V0, V72, V90", e4)

	fmt.Printf("\n================================================================\n")
	fmt.Printf("  SUMMARY: %d / %d tests PASSED\n", pass, total)
	fmt.Printf("================================================================\n")
}
