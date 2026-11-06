# BF-UF Algorithm — Pseudocode to Code Mapping

**Paper**: Löbl et al., "Breadth-first graph traversal union-find decoder",  
arXiv:2407.15988 (2024)  
**Code**: `[[144,12,12]]` BB-QLDPC, NC=72, NV=144, CHK_DEG=6, VAR_DEG=3

---

## Algorithm 1: BF-UF Decoder

| Step | Paper Pseudocode | Go: `bfuf_verify.go` | C: `task.c` |
|------|-----------------|----------------------|-------------|
| 1 | Initialize UF: each node its own singleton | `uf.reset()` → line 44–48 | `for i: uf_par[tid][i]=i` → line 197–200 |
| 2 | Q ← {c ∈ checks : σ_c = 1} | `for c < bfufNC: if syn[c]!=0 { queue append }` → line 148–153 | `for c < BB_NC: if syn[c] { bfs_q[tid][qt++]=c }` → line 206–211 |
| 3 | Mark Q nodes as visited | `visited[c] = true` → line 151 | `visited[tid][c] = 1` → line 208 |
| 4 | **BFS loop**: while Q not empty | `for qh := 0; qh < len(queue); qh++` → line 156 | `while (qh < qt)` → line 213 |
| 5 | u ← dequeue(Q) | `node := queue[qh]` → line 157 | `node = bfs_q[tid][qh++]` → line 214 |
| 6 | **if** u is check node | `if node < bfufNC` → line 158 | `if (node < BB_NC)` → line 216 |
| 7 | &nbsp;&nbsp;for each var v adj. to u | `for j < bfufChkDeg: v = bpChkVarsData[c][j]` → line 160–161 | `for j < BB_CHK_DEG: vn = BFUF_VAR_NODE(bb_chk_vars[c][j])` → line 218–219 |
| 8 | &nbsp;&nbsp;&nbsp;&nbsp;union(u, v) | `uf.unite(node, vn)` → line 163 | `uf_unite(tid, node, vn)` → line 220 |
| 9 | &nbsp;&nbsp;&nbsp;&nbsp;if v not visited: enqueue(Q, v) | `if !visited[vn] { queue append }` → line 164–167 | `if (!visited[tid][vn]) { bfs_q[tid][qt++]=vn }` → line 221–224 |
| 10 | **else** u is variable node | `else` → line 169 | `else` → line 226 |
| 11 | &nbsp;&nbsp;for each check c adj. to u | `for j < bfufVarDeg: cn = bfufVarChks[v][j]` → line 171–172 | `for j < BB_VAR_DEG: cn = bb_var_chks[v][j]` → line 228–229 |
| 12 | &nbsp;&nbsp;&nbsp;&nbsp;union(u, c) | `uf.unite(node, cn)` → line 173 | `uf_unite(tid, node, cn)` → line 230 |
| 13 | &nbsp;&nbsp;&nbsp;&nbsp;if c not visited: enqueue(Q, c) | `if !visited[cn] { queue append }` → line 174–178 | `if (!visited[tid][cn]) { bfs_q[tid][qt++]=cn }` → line 231–234 |
| 14 | **for each** cluster C_k (UF component) | `for _, node := range queue; root := uf.find(node)` → line 185–191 | `for qi < qt; root = uf_find(tid,node)` → line 242–248 |
| 15 | &nbsp;&nbsp;Collect checks ∩ C_k | `for c < bfufNC: if visited[c] && uf.find(c)==root` → line 203–206 | `for c < BB_NC: if visited[tid][c] && uf_find(tid,c)==root` → line 254–256 |
| 16 | &nbsp;&nbsp;Collect vars ∩ C_k | `for v < bfufNV: if visited[vn] && uf.find(vn)==root` → line 207–214 | `for v < BB_NV: if visited[tid][vn] && uf_find(tid,vn)==root` → line 258–263 |
| 17 | &nbsp;&nbsp;Build H_cl, s_cl | `HCl[r][clVarIdx[v]] = 1; sCl[r] = syn[c]` → line 222–237 | `gf2_mat[tid][r][col>>5] |= 1u<<(col&31)` → line 269–285 |
| 18 | &nbsp;&nbsp;ê_cl ← GF2_solve(H_cl, s_cl) | `e := gf2SolveBfuf(HCl, sCl, ncCl, nvCl)` → line 239 | `gf2_solve(tid, nc_cl, nv_cl)` → line 289 |
| 19 | &nbsp;&nbsp;ê[C_k ∩ V] ^= ê_cl | `corr[v] ^= e[col]` → line 241–243 | `corr[cl_vars[tid][col]] ^= e_cl[tid][col]` → line 290–292 |

---

## Algorithm 2: GF(2) Gaussian Elimination

| Step | Paper Pseudocode | Go: `bfuf_verify.go` | C: `task.c` |
|------|-----------------|----------------------|-------------|
| 1 | Build augmented matrix A = [H_cl \| s_cl] | `mat[r] = H[r] ++ [s[r]]` → line 81–86 | rows packed as uint32; RHS at bit `nv_cl` → line 269–285 |
| 2 | prow ← 0 | `prow := 0` → line 88 | `int prow = 0` → line 134 |
| 3 | **for** col = 0 to nv_cl | `for col := 0; col < nv && prow < nc` → line 94 | `for (int col = 0; col < nv && prow < nc; col++)` → line 140 |
| 4 | &nbsp;&nbsp;Find pivot row r ≥ prow with A[r][col]=1 | `for r := prow; r < nc; if mat[r][col]!=0` → line 96–100 | `for (r = prow; r < nc; if mat[r][word]&mask)` → line 145–149 |
| 5 | &nbsp;&nbsp;**if** no pivot: **continue** | `if pivot < 0 { continue }` → line 103 | `if (pivot < 0) continue` → line 149 |
| 6 | &nbsp;&nbsp;Swap A[prow] ↔ A[r] | `mat[prow], mat[pivot] = mat[pivot], mat[prow]` → line 107 | `for w: swap mat[prow][w], mat[pivot][w]` → line 152–158 |
| 7 | &nbsp;&nbsp;**for** all rows r' ≠ prow with A[r'][col]=1 | `for r := 0; r < nc; if r!=prow && mat[r][col]!=0` → line 110–111 | `for (r = 0; r < nc; if r!=prow && mat[r][word]&mask)` → line 162–163 |
| 8 | &nbsp;&nbsp;&nbsp;&nbsp;A[r'] ^= A[prow] | `for c: mat[r][c] ^= mat[prow][c]` → line 112–114 | `for w: mat[r][w] ^= mat[prow][w]` → line 164–165 |
| 9 | &nbsp;&nbsp;prow++ | `prow++` → line 117 | `prow++` → line 169 |
| 10 | Consistency check: no solution if zero row with RHS=1 | `for r := prow; mat[r][nv]!=0 → return nil` → line 121–125 | `for r=prow; if mat[r][aug_word]&aug_mask → return 0` → line 172–175 |
| 11 | Extract solution (free vars = 0) | `e[pivotCol[r]] = mat[r][nv]` → line 128–134 | `e[pcol[r]] = (mat[r][aug_word]&aug_mask)?1:0` → line 178–184 |

---

## Key Implementation Differences (Go vs C)

| Aspect | Go (`bfuf_verify.go`) | C (`task.c`) |
|--------|----------------------|--------------|
| Matrix representation | `[][]uint8` byte-per-element | `uint32_t[GF2_WORDS]` bit-packed (32-bit DPU) |
| UF types | `uint8` indices | `uint8` indices |
| Parallelism | Single goroutine | 11 tasklets (one syndrome each), barrier sync |
| `varChks` init | `init()` at startup | tasklet 0 builds at runtime, barrier before use |
| MRAM I/O | N/A (in-memory) | `mram_read` / `mram_write` for syndrome & correction |
| Correctness link | Reference implementation | Verified identical to Go via uPIMulator (7/7 PASS) |

---

## Correctness Evidence

1. **Syndrome satisfaction**: `H * corr == σ` for all decoded frames  
   → verified in `trace_bfuf.go` (field `H*corr == σ`)

2. **Valid codeword**: `H * (error ⊕ corr) == 0` (residual syndrome is zero)  
   → verified in `trace_bfuf.go` (field `H*(error⊕corr)==0`)

3. **DPU == CPU**: uPIMulator cycle-accurate simulation produces identical corrections  
   → `run_bfuf_bench.sh` Step 2: all 7 error rates report `PASS`
