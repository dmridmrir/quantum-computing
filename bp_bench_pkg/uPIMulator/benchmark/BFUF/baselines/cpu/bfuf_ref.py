"""
bfuf_ref.py - Python reference: BF-UF decoder for [[144,12,12]] BB-QLDPC

Ref: Löbl et al., "Breadth-first graph traversal union-find decoder",
     arXiv:2407.15988 (2024)

Phase 1: BFS from unsatisfied check nodes, merging clusters via Union-Find.
Phase 2: Per-cluster GF(2) Gaussian elimination to find correction.

Usage: python bfuf_ref.py
"""

import numpy as np
from collections import deque
import time
import random

# ------------------------------------------------------------------ #
# [[144,12,12]] BB-QLDPC parity check matrix                         #
# ------------------------------------------------------------------ #
NC, NV, CHK_DEG, VAR_DEG = 72, 144, 6, 3

BB_CHK_VARS = [
    [3,6,12,73,74,90],   [4,7,13,74,75,91],   [5,8,14,75,76,92],
    [0,9,15,76,77,93],   [1,10,16,72,77,94],  [2,11,17,72,73,95],
    [9,12,18,79,80,96],  [10,13,19,80,81,97], [11,14,20,81,82,98],
    [6,15,21,82,83,99],  [7,16,22,78,83,100], [8,17,23,78,79,101],
    [15,18,24,85,86,102],[16,19,25,86,87,103],[17,20,26,87,88,104],
    [12,21,27,88,89,105],[13,22,28,84,89,106],[14,23,29,84,85,107],
    [21,24,30,91,92,108],[22,25,31,92,93,109],[23,26,32,93,94,110],
    [18,27,33,94,95,111],[19,28,34,90,95,112],[20,29,35,90,91,113],
    [27,30,36,97,98,114],[28,31,37,98,99,115],[29,32,38,99,100,116],
    [24,33,39,100,101,117],[25,34,40,96,101,118],[26,35,41,96,97,119],
    [33,36,42,103,104,120],[34,37,43,104,105,121],[35,38,44,105,106,122],
    [30,39,45,106,107,123],[31,40,46,102,107,124],[32,41,47,102,103,125],
    [39,42,48,109,110,126],[40,43,49,110,111,127],[41,44,50,111,112,128],
    [36,45,51,112,113,129],[37,46,52,108,113,130],[38,47,53,108,109,131],
    [45,48,54,115,116,132],[46,49,55,116,117,133],[47,50,56,117,118,134],
    [42,51,57,118,119,135],[43,52,58,114,119,136],[44,53,59,114,115,137],
    [51,54,60,121,122,138],[52,55,61,122,123,139],[53,56,62,123,124,140],
    [48,57,63,124,125,141],[49,58,64,120,125,142],[50,59,65,120,121,143],
    [57,60,66,72,127,128],[58,61,67,73,128,129],[59,62,68,74,129,130],
    [54,63,69,75,130,131],[55,64,70,76,126,131],[56,65,71,77,126,127],
    [0,63,66,78,133,134], [1,64,67,79,134,135], [2,65,68,80,135,136],
    [3,60,69,81,136,137], [4,61,70,82,132,137], [5,62,71,83,132,133],
    [0,6,69,84,139,140],  [1,7,70,85,140,141],  [2,8,71,86,141,142],
    [3,9,66,87,142,143],  [4,10,67,88,138,143], [5,11,68,89,138,139],
]

BB_VAR_MAP = [
    [18,360,396],[24,366,402],[30,372,408],[0,378,414],[6,384,420],
    [12,390,426],[1,54,397],[7,60,403],[13,66,409],[19,36,415],
    [25,42,421],[31,48,427],[2,37,90],[8,43,96],[14,49,102],
    [20,55,72],[26,61,78],[32,67,84],[38,73,126],[44,79,132],
    [50,85,138],[56,91,108],[62,97,114],[68,103,120],[74,109,162],
    [80,115,168],[86,121,174],[92,127,144],[98,133,150],[104,139,156],
    [110,145,198],[116,151,204],[122,157,210],[128,163,180],[134,169,186],
    [140,175,192],[146,181,234],[152,187,240],[158,193,246],[164,199,216],
    [170,205,222],[176,211,228],[182,217,270],[188,223,276],[194,229,282],
    [200,235,252],[206,241,258],[212,247,264],[218,253,306],[224,259,312],
    [230,265,318],[236,271,288],[242,277,294],[248,283,300],[254,289,342],
    [260,295,348],[266,301,354],[272,307,324],[278,313,330],[284,319,336],
    [290,325,379],[296,331,385],[302,337,391],[308,343,361],[314,349,367],
    [320,355,373],[326,362,416],[332,368,422],[338,374,428],[344,380,398],
    [350,386,404],[356,392,410],[27,33,327],[3,34,333],[4,9,339],
    [10,15,345],[16,21,351],[22,28,357],[63,69,363],[39,70,369],
    [40,45,375],[46,51,381],[52,57,387],[58,64,393],[99,105,399],
    [75,106,405],[76,81,411],[82,87,417],[88,93,423],[94,100,429],
    [5,135,141],[11,111,142],[17,112,117],[23,118,123],[29,124,129],
    [35,130,136],[41,171,177],[47,147,178],[53,148,153],[59,154,159],
    [65,160,165],[71,166,172],[77,207,213],[83,183,214],[89,184,189],
    [95,190,195],[101,196,201],[107,202,208],[113,243,249],[119,219,250],
    [125,220,225],[131,226,231],[137,232,237],[143,238,244],[149,279,285],
    [155,255,286],[161,256,261],[167,262,267],[173,268,273],[179,274,280],
    [185,315,321],[191,291,322],[197,292,297],[203,298,303],[209,304,309],
    [215,310,316],[221,352,358],[227,328,359],[233,329,334],[239,335,340],
    [245,341,346],[251,347,353],[257,388,394],[263,364,395],[269,365,370],
    [275,371,376],[281,377,382],[287,383,389],[293,424,430],[299,400,431],
    [305,401,406],[311,407,412],[317,413,418],[323,419,425],
]

# variable → list of adjacent check nodes
VAR_CHKS = [[e // CHK_DEG for e in row] for row in BB_VAR_MAP]

# ------------------------------------------------------------------ #
# Union-Find                                                          #
# ------------------------------------------------------------------ #
class UF:
    def __init__(self, n):
        self.par = list(range(n))
        self.rnk = [0] * n

    def find(self, x):
        while self.par[x] != x:
            self.par[x] = self.par[self.par[x]]
            x = self.par[x]
        return x

    def unite(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.rnk[a] < self.rnk[b]:
            a, b = b, a
        self.par[b] = a
        if self.rnk[a] == self.rnk[b]:
            self.rnk[a] += 1

# ------------------------------------------------------------------ #
# GF(2) Gaussian elimination                                          #
# ------------------------------------------------------------------ #
def gf2_solve(H, s):
    """
    Solve H * e = s over GF(2).
    H: (nc, nv) numpy bool array
    s: (nc,) numpy bool array
    Returns e (nv,) bool array, or None if no solution.
    Free variables are set to 0.
    """
    nc, nv = H.shape
    # augmented matrix [H | s]
    mat = np.hstack([H.astype(np.uint8), s.reshape(-1, 1).astype(np.uint8)])

    pivot_row = 0
    pivot_col_of = {}   # row → pivot column

    for col in range(nv):
        if pivot_row >= nc:
            break
        # find pivot
        found = -1
        for r in range(pivot_row, nc):
            if mat[r, col]:
                found = r
                break
        if found < 0:
            continue
        # swap
        mat[[pivot_row, found]] = mat[[found, pivot_row]]
        pivot_col_of[pivot_row] = col
        # eliminate
        for r in range(nc):
            if r != pivot_row and mat[r, col]:
                mat[r] ^= mat[pivot_row]
        pivot_row += 1

    # consistency check
    for r in range(pivot_row, nc):
        if mat[r, nv]:
            return None   # no solution

    # extract solution
    e = np.zeros(nv, dtype=np.uint8)
    for r, col in pivot_col_of.items():
        e[col] = mat[r, nv]
    return e

# ------------------------------------------------------------------ #
# BF-UF decoder                                                       #
# ------------------------------------------------------------------ #
TOTAL = NC + NV   # 0..71 = checks, 72..215 = variables

def var_node(v): return NC + v
def get_var(n):  return n - NC

def bfuf_decode(syn):
    """
    syn  : list/array of NC bits (1 = unsatisfied check)
    returns corr: numpy array of NV bits (correction vector)
    """
    uf = UF(TOTAL)
    visited = [False] * TOTAL
    queue = deque()

    # seed BFS from unsatisfied checks
    for c in range(NC):
        if syn[c]:
            visited[c] = True
            queue.append(c)

    bfs_order = []

    # BFS on Tanner graph
    while queue:
        node = queue.popleft()
        bfs_order.append(node)

        if node < NC:                           # check node
            c = node
            for v in BB_CHK_VARS[c]:
                vn = var_node(v)
                uf.unite(node, vn)
                if not visited[vn]:
                    visited[vn] = True
                    queue.append(vn)
        else:                                   # variable node
            v = get_var(node)
            for c in VAR_CHKS[v]:
                uf.unite(node, c)
                if not visited[c]:
                    visited[c] = True
                    queue.append(c)

    corr = np.zeros(NV, dtype=np.uint8)
    done_roots = set()

    # decode each cluster
    for node in bfs_order:
        if node >= NC:
            continue   # only use check nodes to enumerate cluster roots
        root = uf.find(node)
        if root in done_roots:
            continue
        done_roots.add(root)

        # collect cluster members
        cl_checks = [c for c in range(NC) if visited[c] and uf.find(c) == root]
        cl_vars   = [v for v in range(NV) if visited[var_node(v)] and uf.find(var_node(v)) == root]

        if not cl_checks or not cl_vars:
            continue

        nv_cl = len(cl_vars)
        nc_cl = len(cl_checks)
        var_to_col = {v: i for i, v in enumerate(cl_vars)}

        # build H_cl and s_cl
        H_cl = np.zeros((nc_cl, nv_cl), dtype=np.uint8)
        s_cl = np.array([syn[c] for c in cl_checks], dtype=np.uint8)
        for r, c in enumerate(cl_checks):
            for v in BB_CHK_VARS[c]:
                if v in var_to_col:
                    H_cl[r, var_to_col[v]] = 1

        e = gf2_solve(H_cl, s_cl)
        if e is not None:
            for col, v in enumerate(cl_vars):
                corr[v] ^= e[col]

    return corr

# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #
def compute_syndrome(bits):
    return np.array([
        int(np.sum(bits[BB_CHK_VARS[c]]) % 2) for c in range(NC)
    ], dtype=np.uint8)

def verify(syn, corr):
    """Check H * corr == syn (mod 2)."""
    return np.array_equal(compute_syndrome(corr), syn)

# ------------------------------------------------------------------ #
# Monte Carlo test                                                    #
# ------------------------------------------------------------------ #
def monte_carlo(p, n_trials=5000, seed=42):
    rng = np.random.default_rng(seed)
    fail = 0
    for _ in range(n_trials):
        error = (rng.random(NV) < p).astype(np.uint8)
        syn   = compute_syndrome(error)
        corr  = bfuf_decode(syn)
        if not verify(syn, corr):
            fail += 1
    return fail, n_trials

if __name__ == "__main__":
    print("BF-UF decoder -- [[144,12,12]] BB-QLDPC (BSC)\n")
    print(f"{'p':>6}  {'FER':>10}  {'fail/total':>12}  {'time(s)':>8}")
    print(f"{'---':>6}  {'---':>10}  {'---':>12}  {'---':>8}")

    N = 2000
    for p in [0.01, 0.03, 0.05, 0.07, 0.10]:
        t0 = time.time()
        fail, total = monte_carlo(p, n_trials=N)
        elapsed = time.time() - t0
        print(f"{p:>6.3f}  {fail/total:>10.4f}  {fail:>5}/{total:<6}  {elapsed:>8.2f}")
