import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

fig = plt.figure(figsize=(18, 13))
fig.patch.set_facecolor('#F8F9FA')

# ── 색상 정의 ──────────────────────────────────────────────
C_BP   = '#D94F3D'   # 기존 BP: 빨강
C_NMS  = '#2E86AB'   # NMS: 파랑
C_DPU  = '#27AE60'   # DPU 친화: 초록
C_WARN = '#E67E22'   # 경고/비용: 주황
C_BG   = '#FFFFFF'
C_HEAD = '#2C3E50'

ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 18)
ax.set_ylim(0, 13)
ax.axis('off')

def box(ax, x, y, w, h, color, alpha=1.0, radius=0.3, lw=1.5, edgecolor=None):
    ec = edgecolor if edgecolor else color
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0.05,rounding_size={radius}",
                       facecolor=color, edgecolor=ec,
                       linewidth=lw, alpha=alpha, zorder=3)
    ax.add_patch(p)

def txt(ax, x, y, s, fs=10, color='black', ha='center', va='center', bold=False, zorder=4):
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va=va,
            fontweight=weight, zorder=zorder,
            fontfamily='DejaVu Sans')

def arrow(ax, x1, y1, x2, y2, color='#555555', lw=1.5, style='->', zorder=3):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle='arc3,rad=0.0'),
                zorder=zorder)

# ═══════════════════════════════════════════════════════════
# 제목
# ═══════════════════════════════════════════════════════════
box(ax, 0.3, 11.8, 17.4, 0.9, C_HEAD, radius=0.4)
txt(ax, 9, 12.25,
    'Why Normalized Min-Sum BP Fits UPMEM DPU Architecture',
    fs=16, color='white', bold=True)

# ═══════════════════════════════════════════════════════════
# 왼쪽 헤더: Standard BP
# ═══════════════════════════════════════════════════════════
box(ax, 0.4, 10.5, 7.8, 0.9, C_BP, radius=0.3)
txt(ax, 4.3, 10.95, 'Standard Belief Propagation  (float32)', fs=13, color='white', bold=True)

# 오른쪽 헤더: NMS on DPU
box(ax, 9.8, 10.5, 7.8, 0.9, C_NMS, radius=0.3)
txt(ax, 13.7, 10.95, 'Normalized Min-Sum on DPU  (int16)', fs=13, color='white', bold=True)

# 중앙 구분선
ax.plot([9.0, 9.0], [0.4, 11.6], color='#CCCCCC', lw=2, ls='--', zorder=2)

# ═══════════════════════════════════════════════════════════
# STEP 1 — 데이터 표현
# ═══════════════════════════════════════════════════════════
y1 = 9.3
box(ax, 0.4, y1, 7.8, 1.0, C_BP, alpha=0.08, edgecolor=C_BP, lw=1.5)
box(ax, 9.8, y1, 7.8, 1.0, C_NMS, alpha=0.08, edgecolor=C_NMS, lw=1.5)

txt(ax, 0.9, y1+0.75, 'STEP 1', fs=8, color='#888888', ha='left')
txt(ax, 10.3, y1+0.75, 'STEP 1', fs=8, color='#888888', ha='left')

txt(ax, 4.3, y1+0.55, 'LLR representation:  float32', fs=11, color=C_BP, bold=True)
txt(ax, 4.3, y1+0.2,
    '4 bytes/value  →  WRAM 64KB holds ~8,000 values\n'
    'Requires hardware FPU for every operation',
    fs=9.5, color='#555555')

txt(ax, 13.7, y1+0.55, r'LLR → int16   ( ×256 fixed-point )', fs=11, color=C_NMS, bold=True)
txt(ax, 13.7, y1+0.2,
    '2 bytes/value  →  WRAM holds 2× more messages\n'
    'Integer-only: DPU ALU handles natively  ✓',
    fs=9.5, color='#27AE60')

# ═══════════════════════════════════════════════════════════
# STEP 2 — Check Node Update
# ═══════════════════════════════════════════════════════════
y2 = 7.5
box(ax, 0.4, y2, 7.8, 1.6, C_BP, alpha=0.08, edgecolor=C_BP, lw=1.5)
box(ax, 9.8, y2, 7.8, 1.6, C_NMS, alpha=0.08, edgecolor=C_NMS, lw=1.5)

txt(ax, 0.9, y2+1.45, 'STEP 2', fs=8, color='#888888', ha='left')
txt(ax, 10.3, y2+1.45, 'STEP 2', fs=8, color='#888888', ha='left')

txt(ax, 4.3, y2+1.25, 'Check Node Update', fs=11, color=C_BP, bold=True)
txt(ax, 4.3, y2+0.88,
    r'$r_{c \to v} = 2 \cdot \tanh^{-1}\!\!\left(\prod_{v\' \neq v} \tanh\!\left(\frac{q_{v\' \to c}}{2}\right)\right)$',
    fs=11, color='#333333')
# 비용 표시
box(ax, 0.6, y2+0.05, 7.4, 0.55, C_WARN, alpha=0.15, edgecolor=C_WARN, lw=1)
txt(ax, 4.3, y2+0.33,
    '⚠  tanh / atanh : transcendental functions\n'
    '   No hardware support on DPU  →  SW emulation, very slow',
    fs=9, color=C_WARN)

txt(ax, 13.7, y2+1.25, 'Check Node Update  (Min-Sum approx.)', fs=11, color=C_NMS, bold=True)
txt(ax, 13.7, y2+0.88,
    r'$r_{c \to v} \approx s \cdot \alpha \cdot \min_{v\' \neq v}|q_{v\' \to c}|$',
    fs=11, color='#333333')
box(ax, 10.0, y2+0.05, 7.4, 0.55, C_DPU, alpha=0.15, edgecolor=C_DPU, lw=1)
txt(ax, 13.7, y2+0.33,
    '✓  Only compare ( min ) + sign XOR\n'
    '   Integer compare: single-cycle on DPU pipeline',
    fs=9, color=C_DPU)

# ═══════════════════════════════════════════════════════════
# STEP 3 — Alpha Scaling
# ═══════════════════════════════════════════════════════════
y3 = 5.7
box(ax, 0.4, y3, 7.8, 1.5, C_BP, alpha=0.08, edgecolor=C_BP, lw=1.5)
box(ax, 9.8, y3, 7.8, 1.5, C_NMS, alpha=0.08, edgecolor=C_NMS, lw=1.5)

txt(ax, 0.9, y3+1.35, 'STEP 3', fs=8, color='#888888', ha='left')
txt(ax, 10.3, y3+1.35, 'STEP 3', fs=8, color='#888888', ha='left')

txt(ax, 4.3, y3+1.15, 'Alpha Scaling  ( α = 0.625 )', fs=11, color=C_BP, bold=True)
txt(ax, 4.3, y3+0.78, r'$\alpha \cdot x \;=\; 0.625 \times x$', fs=11, color='#333333')
box(ax, 0.6, y3+0.05, 7.4, 0.55, C_WARN, alpha=0.15, edgecolor=C_WARN, lw=1)
txt(ax, 4.3, y3+0.33,
    '⚠  Float multiplication per message\n'
    '   DPU has no FPU  →  each mul costs many cycles',
    fs=9, color=C_WARN)

txt(ax, 13.7, y3+1.15, 'Alpha Scaling  →  Shift + Add', fs=11, color=C_NMS, bold=True)
txt(ax, 13.7, y3+0.78,
    r'$0.625 \cdot x \;=\; \frac{(x \ll 2) + x}{8} \;=\; ((x\!<\!<\!2)+x)\!>\!>\!3$',
    fs=11, color='#333333')
box(ax, 10.0, y3+0.05, 7.4, 0.55, C_DPU, alpha=0.15, edgecolor=C_DPU, lw=1)
txt(ax, 13.7, y3+0.33,
    '✓  Bit-shift + integer add: 2 instructions\n'
    '   No FPU needed  →  fully pipelined on DPU',
    fs=9, color=C_DPU)

# ═══════════════════════════════════════════════════════════
# STEP 4 — Variable Node Update
# ═══════════════════════════════════════════════════════════
y4 = 4.1
box(ax, 0.4, y4, 7.8, 1.3, C_BP, alpha=0.08, edgecolor=C_BP, lw=1.5)
box(ax, 9.8, y4, 7.8, 1.3, C_NMS, alpha=0.08, edgecolor=C_NMS, lw=1.5)

txt(ax, 0.9, y4+1.15, 'STEP 4', fs=8, color='#888888', ha='left')
txt(ax, 10.3, y4+1.15, 'STEP 4', fs=8, color='#888888', ha='left')

txt(ax, 4.3, y4+0.95, 'Variable Node Update', fs=11, color=C_BP, bold=True)
txt(ax, 4.3, y4+0.62,
    r'$q_{v \to c} = \lambda_0 + \sum_{c\' \neq c} r_{c\' \to v}$   (float add)',
    fs=11, color='#333333')
box(ax, 0.6, y4+0.05, 7.4, 0.42, C_WARN, alpha=0.15, edgecolor=C_WARN, lw=1)
txt(ax, 4.3, y4+0.27,
    '⚠  Float accumulation  →  rounding error accumulates',
    fs=9, color=C_WARN)

txt(ax, 13.7, y4+0.95, 'Variable Node Update', fs=11, color=C_NMS, bold=True)
txt(ax, 13.7, y4+0.62,
    r'$q_{v \to c} = \lambda_0 + c_0 + c_1 + c_2$   (int16 add, deg=3)',
    fs=11, color='#333333')
box(ax, 10.0, y4+0.05, 7.4, 0.42, C_DPU, alpha=0.15, edgecolor=C_DPU, lw=1)
txt(ax, 13.7, y4+0.27,
    '✓  3 integer additions  →  3 cycles, no accumulation error',
    fs=9, color=C_DPU)

# ═══════════════════════════════════════════════════════════
# 하단 요약 바
# ═══════════════════════════════════════════════════════════
y5 = 0.4
box(ax, 0.4, y5, 17.2, 3.4, '#EAF4FB', edgecolor=C_NMS, lw=2, radius=0.4)
txt(ax, 9.0, y5+3.05, 'DPU Architecture Fit Summary', fs=13, color=C_HEAD, bold=True)

cols = [
    (1.6,  'No FPU Required',
           'DPU has integer ALU only.\nNMS uses compare + shift + add.\nAll operations native to DPU.', C_DPU),
    (5.5,  'WRAM Efficiency',
           'int16 (2B) vs float32 (4B)\n→ 2× more LLR messages fit\nin 64KB WRAM per DPU.', C_NMS),
    (9.4,  'Pipeline Friendly',
           'Fixed-length operations per\niteration → no branch stall,\nfully pipelined execution.', C_NMS),
    (13.3, 'Deterministic Latency',
           'Fixed iteration count +\ninteger ops → cycle-exact\nexecution, no jitter.', C_DPU),
]

for cx, title, body, color in cols:
    box(ax, cx-0.9, y5+0.2, 3.5, 2.6, color, alpha=0.12, edgecolor=color, lw=1.5, radius=0.3)
    txt(ax, cx+0.85, y5+2.5, title, fs=10, color=color, bold=True)
    txt(ax, cx+0.85, y5+1.4, body, fs=8.8, color='#333333')

plt.savefig(r'C:\Users\wjdrm\bp_bench_pkg\nms_dpu_fit.png',
            dpi=150, bbox_inches='tight', facecolor='#F8F9FA')
print('Saved: nms_dpu_fit.png')
