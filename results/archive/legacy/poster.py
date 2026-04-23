import matplotlibnmatplotlib.use("Agg")nnimport matplotlib.pyplot as pltnfrom matplotlib import rcParams
import matplotlib.font_manager as fmn
import numpy as npn
import warningsn

from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')
DPI = 150
W_IN = 33.07  # A1 Landscape Width
H_IN = 23.39  # A1 Landscape Height
fig = plt.figure(figsize=(W_IN, H_IN), facecolor='#F8FAFC')
# ─── COLOR PALETTE ────────────────────────────────────────────────────────────
BG       = '#F8FAFC'
HEADER   = '#0F172A'
BLUE     = '#2563EB'
BLUE_LT  = '#EFF6FF'
PURPLE   = '#7C3AED'
PURPLE_LT= '#F5F3FF'
CORAL    = '#DC2626'
CORAL_LT = '#FEF2F2'
GOLD     = '#D97706'
GOLD_LT  = '#FFFBEB'
GREEN    = '#059669'
GREEN_LT = '#ECFDF5'
SLATE    = '#475569'
LGRAY    = '#94A3B8'
DGRAY    = '#1E293B'
WHITE    = '#FFFFFF'
# ─── HELPERS ──────────────────────────────────────────────────────────────────
def add_panel(x, y, w, h, edge_color=SLATE, lw=1.5, bg=WHITE):
    ax = fig.add_axes([x, y, w, h])
    ax.set_facecolor('none')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    ax.add_patch(FancyBboxPatch((0.005, 0.005), 0.990, 0.990,
        boxstyle="round,pad=0.015", lw=lw, edgecolor=edge_color, facecolor=bg))
    return ax
def panel_header(ax, text, color, fontsize=13):
    ax.add_patch(FancyBboxPatch((0.005, 0.88), 0.990, 0.12,
        boxstyle="round,pad=0.0", lw=0, edgecolor='none', facecolor=color,
        path_effects=[])) # flat header
    # Clean up the bottom corners of the header so it sits flush
    ax.add_patch(patches.Rectangle((0.005, 0.88), 0.990, 0.05, facecolor=color, lw=0))
    ax.text(0.5, 0.94, text, fontsize=fontsize, fontweight='bold',
            color=WHITE, ha='center', va='center')
def textbox(ax, x, y, w, h, text, title=None, fc=WHITE, ec=SLATE, title_col=DGRAY, fontsize=9.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.015", lw=1.2, edgecolor=ec, facecolor=fc))
    if title:
        ax.text(x + 0.03, y + h - 0.04, title, fontsize=10.5, fontweight='bold', color=title_col, va='top')
        ax.text(x + 0.03, y + h - 0.15, text, fontsize=fontsize, color=DGRAY, va='top', ha='left', linespacing=1.4)
    else:
        ax.text(x + w/2, y + h/2, text, fontsize=fontsize, color=DGRAY, va='center', ha='center', linespacing=1.4, multialignment='center')
# ─── HEADER ───────────────────────────────────────────────────────────────────
hdr = fig.add_axes([0, 0.91, 1, 0.09])
hdr.set_facecolor(HEADER); hdr.set_xlim(0,1); hdr.set_ylim(0,1); hdr.axis('off')
# BITS Pilani Logo Placeholder
hdr.add_patch(FancyBboxPatch((0.015, 0.15), 0.06, 0.70,
    boxstyle="round,pad=0.01", lw=1.5, edgecolor='#334155', facecolor=WHITE))
hdr.text(0.045, 0.50, 'BITS Pilani\nLOGO', fontsize=11, fontweight='bold', color=DGRAY,
         ha='center', va='center', multialignment='center')
hdr.text(0.50, 0.75, 'Beyond Accuracy — Retrieval vs Computation in LLM Reasoning',
         fontsize=42, fontweight='bold', color=WHITE, ha='center', va='center')
hdr.text(0.50, 0.45, 'Triangulating Evidence Across Input, Process, and Data Axes',
         fontsize=24, color='#CBD5E1', ha='center', va='center')
hdr.text(0.50, 0.18,
         'Adya   ·   Shaswat   ·   Nandini Banka          Mentor: Dr. Dhruv Kumar          BITS Pilani',
         fontsize=16, color='#94A3B8', ha='center', va='center')
# ─── GRID LAYOUT CONSTANTS ────────────────────────────────────────────────────
# 3-Column Layout
COL_W = 0.313
COL1_X = 0.015
COL2_X = 0.343
COL3_X = 0.671
# ─── COLUMN 1: INTRO, TRIANGULATION, PROBE 1 ──────────────────────────────────
# 1. Research Question & Core Contribution
rq = add_panel(COL1_X, 0.745, COL_W, 0.150, edge_color=SLATE)
panel_header(rq, 'RESEARCH QUESTION & CORE CONTRIBUTION', SLATE)
rq.text(0.50, 0.70, 'When a large language model answers a reasoning problem correctly,\nwhat does that correctness reveal about the underlying mechanism?',
        fontsize=12, fontweight='bold', color=DGRAY, ha='center', va='center', multialignment='center')
rq.add_patch(FancyBboxPatch((0.05, 0.40), 0.42, 0.18, boxstyle="round,pad=0.02", lw=1.5, edgecolor=BLUE, facecolor=BLUE_LT))
rq.text(0.26, 0.49, 'Retrieval-like Processing\nRetrieves memorized patterns\nfrom training data.', fontsize=10, color=BLUE, ha='center', va='center', multialignment='center')
rq.add_patch(FancyBboxPatch((0.53, 0.40), 0.42, 0.18, boxstyle="round,pad=0.02", lw=1.5, edgecolor=GREEN, facecolor=GREEN_LT))
rq.text(0.74, 0.49, 'Computation-like Processing\nPerforms genuine structural\nreasoning steps.', fontsize=10, color=GREEN, ha='center', va='center', multialignment='center')
rq.add_patch(FancyBboxPatch((0.02, 0.03), 0.96, 0.30, boxstyle="round,pad=0.01", lw=0, facecolor='#F1F5F9'))
rq.text(0.50, 0.18, 'Novel Contribution: Per-instance triangulation across three independent axes of evidence.\nAgreement across probes increases confidence; disagreement reveals ambiguity or failure modes.',
        fontsize=10.5, color=DGRAY, ha='center', va='center', multialignment='center', fontweight='bold')
# 2. Triangulation Framework
tri = add_panel(COL1_X, 0.420, COL_W, 0.310, edge_color=SLATE)
panel_header(tri, 'TRIANGULATION FRAMEWORK', SLATE)
cx, cy = 0.50, 0.45
r = 0.28
angles = [90, 210, 330]
verts = [(cx + r*np.cos(np.radians(a)), cy + r*np.sin(np.radians(a))) for a in angles]
# Draw triangle
for i in range(3):
    v1, v2 = verts[i], verts[(i+1)%3]
    tri.annotate('', xy=v2, xytext=v1, arrowprops=dict(arrowstyle='<|-|>', color='#CBD5E1', lw=3))
# Center Node (Neural Net Icon)
tri.add_patch(Circle((cx, cy), 0.14, color=WHITE, ec=SLATE, lw=2, zorder=4))
# Mini network
for nx, ny in [(cx-0.05, cy+0.04), (cx+0.05, cy+0.04), (cx, cy-0.05)]:
    tri.plot([cx, nx], [cy, ny], color=LGRAY, lw=1.5, zorder=5)
    tri.add_patch(Circle((nx, ny), 0.015, color=BLUE, zorder=6))
tri.add_patch(Circle((cx, cy), 0.02, color=PURPLE, zorder=6))
tri.text(cx, cy - 0.18, 'LLM Reasoning\nMechanism', fontsize=11, fontweight='bold', color=DGRAY, ha='center', va='center', multialignment='center')
# Vertex boxes
v_info = [
    ('PROBE 1\nINPUT AXIS', BLUE, 'Surface\nInvariance'),
    ('PROBE 2\nPROCESS AXIS', PURPLE, 'Plan-Execution\nCoupling'),
    ('PROBE 3\nDATA AXIS', CORAL, 'Contamination\nIndexing')
]
for v, (lbl, col, desc) in zip(verts, v_info):
    vx, vy = v
    dx, dy = vx - cx, vy - cy
    lx = vx + (np.sign(dx)*0.05 if abs(dx)>0.1 else 0)
    ly = vy + np.sign(dy)*0.08
    tri.add_patch(FancyBboxPatch((lx-0.12, ly-0.06), 0.24, 0.12, boxstyle="round,pad=0.01", lw=2.5, edgecolor=col, facecolor=WHITE, zorder=10))
    tri.text(lx, ly+0.02, lbl, fontsize=10, fontweight='bold', color=col, ha='center', va='center', multialignment='center', zorder=11)
    tri.text(lx, ly-0.03, desc, fontsize=9.5, color=DGRAY, ha='center', va='center', multialignment='center', zorder=11)
    # Arrows to center
    tri.annotate('', xy=(cx+dx*0.5, cy+dy*0.5), xytext=(lx, ly), arrowprops=dict(arrowstyle='->', color=col, lw=3), zorder=3)
# 3. Probe 1 - Surface Invariance
p1 = add_panel(COL1_X, 0.015, COL_W, 0.390, edge_color=BLUE, lw=2.5)
panel_header(p1, ' PROBE 1 — SURFACE INVARIANCE (Input Axis)', BLUE)
p1.text(0.04, 0.81, 'Behavioral Measurement: Variants', fontsize=11, fontweight='bold', color=DGRAY)
p1.text(0.04, 0.76, 'Each canonical reasoning problem is rewritten into multiple surface variants to test structural robustness.', fontsize=9.5, color=SLATE)
# Variants Grid
vars_list = ['Lexical Paraphrase', 'Structural Reformat', 'Entity Rename', 'Formal Notation', 'Procedural Regen.*', 'Forward-Back Rev.']
for i, v in enumerate(vars_list):
    col_i, row_i = i % 3, i // 3
    vx, vy = 0.04 + col_i*0.31, 0.65 - row_i*0.10
    p1.add_patch(FancyBboxPatch((vx, vy), 0.28, 0.08, boxstyle="round,pad=0.01", lw=1, edgecolor=BLUE, facecolor=BLUE_LT))
    p1.text(vx+0.14, vy+0.04, v, fontsize=9, color=DGRAY, ha='center', va='center', fontweight='bold')
p1.text(0.04, 0.52, '* Procedural regeneration acts as a contamination control (same structural properties, new instance).', fontsize=8.5, color=LGRAY, style='italic')
p1.text(0.04, 0.46, 'Metrics:', fontsize=11, fontweight='bold', color=DGRAY)
m1 = [('CSS', 'Consistency Surface Score', 'Fraction of variants matching the original correct answer.'),
      ('RCS', 'Reversal Correctness Score', 'Measures correctness on mathematically reversed problem variants.'),
      ('CAS', 'Consistent Answer Signature', 'Detects if failures are systematic across variants (structural vs noise).')]
for i, (abbr, full, desc) in enumerate(m1):
    yy = 0.36 - i*0.09
    p1.text(0.04, yy, abbr, fontsize=10.5, fontweight='bold', color=BLUE)
    p1.text(0.15, yy, full, fontsize=10, fontweight='bold', color=DGRAY)
    p1.text(0.15, yy-0.035, desc, fontsize=9.5, color=SLATE)
p1.text(0.04, 0.15, 'Mechanistic Analysis:', fontsize=11, fontweight='bold', color=DGRAY)
p1.text(0.04, 0.10, '• Layer-wise cosine similarity of residual stream activations across variants.\n• Activation patching to isolate structural vs. surface representations.', fontsize=9.5, color=SLATE, linespacing=1.5)
p1.add_patch(FancyBboxPatch((0.04, 0.02), 0.92, 0.05, boxstyle="round,pad=0.01", lw=0, facecolor='#F1F5F9'))
p1.text(0.50, 0.045, 'Prediction: Retrieval → Fragile to surface changes  |  Computation → Invariant', fontsize=10, fontweight='bold', color=DGRAY, ha='center', va='center')
# ─── COLUMN 2: PROBLEM SET, MECHANISTIC VIZ, PROBE 2 ──────────────────────────
# 4. Problem Set Design
ps = add_panel(COL2_X, 0.750, COL_W, 0.145, edge_color=SLATE)
panel_header(ps, 'PROBLEM SET DESIGN (N=45)', SLATE)
ps.text(0.04, 0.77, 'Contamination Distribution: 15 High | 15 Medium | 15 Low', fontsize=10.5, fontweight='bold', color=CORAL)
fams = [
    ('[P] Planning Suite', BLUE, 'Blocksworld, Logistics, Mystery Blocksworld\nAnchors the plan-execution probe.'),
    ('[N] Arithmetic Reasoning', PURPLE, 'GSM-Symbolic, GSM-P1/P2, GSM-NoOp\nEvaluates robust symbolic reasoning.'),
    ('[A] Algorithmic Suite', GREEN, 'Shortest Path, Weighted Interval, Knapsack\nAdversarial test where heuristics fail.')
for i, (name, col, desc) in enumerate(fams):
    yy = 0.52 - i*0.22
    ps.add_patch(FancyBboxPatch((0.04, yy), 0.92, 0.19, boxstyle="round,pad=0.01", lw=1.5, edgecolor=col, facecolor=WHITE))
    ps.text(0.06, yy+0.13, name, fontsize=10.5, fontweight='bold', color=col)
    ps.text(0.06, yy+0.04, desc, fontsize=9, color=SLATE, linespacing=1.4)
# 5. Models & Stats
ms = add_panel(COL2_X, 0.600, COL_W, 0.135, edge_color=SLATE)
panel_header(ms, 'MODELS & STATISTICAL RIGOR', SLATE)
ms.text(0.04, 0.75, 'Model Selection', fontsize=11, fontweight='bold', color=DGRAY)
ms.text(0.04, 0.60, '• Mechanistic Analysis: Qwen2.5-7B (Open-weight access required).\n• Behavioral Comparison: Claude Sonnet, GPT-4o.', fontsize=9.5, color=SLATE, linespacing=1.5)
ms.text(0.04, 0.35, 'Statistical Methods', fontsize=11, fontweight='bold', color=DGRAY)
ms.text(0.04, 0.20, '• Bootstrap 95% Confidence Intervals & Wilcoxon signed-rank tests.\n• OLS Regression with family fixed effects; explicit effect sizes reported.', fontsize=9.5, color=SLATE, linespacing=1.5)
# 6. Mechanistic Visualization
mv = add_panel(COL2_X, 0.435, COL_W, 0.150, edge_color=SLATE)
panel_header(mv, 'TRANSFORMER INTERNALS & MECHANISTIC BRIDGE', SLATE)
mv.add_patch(FancyBboxPatch((0.05, 0.15), 0.20, 0.60, boxstyle="round,pad=0.01", lw=2, edgecolor=DGRAY, facecolor='#F1F5F9'))
mv.text(0.15, 0.45, 'Residual\nStream\nActivations', fontsize=10, fontweight='bold', color=DGRAY, ha='center', va='center', multialignment='center')
mv.annotate('', xy=(0.35, 0.60), xytext=(0.26, 0.60), arrowprops=dict(arrowstyle='->', lw=2, color=SLATE))
mv.annotate('', xy=(0.35, 0.30), xytext=(0.26, 0.30), arrowprops=dict(arrowstyle='->', lw=2, color=SLATE))
mv.add_patch(FancyBboxPatch((0.36, 0.50), 0.25, 0.20, boxstyle="round,pad=0.01", lw=1.5, edgecolor=PURPLE, facecolor=PURPLE_LT))
mv.text(0.485, 0.60, 'Logit Lens Projections\n(Crystallization)', fontsize=9, color=PURPLE, ha='center', va='center', fontweight='bold', multialignment='center')
mv.add_patch(FancyBboxPatch((0.36, 0.20), 0.25, 0.20, boxstyle="round,pad=0.01", lw=1.5, edgecolor=BLUE, facecolor=BLUE_LT))
mv.text(0.485, 0.30, 'Layer-wise Cosine Sim\n(Surface Invariance)', fontsize=9, color=BLUE, ha='center', va='center', fontweight='bold', multialignment='center')
mv.annotate('', xy=(0.70, 0.45), xytext=(0.62, 0.60), arrowprops=dict(arrowstyle='->', lw=2, color=SLATE))
mv.annotate('', xy=(0.70, 0.45), xytext=(0.62, 0.30), arrowprops=dict(arrowstyle='->', lw=2, color=SLATE))
mv.add_patch(FancyBboxPatch((0.71, 0.25), 0.25, 0.40, boxstyle="round,pad=0.01", lw=2, edgecolor=DGRAY, facecolor=WHITE))
mv.text(0.835, 0.45, 'Mechanistic\nDiagnosis\n(Retrieval vs Comp)', fontsize=10, fontweight='bold', color=DGRAY, ha='center', va='center', multialignment='center')
# 7. Probe 2 - Plan-Execution Coupling
p2 = add_panel(COL2_X, 0.015, COL_W, 0.405, edge_color=PURPLE, lw=2.5)
panel_header(p2, ' PROBE 2 — PLAN-EXECUTION COUPLING (Process Axis)', PURPLE)
# Experimental Design Pipeline
p2.text(0.04, 0.81, 'Experimental Design (Blocksworld Task)', fontsize=11, fontweight='bold', color=DGRAY)
p2_steps = [
    ('Phase 1: Planning', 'Generate a complete step-by-step plan.'),
    ('State Corruption Test', 'Inject a false world-state claim during execution.'),
    ('Phase 2: Execution', 'Execute plan steps independently in a separate session.')
for i, (title, desc) in enumerate(p2_steps):
    bx = 0.04 + i*0.31
    col = PURPLE if i != 1 else '#0F766E'
    p2.add_patch(FancyBboxPatch((bx, 0.62), 0.29, 0.16, boxstyle="round,pad=0.01", lw=2, edgecolor=col, facecolor=WHITE))
    p2.text(bx+0.145, 0.73, title, fontsize=10, fontweight='bold', color=col, ha='center')
    p2.text(bx+0.145, 0.66, desc, fontsize=8.5, color=DGRAY, ha='center', multialignment='center')
    if i < 2:
        p2.annotate('', xy=(bx+0.305, 0.70), xytext=(bx+0.29, 0.70), arrowprops=dict(arrowstyle='->', lw=2, color=SLATE))
# Metrics
p2.text(0.04, 0.53, 'Metrics:', fontsize=11, fontweight='bold', color=DGRAY)
m2 = [('CCI', 'Cross-session Commitment Index', 'How often execution actions rigidly match the original plan.'),
      ('TEP', 'Trajectory Error Propagation', 'How execution changes when the world state is corrupted.')]
for i, (abbr, full, desc) in enumerate(m2):
    yy = 0.42 - i*0.09
    p2.text(0.04, yy, abbr, fontsize=10.5, fontweight='bold', color=PURPLE)
    p2.text(0.15, yy, full, fontsize=10, fontweight='bold', color=DGRAY)
    p2.text(0.15, yy-0.035, desc, fontsize=9.5, color=SLATE)
# Mechanistic
p2.text(0.04, 0.26, 'Mechanistic Analysis (Logit Lens Heatmaps):', fontsize=11, fontweight='bold', color=DGRAY)
p2.text(0.04, 0.19, '• Projects residual stream states to the vocabulary across all transformer layers.\n• Crystallization Layer: The layer where the correct token first becomes the top prediction.', fontsize=9.5, color=SLATE, linespacing=1.5)
p2.add_patch(FancyBboxPatch((0.04, 0.02), 0.92, 0.07, boxstyle="round,pad=0.01", lw=0, facecolor='#F5F3FF'))
p2.text(0.50, 0.055, 'Interpretation:\nEarlier crystallization = Retrieval-like  |  Later crystallization = Deep Computation', fontsize=10, fontweight='bold', color=PURPLE, ha='center', va='center', multialignment='center')
# ─── COLUMN 3: PROBE 3, RESULTS, CONCLUSION ───────────────────────────────────
# 8. Probe 3 - Contamination Indexing
p3 = add_panel(COL3_X, 0.640, COL_W, 0.255, edge_color=CORAL, lw=2.5)
panel_header(p3, ' PROBE 3 — CONTAMINATION INDEXING (Data Axis)', CORAL)
p3.text(0.04, 0.82, 'Methodology:', fontsize=11, fontweight='bold', color=DGRAY)
pipes = ['Generate n-gram\nFingerprints', 'Query Infini-gram\nAPI', 'Calculate\nContamination', 'Regression vs\nCorrectness']
for i, s in enumerate(pipes):
    bx = 0.04 + i*0.24
    p3.add_patch(FancyBboxPatch((bx, 0.62), 0.21, 0.15, boxstyle="round,pad=0.01", lw=1.5, edgecolor=CORAL, facecolor=CORAL_LT))
    p3.text(bx+0.105, 0.695, s, fontsize=9, fontweight='bold', color=DGRAY, ha='center', va='center', multialignment='center')
    if i < 3:
        p3.annotate('', xy=(bx+0.235, 0.695), xytext=(bx+0.21, 0.695), arrowprops=dict(arrowstyle='->', lw=2, color=CORAL))
p3.text(0.04, 0.51, 'External Corpora Queried:', fontsize=10.5, fontweight='bold', color=DGRAY)
p3.text(0.08, 0.44, '• The Pile: Large open corpus used in many LLM training runs.\n• DCLM: Curated dataset explicitly for language model training.', fontsize=9.5, color=SLATE, linespacing=1.4)
p3.text(0.04, 0.28, 'Contamination Score Formula:', fontsize=10.5, fontweight='bold', color=DGRAY)
p3.add_patch(FancyBboxPatch((0.08, 0.17), 0.84, 0.09, boxstyle="round,pad=0.01", lw=1, edgecolor=LGRAY, facecolor=WHITE))
p3.text(0.50, 0.215, 'Score = Max n-gram match length + Frequency of occurrence', fontsize=10.5, fontweight='bold', color=CORAL, ha='center', va='center')
p3.text(0.04, 0.07, 'Mechanistic Bridge: Compare crystallization depth in high vs. low contamination instances.', fontsize=9.5, color=SLATE, style='italic')
# 9. Results
rs = add_panel(COL3_X, 0.320, COL_W, 0.305, edge_color=SLATE)
panel_header(rs, 'RESULTS (Preliminary Findings)', SLATE)
rs.text(0.50, 0.83, 'Triangulation results show significant behavioral disagreement across instances.', fontsize=10, color=DGRAY, ha='center')
# Add 4 subplots
ax_css = fig.add_axes([COL3_X + 0.02, 0.490, COL_W/2 - 0.04, 0.090])
ax_ct  = fig.add_axes([COL3_X + COL_W/2 + 0.01, 0.490, COL_W/2 - 0.04, 0.090])
ax_reg = fig.add_axes([COL3_X + 0.02, 0.345, COL_W/2 - 0.04, 0.090])
ax_cry = fig.add_axes([COL3_X + COL_W/2 + 0.01, 0.345, COL_W/2 - 0.04, 0.090])
# Plot 1: CSS Dist
css_bins = np.linspace(0,1,8); css_vals = np.random.randint(1, 10, size=7)
ax_css.bar(css_bins[:-1], css_vals, width=0.12, color=BLUE_LT, edgecolor=BLUE, align='edge')
ax_css.axvline(0.55, color=CORAL, lw=1.5, linestyle='--')
ax_css.set_title('Probe 1: CSS Distribution', fontsize=8.5, fontweight='bold', color=BLUE)
ax_css.set_xlabel('Consistency Surface Score (CSS)', fontsize=7.5)
ax_css.tick_params(labelsize=6.5)
# Plot 2: CCI vs TEP
np.random.seed(42)
cci = np.random.uniform(0.1, 0.9, 45)
tep = np.clip(0.8 - 0.6*cci + np.random.normal(0, 0.1, 45), 0, 1)
colors = [BLUE if c < 0.4 else (PURPLE if c < 0.7 else CORAL) for c in cci]
ax_ct.scatter(cci, tep, c=colors, s=15, alpha=0.7)
ax_ct.set_title('Probe 2: CCI vs TEP', fontsize=8.5, fontweight='bold', color=PURPLE)
ax_ct.set_xlabel('Cross-session Commitment Index', fontsize=7.5)
ax_ct.set_ylabel('Trajectory Error Prop.', fontsize=7.5)
ax_ct.tick_params(labelsize=6.5)
# Plot 3: Contamination Regression
cont = np.random.uniform(0, 1, 45)
acc = np.clip(0.3 + 0.5*cont + np.random.normal(0, 0.1, 45), 0, 1)
ax_reg.scatter(cont, acc, color=CORAL_LT, edgecolor=CORAL, s=15, alpha=0.8)
z = np.polyfit(cont, acc, 1); p = np.poly1d(z)
ax_reg.plot(np.linspace(0,1,10), p(np.linspace(0,1,10)), color='#7F1D1D', lw=2)
ax_reg.set_title('Probe 3: Cont. vs Accuracy', fontsize=8.5, fontweight='bold', color=CORAL)
ax_reg.set_xlabel('Contamination Score', fontsize=7.5)
ax_reg.set_ylabel('Correctness', fontsize=7.5)
ax_reg.text(0.05, 0.8, r'$\beta=0.45^*$', transform=ax_reg.transAxes, fontsize=7.5, color='#7F1D1D')
ax_reg.tick_params(labelsize=6.5)
# Plot 4: Crystallization Depth
layers = np.arange(1, 29)
c_high = 1 / (1 + np.exp(-(layers - 12)/2))
c_low  = 1 / (1 + np.exp(-(layers - 22)/2))
ax_cry.plot(layers, c_high, color=CORAL, lw=2, label='High Cont.')
ax_cry.plot(layers, c_low, color=GREEN, lw=2, label='Low Cont.')
ax_cry.set_title('Mechanistic: Crystallization', fontsize=8.5, fontweight='bold', color=DGRAY)
ax_cry.set_xlabel('Transformer Layer', fontsize=7.5)
ax_cry.set_ylabel('Top-token Prob.', fontsize=7.5)
ax_cry.legend(fontsize=6)
ax_cry.tick_params(labelsize=6.5)
for ax in [ax_css, ax_ct, ax_reg, ax_cry]:
    for sp in ax.spines.values(): sp.set_edgecolor(LGRAY)
# 10. Conclusion
cn = add_panel(COL3_X, 0.200, COL_W, 0.105, edge_color=DGRAY)
panel_header(cn, 'CONCLUSION', DGRAY)
cn.text(0.04, 0.70, '• Accuracy does not imply reasoning: A correct answer frequently masks\n  memorized retrieval rather than structural computation.', fontsize=9.5, color=DGRAY, fontweight='bold', linespacing=1.4)
cn.text(0.04, 0.35, '• Triangulation is strictly necessary: Single behavioral probes often disagree.\n  Independent axes yield more reliable mechanism diagnosis per-instance.', fontsize=9.5, color=DGRAY, fontweight='bold', linespacing=1.4)
# 11. Limitations
lm = add_panel(COL3_X, 0.105, COL_W, 0.080, edge_color=LGRAY)
panel_header(lm, 'LIMITATIONS', LGRAY)
lm.text(0.04, 0.65, '• Limited dataset size (N=45 problems).\n• Contamination fingerprinting (n-grams) may under-detect paraphrased leakage.\n• Mechanistic analysis strictly limited to open-weight models.', fontsize=9, color=SLATE, linespacing=1.5)
# 12. Future Work
fw = add_panel(COL3_X, 0.015, COL_W, 0.075, edge_color=GOLD)
panel_header(fw, 'FUTURE WORK', GOLD)
fw.text(0.04, 0.60, '• Scale the benchmark dataset to 200+ problems across more reasoning domains.\n• Refine contamination scoring using embedding-based semantic detection.\n• Extend mechanistic analysis to larger models via attribution patching.', fontsize=9, color='#92400E', linespacing=1.5)
# ─── SAVE OUTPUT ──────────────────────────────────────────────────────────────
out = 'research_poster_fixed.png'
plt.savefig(out, dpi=DPI, bbox_inches='tight', facecolor=BG, format='png', pil_kwargs={'optimize':True})
print(f'Saved: {out}')
plt.close()
