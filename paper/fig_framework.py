"""生成框架流程图"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

fig, ax = plt.subplots(1, 1, figsize=(12, 3.5))
ax.set_xlim(0, 12)
ax.set_ylim(0, 3.5)
ax.axis('off')

# 颜色定义
blue = '#4472C4'
orange = '#ED7D31'
green = '#70AD47'
red = '#FF6B6B'
gray = '#D9D9D9'

def add_box(x, y, w, h, text, color, fontsize=9, bold=False):
    weight = 'bold' if bold else 'normal'
    rect = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor='white', linewidth=1.5, alpha=0.9)
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            fontweight=weight, color='white')

def add_arrow(x1, y1, x2, y2, color='black', text='', fs=8):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
    if text:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my+0.1, text, ha='center', va='bottom', fontsize=fs, color=color)

# Row 1: Feature Layer (standard LightGCN - NO LLM)
add_box(4.5, 2.5, 4.0, 0.8, 'ID Embedding\n(random init, trainable)', blue, 8)

# Row 2: Graph Propagation
add_box(4.5, 1.4, 6.0, 0.8,
        'Graph Propagation: E = mean(E0 + ... + EK)\nLightGCN K=3 layers, standard sparse.mm', blue, 8)
add_arrow(4.5, 2.1, 4.5, 1.8, 'black')

# Row 3: BPR Prediction
add_box(1.8, 0.4, 2.5, 0.7, 'BPR Forward\npos_score, neg_score', green, 8)

# Row 3: Semantic Weighting (side)
add_box(6.5, 0.4, 3.5, 0.7,
        'LLM Semantic Emb (frozen)\ncos_sim(pos, neg) -> weight', gray, 8)
add_box(10.3, 0.4, 2.0, 0.7,
        'loss = -(weight)\n* log sigma(pos-neg)', red, 8, bold=True)
add_arrow(3.05, 0.4, 4.75, 0.4, 'black')
add_arrow(8.25, 0.4, 9.3, 0.4, 'black')
add_arrow(4.5, 1.0, 1.8, 0.75, 'black')
add_arrow(4.5, 1.0, 6.5, 0.75, 'black')

# Highlight
rect = FancyBboxPatch((6.0, -0.05), 6.8, 0.9, boxstyle="round,pad=0.1",
                      facecolor='none', edgecolor=red, linewidth=2, linestyle='--')
ax.add_patch(rect)
ax.text(9.4, 1.05, 'OUR CONTRIBUTION', fontsize=9, fontweight='bold', color=red, va='center')

plt.tight_layout()
plt.savefig('I:/claude_code文件/run_recmodels/paper/fig_framework.pdf',
            dpi=150, bbox_inches='tight')
plt.savefig('I:/claude_code文件/run_recmodels/paper/fig_framework.png',
            dpi=150, bbox_inches='tight')
print('fig_framework saved')
