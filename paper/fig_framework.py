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

# Row 1: Feature Layer
add_box(2, 2.5, 2.2, 0.8, 'ID Embedding\n(random, trainable)', blue, 8)
add_box(4.7, 2.5, 2.2, 0.8, 'LLM Semantic Emb\n(frozen, 1536-dim)', gray, 8)
add_box(7.4, 2.5, 1.5, 0.8, 'Projection\nLinear(1536,64)', green, 8)
add_arrow(3.1, 2.5, 3.6, 2.5, 'black')
add_arrow(5.8, 2.5, 6.65, 2.5, 'black')

# Row 2: Graph Propagation
add_box(4.5, 1.4, 5.5, 0.8,
        'Graph Propagation: E = mean(E0 + E1 + E2 + E3)\nLightGCN K=3 layers, sparse.mm', blue, 8)

add_arrow(2, 2.1, 3, 1.8, 'black')
add_arrow(7.4, 2.1, 6.5, 1.8, 'black')

# Row 3: BPR Loss
add_box(4.5, 0.4, 5.5, 0.7,
        'BPR Loss with Semantic Weighting\nloss = -(1 + beta*cos_sim) * log(sigma(pos-neg))', red, 8, bold=True)

add_arrow(4.5, 1.0, 4.5, 0.75, 'black')

# Highlight box
rect = FancyBboxPatch((1.5, 0.0), 6.0, 0.1 + 0.7 + 0.1, boxstyle="round,pad=0.1",
                      facecolor='none', edgecolor=red, linewidth=2, linestyle='--')
ax.add_patch(rect)
ax.text(8.0, 0.4, 'OUR CONTRIBUTION', fontsize=9, fontweight='bold', color=red,
        va='center')

plt.tight_layout()
plt.savefig('I:/claude_code文件/run_recmodels/paper/fig_framework.pdf',
            dpi=150, bbox_inches='tight')
plt.savefig('I:/claude_code文件/run_recmodels/paper/fig_framework.png',
            dpi=150, bbox_inches='tight')
print('fig_framework saved')
