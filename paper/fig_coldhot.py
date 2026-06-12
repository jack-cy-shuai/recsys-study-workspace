"""生成冷/温/热用户分组 Recall@20 对比图"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# Data from experiments
groups = ['Cold (<5)', 'Warm (5-20)', 'Hot (>20)']
group_sizes = [1970, 6736, 1154]

lgc   = [0.0253, 0.0278, 0.0225]
llm   = [0.0629, 0.0636, 0.0533]
hardbpr = [0.0639, 0.0647, 0.0539]

x = np.arange(len(groups))
width = 0.25

fig, ax = plt.subplots(figsize=(8, 4.5))

bars1 = ax.bar(x - width, lgc, width, label='LightGCN (random)', color='#4472C4', edgecolor='white')
bars2 = ax.bar(x, llm, width, label='+LLM (additive)', color='#ED7D31', edgecolor='white')
bars3 = ax.bar(x + width, hardbpr, width, label='+HardBPR-W (ours)', color='#70AD47', edgecolor='white')

# Add values on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.001, f'{h:.4f}',
                ha='center', va='bottom', fontsize=8)

ax.set_ylabel('Recall@20', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels([f'{g}\n({n} users)' for g, n in zip(groups, group_sizes)], fontsize=10)
ax.legend(fontsize=10, loc='upper right')
ax.set_ylim(0, 0.08)
ax.grid(axis='y', alpha=0.3)

# Add improvement annotation
for i in range(3):
    imp = (hardbpr[i] - llm[i]) / llm[i] * 100
    ax.annotate(f'+{imp:.1f}%', xy=(x[i] + width, hardbpr[i]),
                xytext=(x[i] + width + 0.1, hardbpr[i] + 0.008),
                fontsize=9, color='#70AD47', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#70AD47', lw=1))

plt.tight_layout()
plt.savefig('I:/claude_code文件/run_recmodels/paper/fig_coldhot.pdf',
            dpi=150, bbox_inches='tight')
plt.savefig('I:/claude_code文件/run_recmodels/paper/fig_coldhot.png',
            dpi=150, bbox_inches='tight')
print('fig_coldhot saved')
