import matplotlib.pyplot as plt, numpy as np, os
metrics=['clean','gaussian','motion_blur','occlusion']
values={'Baseline':[62.76,57.30,27.16,55.06], 'DRAGS':[64.15,57.54,25.41,55.26]}
errors={'Baseline':[1.05,2.51,1.64,2.71], 'DRAGS':[2.30,2.08,1.12,1.38]}
x=np.arange(len(metrics)); width=0.35; fig,ax=plt.subplots(figsize=(3.5,2.5))
ax.bar(x-width/2, values['Baseline'], width, yerr=errors['Baseline'], capsize=4, label='Baseline', edgecolor='black')
ax.bar(x+width/2, values['DRAGS'], width, yerr=errors['DRAGS'], capsize=4, label='DRAGS', edgecolor='black')
for i in range(len(metrics)):
    if values['DRAGS'][i] > values['Baseline'][i]:
        ax.text(x[i]+width/2, values['DRAGS'][i]+errors['DRAGS'][i]+0.5, '★', color='red', ha='center', va='bottom', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=9); ax.set_xticks(x); ax.set_xticklabels(metrics, rotation=15, ha='right', fontsize=8)
ax.set_ylim(0, max(max(values['Baseline']),max(values['DRAGS']))+10); ax.legend(loc='upper center', bbox_to_anchor=(0.5,1.15), ncol=2, fontsize=8, frameon=False); ax.yaxis.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout(); os.makedirs('drags_outputs/figures', exist_ok=True); plt.savefig('drags_outputs/figures/performance_comparison_vibrant.pdf', format='pdf', bbox_inches='tight'); print('Saved → drags_outputs/figures/performance_comparison_vibrant.pdf')
