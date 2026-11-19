"""Fig. 3: distribution of warfarin discharge doses (every distinct dose, numeric mg axis broken between 10 and 15 mg,
2-3 mg band). Aggregated counts only.
Run: python figures/make_fig3_dose_distribution.py --split-dir DATA/warfarin --out Fig3.png
(split_train.csv, split_valid.csv, split_test.csv with the column final_dose."""
import argparse

import pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument('--split-dir', required=True); ap.add_argument('--out', default='Fig3.png')
ap.add_argument('--n-expected', type=int, default=0, help='optional record count check; 0 = skip')
A = ap.parse_args(); D = A.split_dir
dose = pd.concat([pd.read_csv(f'{D}/split_{s}.csv', usecols=['final_dose']) for s in ('train', 'valid', 'test')])['final_dose']
cnt = dose.value_counts().sort_index()
assert A.n_expected in (0, cnt.sum()), cnt.sum()
STEM, DOT, INK, MUTED = '#74C3B9', '#17998A', '#222', '#6B6B6B'
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True,
                              gridspec_kw={'width_ratios': [10.5, 2.6], 'wspace': 0.06})
for a, lo, hi in ((ax, 0, 10.6), (ax2, 13.6, 26.2)):
    sub = cnt[(cnt.index >= lo) & (cnt.index <= hi)]
    a.vlines(sub.index, 0, sub.values, color=STEM, lw=2)
    a.scatter(sub.index, sub.values, s=34, color=DOT, zorder=3)
    for x, y in sub.items():
        if y >= 7 or x >= 10:
            a.annotate(f'{y:,}', (x, y), xytext=(0, 6), textcoords='offset points', ha='center', fontsize=8.5, color=INK)
    a.set_xlim(lo, hi)
    a.spines[['top', 'right']].set_visible(False)
    a.tick_params(colors=INK, labelsize=9)
ax.axvspan(2, 3, color='#000', alpha=0.05, lw=0)
ax.text(2.5, 690, '2–3 mg', ha='center', va='bottom', fontsize=8.5, color=MUTED)
ax.set_xticks(range(0, 11))
ax2.set_xticks([15, 20, 25])
ax2.spines['left'].set_visible(False)
ax2.tick_params(left=False)
ax.set_ylim(0, 730)
ax.set_ylabel('Number of patients', color=INK)
fig.supxlabel('Warfarin discharge dose (mg)', fontsize=10.5, color=INK, y=0.02)
# break marks
d = 0.012
for a, xs in ((ax, (1,)), (ax2, (0,))):
    for x0 in xs:
        a.plot((x0 - d * (1 if a is ax else 4), x0 + d * (1 if a is ax else 4)), (-d * 2, d * 2),
               transform=a.transAxes, color=INK, lw=1, clip_on=False)
ax2.text(26.0, 700, f'n = {cnt.sum():,}\n{len(cnt)} distinct doses', ha='right', va='top', fontsize=8.5, color=MUTED)
fig.savefig(A.out, dpi=300, bbox_inches='tight')
print(cnt.to_dict())
