"""
Two-panel feature-analysis figure:
    (a) mean raw profile (bare DNA vs protein-bound) for a chosen image
    (b) top feature importances of the trained classifier

Usage:
    python plot_feature_analysis.py --image Alu4 --xlsx_dir heatmap/
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import load

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from ml_profile_classifier import load_one, NM_RANGE, NM_SAMPLES
from dna_backbone_trace import load_pxum_config


DIP_FEATS = {
    'center_depth_rel', 'peak_depth_rel',
    'integrated_dip', 'peakiness', 'mean_rel',
    'outer_peak', 'outer_mean', 'outer_to_peak_ratio',
}
WIDTH_FEATS = {
    'std_rel', 'range_rel',
    'lr_asymmetry_rel', 'lr_asymmetry_w10',
    'max_abs_d1', 'd1_std', 'max_abs_d2', 'd2_std',
    'curv_center', 'argmax_offset', 'abs_argmax_offset',
    'skew', 'kurt',
}


def feat_category(name):
    if name.startswith('raw_p'):
        return 'raw'
    if name.startswith('mean_w') or name.startswith('peak_w'):
        return 'dip'
    if name.startswith('std_w'):
        return 'width'
    if name.startswith('c+') or name.startswith('c-') or name.startswith('c+0'):
        return 'dip'
    if name.startswith('width_ge_'):
        return 'width'
    if name in DIP_FEATS:
        return 'dip'
    if name in WIDTH_FEATS:
        return 'width'
    return 'width'


def pretty_label(name, nm_per_sample):
    def nm_from_halfw(h):
        return int(round(h * nm_per_sample))

    def nm_from_off(off):
        return off * nm_per_sample

    if name.startswith('mean_w'):
        h = int(name[len('mean_w'):])
        return f'Mean intensity (center ±{nm_from_halfw(h)} nm)'
    if name.startswith('peak_w'):
        h = int(name[len('peak_w'):])
        return f'Peak intensity (center ±{nm_from_halfw(h)} nm)'
    if name.startswith('std_w'):
        h = int(name[len('std_w'):])
        return f'Intensity std (center ±{nm_from_halfw(h)} nm)'
    if name == 'center_depth_rel':
        return 'Center dip depth'
    if name == 'peak_depth_rel':
        return 'Peak dip depth'
    if name.startswith('c+') or name.startswith('c-'):
        off = int(name.split('_')[0][1:])
        nm = nm_from_off(off)
        sign = '+' if nm >= 0 else '-'
        return f'Relative at {sign}{abs(nm):.1f} nm'
    if name.startswith('raw_p'):
        off = int(name[len('raw_p'):])
        nm = nm_from_off(off)
        sign = '+' if nm >= 0 else '-'
        return f'Profile at {sign}{abs(nm):.1f} nm'
    if name == 'integrated_dip':
        return 'Integrated dip area'
    if name == 'mean_rel':
        return 'Mean relative'
    if name == 'std_rel':
        return 'Std relative'
    if name == 'range_rel':
        return 'Range relative'
    if name == 'peakiness':
        return 'Peakiness'
    if name.startswith('width_ge_'):
        pct = name[len('width_ge_'):-len('pct')]
        return f'Width ≥ {pct}% peak'
    if name == 'lr_asymmetry_rel':
        return f'L/R asymmetry (±{nm_from_halfw(5)} nm)'
    if name == 'lr_asymmetry_w10':
        return f'L/R asymmetry (±{nm_from_halfw(10)} nm)'
    if name == 'max_abs_d1':
        return 'Max |dP/dx|'
    if name == 'd1_std':
        return 'Std of dP/dx'
    if name == 'max_abs_d2':
        return 'Max |d²P/dx²|'
    if name == 'd2_std':
        return 'Std of d²P/dx²'
    if name == 'curv_center':
        return 'Central curvature'
    if name == 'argmax_offset':
        return 'Argmax offset'
    if name == 'abs_argmax_offset':
        return '|Argmax offset|'
    if name == 'skew':
        return 'Skewness'
    if name == 'kurt':
        return 'Kurtosis'
    if name == 'outer_peak':
        return 'Outer peak'
    if name == 'outer_mean':
        return 'Outer mean'
    if name == 'outer_to_peak_ratio':
        return 'Outer/peak ratio'
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', type=str, required=True,
                    help='Image name whose profiles feed panel (a)')
    ap.add_argument('--xlsx_dir', type=str, default=HERE,
                    help='Directory containing {image}_profile_heatmap.xlsx')
    ap.add_argument('--model', type=str,
                    default=os.path.join(HERE, 'profile_classifier.joblib'))
    ap.add_argument('--top_n', type=int, default=12)
    ap.add_argument('--out_path', type=str,
                    default=os.path.join(HERE, 'feature_analysis.png'))
    args = ap.parse_args()

    model_info = load(args.model)
    feat_names = model_info['feature_names']
    pipe = model_info['pipeline']
    clf = pipe.named_steps['clf']
    imps = np.mean([cc.estimator.feature_importances_
                    for cc in clf.calibrated_classifiers_], axis=0)

    xlsx = os.path.join(args.xlsx_dir, f'{args.image}_profile_heatmap.xlsx')
    if not os.path.exists(xlsx):
        raise SystemExit(f"xlsx for '{args.image}' not found at {xlsx}")
    ti, P_raw, y = load_one(xlsx)
    print(f"Panel (a) source: {xlsx}")
    print(f"  {len(y)} profiles  ({int(y.sum())} yellow / "
          f"{int((y==0).sum())} bare)")

    pos_mask = y == 1
    n_bare = int((~pos_mask).sum())
    n_prot = int(pos_mask.sum())
    bare_mean = np.nanmean(P_raw[~pos_mask], axis=0)
    bare_sem = np.nanstd(P_raw[~pos_mask], axis=0) / np.sqrt(max(n_bare, 1))
    prot_mean = np.nanmean(P_raw[pos_mask], axis=0)
    prot_sem = np.nanstd(P_raw[pos_mask], axis=0) / np.sqrt(max(n_prot, 1))
    W = P_raw.shape[1]
    center = W // 2
    distances_px = np.arange(W) - center

    pxnm_cfg = load_pxum_config()
    pxnm = pxnm_cfg.get(args.image, 3.47)
    distances_nm = distances_px * pxnm

    nm_per_sample = 2 * model_info['nm_range'] / (model_info['nm_samples'] - 1)

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(11, 4.3), gridspec_kw={'width_ratios': [1, 1.35]})

    ax_a.plot(distances_nm, bare_mean, color='#4B5C6B', lw=1.3,
              label=f'Bare DNA (n={n_bare})')
    ax_a.fill_between(distances_nm, bare_mean - bare_sem, bare_mean + bare_sem,
                      color='#4B5C6B', alpha=0.25, linewidth=0)
    ax_a.plot(distances_nm, prot_mean, color='#2CA02C', lw=1.3,
              label=f'Protein-bound (n={n_prot})')
    ax_a.fill_between(distances_nm, prot_mean - prot_sem, prot_mean + prot_sem,
                      color='#2CA02C', alpha=0.3, linewidth=0)
    ax_a.set_xlabel('Distance from backbone (nm)', fontsize=11)
    ax_a.set_ylabel('Intensity (a.u.)', fontsize=11)
    leg_a = ax_a.legend(fontsize=7, frameon=True, loc='lower right',
                        edgecolor='black')
    leg_a.get_frame().set_linewidth(0.8)
    ax_a.text(-0.18, 1.02, '(a)', transform=ax_a.transAxes,
              fontsize=13, fontweight='bold', va='bottom', ha='left')
    ax_a.tick_params(labelsize=10)

    order = np.argsort(imps)[::-1][:args.top_n]
    sel_names = [feat_names[i] for i in order]
    sel_imps = imps[order]
    sel_cats = [feat_category(n) for n in sel_names]
    sel_labels = [pretty_label(n, nm_per_sample) for n in sel_names]

    color_map = {'dip': '#1F77B4', 'width': '#FF7F0E', 'raw': '#9A9A9A'}
    bar_colors = [color_map[c] for c in sel_cats]

    y_pos = np.arange(len(sel_names))[::-1]
    ax_b.barh(y_pos, sel_imps, color=bar_colors,
              edgecolor='black', linewidth=0.6)
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(sel_labels, fontsize=9)
    ax_b.set_xlabel('Feature importance', fontsize=11)
    ax_b.tick_params(labelsize=10)
    ax_b.text(-0.48, 1.02, '(b)', transform=ax_b.transAxes,
              fontsize=13, fontweight='bold', va='bottom', ha='left')

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color=color_map['dip'], label='Dip magnitude'),
        Patch(color=color_map['width'], label='Width / variability'),
        Patch(color=color_map['raw'], label='Raw profile value'),
    ]
    leg_b = ax_b.legend(handles=legend_handles, loc='lower right',
                        fontsize=9, frameon=True, edgecolor='black')
    leg_b.get_frame().set_linewidth(0.8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    fig.savefig(args.out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {args.out_path}")


if __name__ == '__main__':
    main()
