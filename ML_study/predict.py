"""
Predict DNA-protein binding using the trained classifier.

Usage:
    python predict.py image.tif
    python predict.py tif/

Per-image px/µm scale must be present in pxum_config.json (no defaults).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image
from joblib import load
import openpyxl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_profile_classifier import (
    make_features, resample_to_nm, smooth_proba_along_trace, apply_min_run,
)
from dna_backbone_trace import (
    compute_line_strength, auto_select_trace, get_or_pick_endpoints,
    load_pxum_config, ensure_pxum_entries,
)
from dna_edge_width import (
    extract_perpendicular_profiles, measure_edge_width_subpixel,
)

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
MODEL_PATH = os.path.join(HERE, 'profile_classifier.joblib')
ENDPOINTS_JSON = os.path.join(ROOT, 'endpoints.json')


def load_endpoints_cache():
    """Load endpoints.json → dict of {image_name: (start_pt, end_pt)}."""
    if not os.path.exists(ENDPOINTS_JSON):
        return {}
    with open(ENDPOINTS_JSON) as f:
        data = json.load(f)
    out = {}
    for k, v in data.items():
        name = os.path.splitext(k)[0]
        out[name] = (v['start'], v['end'])
    return out


def runs_of(binary):
    b = np.concatenate([[0], binary.astype(int), [0]])
    d = np.diff(b)
    starts = np.where(d == 1)[0].tolist()
    ends = np.where(d == -1)[0].tolist()
    return list(zip(starts, ends))


def load_truth_xlsx(xlsx_path):
    """Load CV predictions xlsx → dict of {image: (trace_idx, true, proba, pred)}."""
    wb = openpyxl.load_workbook(xlsx_path)
    ws = wb.active
    data = {}
    for r in range(2, ws.max_row + 1):
        img = ws.cell(r, 1).value
        ti = ws.cell(r, 2).value
        true = ws.cell(r, 3).value
        if img is None:
            continue
        data.setdefault(img, []).append((ti, true))
    out = {}
    for k, rows in data.items():
        rows.sort(key=lambda x: x[0])
        ti = np.array([r[0] for r in rows], dtype=int)
        tr = np.array([r[1] for r in rows], dtype=int)
        out[k] = (ti, tr)
    return out


def process(tif_path, model_info, pxnm=None, out_dir=None,
            truth=None, endpoints=None):
    name = os.path.splitext(os.path.basename(tif_path))[0]
    img = Image.open(tif_path)
    gray = np.array(img.convert('L')).astype(float)
    h, w = gray.shape

    threshold = model_info['threshold']
    pp = model_info.get('post_process', {})
    input_type = model_info.get('input_type', 'pixel')

    # trace (full image width, endpoint masking applied later)
    line_strength = compute_line_strength(gray)
    trace_local, _ = auto_select_trace(
        gray, line_strength, 0, h - 1,
        max_step=2, template_sigma=1.5, continuity_weight=1.0, x_avg=1)

    cx = np.arange(w).astype(float)
    cy = trace_local.astype(float)

    # endpoint range
    if endpoints is not None:
        start_pt, end_pt = endpoints
        x_start = min(int(start_pt[0]), int(end_pt[0]))
        x_end = max(int(start_pt[0]), int(end_pt[0]))
        print(f"  endpoints: ({start_pt[0]},{start_pt[1]}) -> "
              f"({end_pt[0]},{end_pt[1]})  x=[{x_start}, {x_end}]")
        valid_mask_ep = (cx >= x_start) & (cx <= x_end)
    else:
        x_start, x_end = 0, w - 1
        valid_mask_ep = np.ones(len(cx), dtype=bool)

    # profiles
    profiles, distances = extract_perpendicular_profiles(
        gray, cx, cy, half_width=20)

    # NaN out profiles outside endpoint range
    profiles[~valid_mask_ep, :] = np.nan

    # feature extraction depends on input_type
    if input_type == 'nm_resampled':
        if pxnm is None:
            raise ValueError(f"Continuous model requires --pxnm for {name}")
        nm_range = model_info.get('nm_range', 55)
        nm_samples = model_info.get('nm_samples', 41)
        P_nm = resample_to_nm(profiles, pxnm, target_nm=nm_range,
                              n_samples=nm_samples)
        X, _ = make_features(P_nm)
    else:
        X, _ = make_features(profiles)

    proba = model_info['pipeline'].predict_proba(X)[:, 1]
    # zero out predictions outside endpoint range
    proba[~valid_mask_ep] = 0.0

    # post-process
    trace_idx = np.arange(len(cx))
    src = np.array([name] * len(trace_idx))

    if input_type == 'nm_resampled':
        # continuous: convert nm params to px
        smooth_w = max(1, int(round(pp.get('smooth_nm', 28) / pxnm)))
        min_run = max(1, int(round(pp.get('min_run_nm', 30) / pxnm)))
    else:
        smooth_w = pp.get('smooth_w', pp.get('smooth_window', 7))
        min_run = pp.get('min_run', 4)

    proba_sm = smooth_proba_along_trace(proba, trace_idx, src,
                                         window=smooth_w)
    pred = (proba_sm >= threshold).astype(int)
    pred = apply_min_run(pred, trace_idx, src, min_run=min_run)

    pred_runs = runs_of(pred)
    run_lengths = [e - s for s, e in pred_runs]
    print(f"\n── {name}  ({w}x{h}) ──")
    print(f"  threshold={threshold:.2f}, "
          f"smooth_w={smooth_w}, min_run={min_run}")
    print(f"  predicted positive: {int(pred.sum())}")
    print(f"  predicted runs: {len(pred_runs)}  lengths={run_lengths}")

    # edge width per predicted run: pool all profiles in the run, then fit edges
    run_edge_info = []
    for s, e in pred_runs:
        seg = profiles[s:e]
        valid = ~np.any(np.isnan(seg), axis=1)
        n_valid = int(valid.sum())
        width_px = left_px = right_px = None
        if n_valid >= 3:
            avg_prof = seg[valid].mean(axis=0)
            width_px, left_px, right_px, _ = measure_edge_width_subpixel(
                distances, avg_prof, smooth_sigma=0)
        run_edge_info.append((s, e, n_valid, width_px, left_px, right_px))
        w_str = (f"{width_px:.2f}px" if width_px is not None else "failed")
        if width_px is not None and pxnm is not None:
            w_str += f" ({width_px * pxnm:.2f}nm)"
        print(f"    run [{s:4d}, {e-1:4d}]  len={e-s}  "
              f"n_valid={n_valid}  edge_width={w_str}")

    # overlay
    n_pred = int(pred.sum())

    # match truth labels if available
    true_labels = None
    if truth is not None:
        ti_truth, y_truth = truth
        true_labels = np.zeros(len(cx), dtype=int)
        valid = (ti_truth >= 0) & (ti_truth < len(cx))
        true_labels[ti_truth[valid]] = y_truth[valid]

    # x-axis: convert to nm if pxnm available
    if pxnm is not None:
        x_plot = cx * pxnm  # nm
        x_label = 'Position along DNA (nm)'
    else:
        x_plot = cx
        x_label = 'Position along DNA (px)'

    fig, axes = plt.subplots(3, 1, figsize=(14, 6.5),
                             gridspec_kw={'height_ratios': [2, 1, 0.55]},
                             sharex=True)
    ax = axes[0]
    if pxnm is not None:
        extent = [x_plot[0], x_plot[-1], gray.shape[0], 0]
        ax.imshow(gray, cmap='gray', aspect='auto', extent=extent)
    else:
        ax.imshow(gray, cmap='gray', aspect='auto')

    # only draw points within endpoint range
    ep = valid_mask_ep

    if true_labels is not None:
        TP = (pred == 1) & (true_labels == 1) & ep
        FN = (pred == 0) & (true_labels == 1) & ep
        FP = (pred == 1) & (true_labels == 0) & ep
        TN = (pred == 0) & (true_labels == 0) & ep
        tp, fp, fn = int(TP.sum()), int(FP.sum()), int(FN.sum())
        rec = tp / (tp + fn) if (tp + fn) else float('nan')
        prec = tp / (tp + fp) if (tp + fp) else float('nan')

        ax.scatter(x_plot[TN], cy[TN] - 15, s=8, c='black', alpha=0.3, label='TN')
        ax.scatter(x_plot[TP], cy[TP] - 15, s=8, c='#22c55e', label=f'TP ({tp})')
        ax.scatter(x_plot[FN], cy[FN] - 15, s=8, c='#f59e0b', label=f'FN ({fn})')
        ax.scatter(x_plot[FP], cy[FP] - 15, s=8, c='#ef4444', label=f'FP ({fp})')
        ax.set_title(f'{name}  |  TP={tp}  FP={fp}  FN={fn}  '
                     f'Recall={rec:.2f}  Precision={prec:.2f}', fontsize=13)
    else:
        neg = (pred == 0) & ep; pos = (pred == 1) & ep
        ax.scatter(x_plot[neg], cy[neg] - 15, s=8, c='black', alpha=0.3, label='TN')
        ax.scatter(x_plot[pos], cy[pos] - 15, s=8, c='#22c55e',
                   label=f'Predicted ({n_pred})')
        ax.set_title(f'{name}  |  Predicted runs={len(pred_runs)}  '
                     f'(N={n_pred})', fontsize=13)

    ax.set_ylabel('y (px)', fontsize=12)
    ax.legend(loc='upper right', fontsize=9, ncol=4,
              handlelength=1.5, handletextpad=0.5, borderpad=0.3)

    ax2 = axes[1]
    ax2.plot(x_plot, proba_sm, color='#64748b', linewidth=0.8)
    ax2.axhline(threshold, color='red', linestyle='--', linewidth=1.0,
                label=f'Threshold={threshold:.2f}')
    if true_labels is not None:
        ax2.fill_between(x_plot, 0, 1, where=true_labels.astype(bool),
                         color='#fbbf24', alpha=0.18, label='Truth')
    ax2.fill_between(x_plot, 0, 1, where=pred.astype(bool), color='#22c55e',
                     alpha=0.15, label='Predicted')
    ax2.set_ylabel('P(binding)', fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.legend(loc='upper right', fontsize=9, ncol=3,
               handlelength=1.5, handletextpad=0.5, borderpad=0.3)

    # ── Third panel: trace line colored by binding/non-binding + lengths ──
    ax3 = axes[2]
    valid_idx = np.where(valid_mask_ep)[0]
    if len(valid_idx) > 0:
        v_start, v_end = int(valid_idx[0]), int(valid_idx[-1])

        # Draw trace segments colored by binding (green) / non-binding (gray)
        segments = []
        cur = int(pred[v_start])
        seg_s = v_start
        for j in range(v_start + 1, v_end + 1):
            if int(pred[j]) != cur:
                segments.append((seg_s, j - 1, bool(cur)))
                seg_s = j
                cur = int(pred[j])
        segments.append((seg_s, v_end, bool(cur)))

        bind_total = 0.0
        nonbind_total = 0.0
        for s, e, is_bind in segments:
            idx = np.arange(s, e + 1)
            color = '#22c55e' if is_bind else '#94a3b8'
            lw = 2.5 if is_bind else 1.2
            ax3.plot(x_plot[idx], cy[idx], color=color, linewidth=lw,
                     solid_capstyle='round')

            # Length annotation
            x0, x1 = x_plot[s], x_plot[e]
            seg_w = x1 - x0
            if pxnm is not None:
                seg_len = (e - s + 1) * pxnm
                label_txt = f"{seg_len:.0f}"
            else:
                seg_len = e - s + 1
                label_txt = f"{seg_len}"
            if is_bind:
                bind_total += seg_len
            else:
                nonbind_total += seg_len

            mid_idx = (s + e) // 2
            y_pos = float(cy[mid_idx])
            txt_color = '#15803d' if is_bind else '#64748b'
            fs = 7 if is_bind else 6
            ax3.annotate(
                label_txt,
                xy=((x0 + x1) / 2, y_pos),
                xytext=(0, -12 if is_bind else 10),
                textcoords='offset points',
                ha='center', va='center', fontsize=fs,
                color=txt_color, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15',
                          fc='white', ec=txt_color, alpha=0.85,
                          lw=0.5))

        unit = 'nm' if pxnm is not None else 'px'
        ax3.text(0.005, 1.08,
                 f"Σ binding = {bind_total:.0f} {unit}   "
                 f"Σ non-binding = {nonbind_total:.0f} {unit}",
                 transform=ax3.transAxes, fontsize=8, va='bottom',
                 color='#334155')

    # Match y-axis range to trace shape
    cy_valid = cy[valid_idx] if len(valid_idx) > 0 else cy
    y_margin = (cy_valid.max() - cy_valid.min()) * 0.3 + 5
    ax3.set_ylim(cy_valid.max() + y_margin, cy_valid.min() - y_margin)
    ax3.set_ylabel('y (px)', fontsize=10)
    ax3.set_xlabel(x_label, fontsize=12)

    plt.tight_layout()
    save_dir = out_dir or os.path.join(HERE, 'overlay')
    os.makedirs(save_dir, exist_ok=True)
    suffix = 'overlay' if true_labels is not None else 'predict'
    out_path = os.path.join(save_dir, f'{name}_{suffix}.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved overlay -> {out_path}")

    return name, trace_idx, cx, cy, proba, proba_sm, pred, run_edge_info


def main():
    parser = argparse.ArgumentParser(
        description='Predict DNA-protein binding using the trained classifier')
    parser.add_argument('input', help='Input .tif file or folder of .tif files')
    parser.add_argument('pxum', type=float, nargs='?', default=None,
                        help='Default px/µm for any image missing from '
                             'pxum_config.json (auto-saved).')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Output directory for overlay PNGs')
    parser.add_argument('--truth_xlsx', type=str, default=None,
                        help='CV predictions xlsx for TP/FP/FN overlay')
    parser.add_argument('--no-interactive', dest='interactive',
                        action='store_false', default=True,
                        help='Disable interactive endpoint picker for images '
                             'not in endpoints.json (use full-range instead)')
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        print(f"Error: model not found: {MODEL_PATH}")
        sys.exit(1)

    model_info = load(MODEL_PATH)
    print(f"Model: {model_info.get('model_name')}  "
          f"threshold={model_info['threshold']:.2f}  "
          f"post_process={model_info.get('post_process')}")

    # pxum_config.json will be loaded after we know which tifs to process
    pxnm_config = {}

    # load endpoints cache
    ep_cache = load_endpoints_cache()
    if ep_cache:
        print(f"Loaded endpoints for {len(ep_cache)} images")

    # load truth labels if provided
    truth_data = {}
    if args.truth_xlsx:
        truth_data = load_truth_xlsx(args.truth_xlsx)
        print(f"Loaded truth for {len(truth_data)} images from "
              f"{os.path.basename(args.truth_xlsx)}")

    # collect tif files
    if os.path.isdir(args.input):
        tif_files = sorted(glob.glob(os.path.join(args.input, '*.tif')))
        if not tif_files:
            print(f"No .tif files found in {args.input}")
            sys.exit(1)
        print(f"Found {len(tif_files)} .tif files")
    else:
        tif_files = [args.input]


    wb = openpyxl.Workbook()
    summary = wb.active
    summary.title = 'summary'
    summary.append(['Image', 'N_pos', 'N_runs', 'Run_lengths'])

    runs_ws = wb.create_sheet('runs')
    runs_ws.append(['Image', 'Run_idx', 'Start', 'End', 'Length',
                    'N_valid_profiles', 'Edge_width_px', 'Edge_width_nm',
                    'Left_edge_px', 'Right_edge_px', 'pxnm'])

    all_ws = wb.create_sheet('predictions')
    all_ws.append(['Image', 'Trace_idx', 'cx_px', 'cy_px',
                   'proba_raw', 'proba_smoothed', 'pred'])

    # Ensure every input image has a px/µm entry; prompts for missing ones
    # (or applies CLI default_pxum) and persists them.
    names = [os.path.splitext(os.path.basename(t))[0] for t in tif_files]
    pxnm_config = ensure_pxum_entries(names, interactive=args.interactive,
                                      default_pxum=args.pxum)

    for tif_path in tif_files:
        img_name = os.path.splitext(os.path.basename(tif_path))[0]
        pxnm = pxnm_config[img_name]  # nm/px (already converted)

        img_truth = truth_data.get(img_name)
        if img_name in ep_cache:
            img_ep = ep_cache[img_name]
        else:
            # prompt GUI (or fall back to full-range if --no-interactive)
            img_gray = np.array(Image.open(tif_path).convert('L')).astype(float)
            img_ep = get_or_pick_endpoints(
                img_gray, img_name, interactive=args.interactive,
                title=img_name)
            ep_cache[img_name] = img_ep  # update in-memory cache
        name, ti, cx, cy, proba, proba_sm, pred, run_edge_info = process(
            tif_path, model_info, pxnm=pxnm, out_dir=args.out_dir,
            truth=img_truth, endpoints=img_ep)
        rl = runs_of(pred)
        summary.append([name, int(pred.sum()),
                        len(rl), str([e - s for s, e in rl])])

        for r_idx, (s, e, n_valid, w_px, le_px, re_px) in enumerate(run_edge_info):
            w_nm = (w_px * pxnm) if (w_px is not None and pxnm is not None) else None
            runs_ws.append([
                name, r_idx, int(s), int(e - 1), int(e - s), n_valid,
                round(float(w_px), 4) if w_px is not None else None,
                round(float(w_nm), 4) if w_nm is not None else None,
                round(float(le_px), 4) if le_px is not None else None,
                round(float(re_px), 4) if re_px is not None else None,
                pxnm,
            ])

        ws = wb.create_sheet(name[:31])
        ws.append(['trace_idx', 'cx_px', 'cy_px', 'proba_raw',
                   'proba_smoothed', 'pred'])
        for i in ti:
            ws.append([int(i), round(float(cx[i]), 2), round(float(cy[i]), 2),
                       round(float(proba[i]), 4), round(float(proba_sm[i]), 4),
                       int(pred[i])])
            all_ws.append([name, int(i),
                           round(float(cx[i]), 2), round(float(cy[i]), 2),
                           round(float(proba[i]), 4),
                           round(float(proba_sm[i]), 4),
                           int(pred[i])])

    out_xlsx = os.path.join(args.out_dir or HERE, 'predictions.xlsx')
    wb.save(out_xlsx)
    print(f"\nSaved predictions -> {out_xlsx}")


if __name__ == '__main__':
    main()
