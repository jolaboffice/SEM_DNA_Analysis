"""
Train a classifier to detect "yellow" profiles from *_profile_heatmap.xlsx.

Each row of the 'Profiles' sheet is a 41-pixel perpendicular cross-section
profile along a DNA trace. Rows highlighted yellow by the user are the
positive class (profiles of interest); remaining rows are negatives.

Loads every *_profile_heatmap.xlsx in ML_study/, uses yellow-fill rows as
positives and all other rows as negatives, trains several classifiers with
group-aware CV (grouped by source image), picks the best by F1, saves the
model + report + per-row predictions.
"""
import argparse
import glob
import json
import os
import numpy as np
import openpyxl
from PIL import Image
from joblib import dump
from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier,
                              HistGradientBoostingClassifier,
                              ExtraTreesClassifier, VotingClassifier)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (StratifiedKFold, GroupKFold,
                                     StratifiedGroupKFold, cross_val_predict)
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score, average_precision_score,
                             classification_report, confusion_matrix)

HERE = os.path.dirname(__file__)
XLSX_GLOB = os.path.join(HERE, '*_profile_heatmap.xlsx')

# Project root holds the unified pxum_config.json (px per µm) used by all
# scripts. Internally we still work in nm/px (NM_RANGE etc.) for backward
# compatibility with existing trained models.
import sys
sys.path.insert(0, os.path.dirname(HERE))
from dna_backbone_trace import load_pxum_config, ensure_pxum_entries  # noqa: E402
PXUM_JSON = os.path.join(os.path.dirname(HERE), 'pxum_config.json')

# Common nm grid for resampling profiles (scale-invariant across magnifications)
NM_RANGE = 65.0   # ± nm each side of center
NM_SAMPLES = 41   # number of samples across the common grid


# ── 1. Load profiles + yellow labels from every xlsx ────────────────────
def load_one(xlsx_path):
    wb = openpyxl.load_workbook(xlsx_path)
    ws = wb['Profiles']
    trace_idx, X_raw, y = [], [], []
    for r in range(2, ws.max_row + 1):
        fg = str(ws.cell(r, 1).fill.fgColor.rgb).upper()
        label = 1 if fg == 'FFFFFF00' else 0
        row = [ws.cell(r, c).value for c in range(2, ws.max_column + 1)]
        trace_idx.append(ws.cell(r, 1).value)
        X_raw.append(row)
        y.append(label)
    return (np.array(trace_idx), np.array(X_raw, dtype=float),
            np.array(y, dtype=int))


def resample_to_nm(P, px_size_nm, target_nm=NM_RANGE, n_samples=NM_SAMPLES):
    """Resample each row of P (N, W) from native px axis to a common nm grid.

    Assumes profile is centered at column W//2. Returns (N, n_samples).
    Out-of-range samples remain NaN.
    """
    N, W = P.shape
    center = W // 2
    src_px = np.arange(W) - center
    src_nm = src_px * px_size_nm
    tgt_nm = np.linspace(-target_nm, target_nm, n_samples)
    out = np.full((N, n_samples), np.nan)
    in_range = (tgt_nm >= src_nm.min()) & (tgt_nm <= src_nm.max())
    for i in range(N):
        row = P[i]
        valid = ~np.isnan(row)
        if valid.sum() < 2:
            continue
        v_nm = src_nm[valid]
        v_val = row[valid]
        # np.interp clamps out-of-range; mask those afterwards
        interp = np.interp(tgt_nm, v_nm, v_val)
        mask = (tgt_nm >= v_nm.min()) & (tgt_nm <= v_nm.max())
        out[i, mask] = interp[mask]
    return out


def load_all(pattern, pxnm_config, nm_range=NM_RANGE):
    """Load profiles from xlsx files, optionally resampling to nm grid.

    pxnm_config: dict with image_name -> nm/pixel, plus optional 'default'.
                 If None, profiles are used as-is (pixel-based).
    nm_range: half-range of common nm grid (default NM_RANGE).
    """
    files = sorted(glob.glob(pattern))
    all_trace, all_P, all_y, all_src = [], [], [], []
    use_nm = pxnm_config is not None
    if use_nm:
        print(f"Found {len(files)} xlsx files:")
        print(f"Resampling to common nm grid: ±{nm_range} nm, {NM_SAMPLES} samples "
              f"(step={2*nm_range/(NM_SAMPLES-1):.2f} nm)")
    else:
        print(f"Found {len(files)} xlsx files (pixel-based, no resampling):")
    for f in files:
        name = os.path.basename(f).replace('_profile_heatmap.xlsx', '')
        ti, P, y = load_one(f)

        if use_nm:
            if name not in pxnm_config:
                raise SystemExit(
                    f"Error: no entry for image '{name}' in "
                    f"pxum_config.json")
            px_nm = pxnm_config[name]
            P_out = resample_to_nm(P, px_nm, target_nm=nm_range)
            n_nan = np.isnan(P_out).sum(axis=1)
            print(f"  {name}: {len(y)} profiles  {px_nm:.2f}nm/px  "
                  f"{y.sum()} yellow  "
                  f"(avg NaN/row after resample: {n_nan.mean():.1f})")
        else:
            P_out = P
            print(f"  {name}: {len(y)} profiles  {y.sum()} yellow")

        all_trace.append(ti)
        all_P.append(P_out)
        all_y.append(y)
        all_src.append(np.array([name] * len(y)))
    return (np.concatenate(all_trace), np.concatenate(all_P, axis=0),
            np.concatenate(all_y), np.concatenate(all_src))


# ── 2. Feature engineering from raw 41-pixel profile ────────────────────
#     All features are SCALE-INVARIANT: each profile is normalized by its
#     own edge baseline so features do not depend on image brightness.
def make_features(P):
    N, W = P.shape
    center = W // 2
    names, feats = [], []

    # per-profile baseline (outer ±5 samples on each side) used for normalization
    edge = np.nanmean(np.concatenate([P[:, :5], P[:, -5:]], axis=1), axis=1)
    edge_safe = edge + 1e-6

    # normalized profile: fractional dip below baseline (0 = baseline, >0 = darker)
    Pn = (edge[:, None] - P) / edge_safe[:, None]

    # ── relative dip magnitudes ──
    feats.append(Pn[:, center]); names.append('center_depth_rel')
    feats.append(np.nanmax(Pn, axis=1)); names.append('peak_depth_rel')
    for off in (-2, -1, 0, 1, 2):
        feats.append(Pn[:, center + off]); names.append(f'c{off:+d}_rel')

    # multi-scale central windows
    for halfw in (3, 5, 8, 12):
        win = Pn[:, center - halfw:center + halfw + 1]
        feats.append(np.nanmean(win, axis=1)); names.append(f'mean_w{halfw}')
        feats.append(np.nanmax(win, axis=1)); names.append(f'peak_w{halfw}')
        feats.append(np.nanstd(win, axis=1)); names.append(f'std_w{halfw}')

    # whole-profile spread (normalized)
    feats.append(np.nanstd(Pn, axis=1)); names.append('std_rel')
    feats.append(np.nanmax(Pn, axis=1) - np.nanmin(Pn, axis=1)); names.append('range_rel')
    feats.append(np.nanmean(Pn, axis=1)); names.append('mean_rel')

    # ── FWHM-ish widths at multiple levels of peak dip ──
    peak_dip = np.nanmax(Pn, axis=1)
    peak_dip_safe = peak_dip + 1e-6
    for frac in (0.3, 0.5, 0.7):
        thr = peak_dip_safe[:, None] * frac
        feats.append((Pn >= thr).sum(axis=1).astype(float))
        names.append(f'width_ge_{int(frac*100)}pct')

    # ── L/R asymmetry on normalized profile ──
    left5 = np.nanmean(Pn[:, center - 5:center], axis=1)
    right5 = np.nanmean(Pn[:, center + 1:center + 6], axis=1)
    feats.append(np.abs(left5 - right5)); names.append('lr_asymmetry_rel')
    left10 = np.nanmean(Pn[:, center - 10:center], axis=1)
    right10 = np.nanmean(Pn[:, center + 1:center + 11], axis=1)
    feats.append(np.abs(left10 - right10)); names.append('lr_asymmetry_w10')

    # ── 1st & 2nd derivatives ──
    d1 = np.diff(Pn, axis=1)
    d2 = np.diff(d1, axis=1)
    feats.append(np.nanmax(np.abs(d1), axis=1)); names.append('max_abs_d1')
    feats.append(np.nanstd(d1, axis=1)); names.append('d1_std')
    feats.append(np.nanmax(np.abs(d2), axis=1)); names.append('max_abs_d2')
    feats.append(np.nanstd(d2, axis=1)); names.append('d2_std')
    # central curvature (2nd derivative at center)
    feats.append(d2[:, center - 1]); names.append('curv_center')

    # ── argmax of dip (where darkest sample is) ──
    # nanargmax can fail on all-NaN rows, mask those
    safe_Pn = np.where(np.isnan(Pn), -np.inf, Pn)
    argmax = safe_Pn.argmax(axis=1) - center
    feats.append(argmax.astype(float)); names.append('argmax_offset')
    feats.append(np.abs(argmax).astype(float)); names.append('abs_argmax_offset')

    # ── moment-like features (skewness, kurtosis of normalized profile) ──
    mu = np.nanmean(Pn, axis=1, keepdims=True)
    sig = np.nanstd(Pn, axis=1, keepdims=True) + 1e-6
    zs = (Pn - mu) / sig
    skew = np.nanmean(zs**3, axis=1)
    kurt = np.nanmean(zs**4, axis=1) - 3.0
    feats.append(skew); names.append('skew')
    feats.append(kurt); names.append('kurt')

    # ── "shoulder" detection: how much the dip spreads outside ±center inner window ──
    outer = np.concatenate([Pn[:, 5:center - 5], Pn[:, center + 6:W - 5]], axis=1)
    feats.append(np.nanmax(outer, axis=1)); names.append('outer_peak')
    feats.append(np.nanmean(outer, axis=1)); names.append('outer_mean')
    # ratio of outer_peak to center peak (broad vs narrow dip)
    feats.append(np.nanmax(outer, axis=1) / (peak_dip_safe))
    names.append('outer_to_peak_ratio')

    # ── integrated dip (area under normalized curve, positive part) ──
    pos = np.clip(Pn, 0, None)
    feats.append(np.nansum(pos, axis=1)); names.append('integrated_dip')
    # peakiness: peak / integrated
    feats.append(peak_dip / (np.nansum(pos, axis=1) + 1e-6))
    names.append('peakiness')

    # ── raw normalized profile values (41 features) ──
    # lets the tree models look at the full profile shape directly
    for j in range(W):
        off = j - center
        feats.append(Pn[:, j])
        names.append(f'raw_p{off:+d}')

    X = np.vstack(feats).T
    return X, names


# ── 3. Evaluate ─────────────────────────────────────────────────────────
def evaluate(name, pipe, X, y, cv, groups=None):
    fit_params = {}
    proba = cross_val_predict(pipe, X, y, cv=cv, method='predict_proba',
                              groups=groups, n_jobs=-1)[:, 1]
    pred = (proba >= 0.5).astype(int)
    ts = np.linspace(0.05, 0.95, 91)
    f1s = [f1_score(y, (proba >= t).astype(int), zero_division=0) for t in ts]
    best_t = ts[int(np.argmax(f1s))]
    pred_best = (proba >= best_t).astype(int)
    return {
        'name': name,
        'roc_auc': roc_auc_score(y, proba),
        'ap': average_precision_score(y, proba),
        'f1@0.5': f1_score(y, pred, zero_division=0),
        'precision@0.5': precision_score(y, pred, zero_division=0),
        'recall@0.5': recall_score(y, pred, zero_division=0),
        'best_threshold': best_t,
        'f1@best': max(f1s),
        'precision@best': precision_score(y, pred_best, zero_division=0),
        'recall@best': recall_score(y, pred_best, zero_division=0),
        'proba': proba,
        'pred_best': pred_best,
    }


def smooth_proba_along_trace(proba, trace_idx, src, window=7):
    """Apply rolling mean smoothing to probabilities along each image's trace."""
    out = proba.copy()
    for s in np.unique(src):
        m = np.where(src == s)[0]
        ti = trace_idx[m]
        order = np.argsort(ti)
        idx_sorted = m[order]
        p = proba[idx_sorted]
        # rolling mean with reflect padding
        if window % 2 == 0:
            window_eff = window + 1
        else:
            window_eff = window
        half = window_eff // 2
        pad = np.pad(p, half, mode='edge')
        sm = np.convolve(pad, np.ones(window_eff) / window_eff, mode='valid')
        out[idx_sorted] = sm
    return out


def apply_min_run(pred, trace_idx, src, min_run=5):
    """Remove positive runs shorter than min_run for each image."""
    out = pred.copy()
    for s in np.unique(src):
        m = np.where(src == s)[0]
        ti = trace_idx[m]
        order = np.argsort(ti)
        idx_sorted = m[order]
        seq = pred[idx_sorted].astype(int)
        # find runs of 1s
        d = np.diff(np.concatenate([[0], seq, [0]]))
        starts = np.where(d == 1)[0]
        ends = np.where(d == -1)[0]
        for a, b in zip(starts, ends):
            if b - a < min_run:
                seq[a:b] = 0
        out[idx_sorted] = seq
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default=None,
                        help='Directory containing *_profile_heatmap.xlsx '
                             '(defaults to script directory)')
    parser.add_argument('--smooth_w', type=int, default=-1,
                        help='Force post-processing smoothing window '
                             '(default: auto-selected by grid sweep)')
    parser.add_argument('--min_run', type=int, default=-1,
                        help='Force post-processing minimum run length '
                             '(default: auto-selected by grid sweep)')
    parser.add_argument('--pixel', action='store_true',
                        help='Use pixel-grid profiles (no nm resampling)')
    parser.add_argument('--nm_range', type=float, default=NM_RANGE,
                        help=f'Half-range of nm grid (default {NM_RANGE})')
    parser.add_argument('--smooth_nm', type=float, default=None,
                        help='Post-processing smoothing window in nm')
    parser.add_argument('--min_run_nm', type=float, default=None,
                        help='Post-processing minimum run length in nm')
    parser.add_argument('--pxum', type=float, default=None,
                        help='Default px/µm for training images missing from '
                             'pxum_config.json (auto-saved).')
    args = parser.parse_args()

    out_model = os.path.join(HERE, 'profile_classifier.joblib')
    out_report = os.path.join(HERE, 'profile_classifier_report.txt')
    out_pred = os.path.join(HERE, 'profile_classifier_predictions.xlsx')

    # Discover training image names from heatmap xlsx filenames so we can
    # prompt for any missing px/µm entries before training.
    xlsx_glob = os.path.join(args.dir or HERE, '*_profile_heatmap.xlsx')
    training_files = sorted(glob.glob(xlsx_glob))
    training_names = [
        os.path.basename(f).replace('_profile_heatmap.xlsx', '')
        for f in training_files
    ]
    pxnm_config = ensure_pxum_entries(training_names, interactive=True,
                                       default_pxum=args.pxum)
    print(f"pxum_config.json covers {len(pxnm_config)} training images")
    # --pixel: keep raw pixel-grid profiles (bit-identical to legacy short
    # behavior). The unit-display is still in nm (via pxnm) at output time.
    # Default (no --pixel): nm-resampling onto a common ±NM_RANGE nm grid for
    # cross-magnification training (continuous behavior).
    if args.pixel:
        pxnm_for_load = None
    else:
        pxnm_for_load = pxnm_config
    trace_idx, P, y, src = load_all(xlsx_glob, pxnm_for_load,
                                     nm_range=args.nm_range)
    print(f"\nTotal profiles: {P.shape[0]}  (pos={y.sum()}, neg={(y==0).sum()})")

    # drop all-NaN rows
    valid = ~np.all(np.isnan(P), axis=1)
    n_drop = (~valid).sum()
    if n_drop:
        print(f"Dropping {n_drop} all-NaN profile rows")
    trace_idx, P, y, src = trace_idx[valid], P[valid], y[valid], src[valid]
    print(f"After cleanup: {P.shape[0]} profiles "
          f"(pos={y.sum()}, neg={(y==0).sum()}, images={len(np.unique(src))})")

    X, feat_names = make_features(P)
    print(f"Features: {X.shape[1]} -> {feat_names}")

    imputer = SimpleImputer(strategy='median')
    # Single-model pipeline: CalibratedRF wraps a RandomForest with
    # sigmoid calibration (CV=5). Post-processing
    # (smoothing window, min run) and threshold are still re-determined for
    # the new training data via the grid search below.
    models = {
        'CalibratedRF': Pipeline([
            ('imp', imputer),
            ('clf', CalibratedClassifierCV(
                RandomForestClassifier(
                    n_estimators=600, max_depth=None, min_samples_leaf=2,
                    max_features='sqrt',
                    class_weight='balanced_subsample', random_state=42,
                    n_jobs=-1),
                cv=5, method='sigmoid'))]),
    }

    # Choose CV: only use GroupKFold if ≥2 images carry positives.
    unique_src = np.unique(src)
    n_img = len(unique_src)
    imgs_with_pos = [s for s in unique_src if y[src == s].sum() > 0]
    if len(imgs_with_pos) >= 2:
        # Leave-one-positive-image-out + round-robin neg-only images.
        # Each fold holds out ONE positive image (+ its own profiles) and
        # a share of neg-only images, so every sample appears in exactly
        # one test fold (cross_val_predict needs a partition).
        neg_only = [s for s in unique_src if s not in imgs_with_pos]
        # Split neg-only across folds round-robin
        neg_per_fold = [[] for _ in imgs_with_pos]
        for i, s in enumerate(neg_only):
            neg_per_fold[i % len(imgs_with_pos)].append(s)

        custom_splits = []
        all_idx = np.arange(len(y))
        for i, pos_held in enumerate(imgs_with_pos):
            test_imgs = {pos_held, *neg_per_fold[i]}
            test_mask = np.isin(src, list(test_imgs))
            train_mask = ~test_mask
            custom_splits.append((all_idx[train_mask], all_idx[test_mask]))
        cv = custom_splits
        groups = None
        print(f"\nUsing Leave-One-Positive-Image-Out "
              f"({len(imgs_with_pos)} folds)")
        for i, pos_held in enumerate(imgs_with_pos):
            print(f"  Fold {i}: test = [{pos_held}] + {neg_per_fold[i]}")
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        groups = None
        print(f"\nOnly {len(imgs_with_pos)} image(s) carry positives "
              f"→ using StratifiedKFold (5 splits)")

    results = []
    for name, pipe in models.items():
        print(f"\n── {name} ──")
        r = evaluate(name, pipe, X, y, cv, groups=groups)
        print(f"  ROC-AUC={r['roc_auc']:.3f}  AP={r['ap']:.3f}")
        print(f"  @0.5:  F1={r['f1@0.5']:.3f}  P={r['precision@0.5']:.3f}  "
              f"R={r['recall@0.5']:.3f}")
        print(f"  @best(t={r['best_threshold']:.2f}):  "
              f"F1={r['f1@best']:.3f}  P={r['precision@best']:.3f}  "
              f"R={r['recall@best']:.3f}")
        results.append(r)

    # per-model per-image breakdown
    print("\n── Per-model per-image breakdown (@ each model's best threshold) ──")
    for r in results:
        print(f"\n[{r['name']}] threshold={r['best_threshold']:.2f}")
        for s in unique_src:
            m = src == s
            npos = y[m].sum()
            nneg = (y[m] == 0).sum()
            tp = ((r['pred_best'][m] == 1) & (y[m] == 1)).sum()
            fp = ((r['pred_best'][m] == 1) & (y[m] == 0)).sum()
            fn = ((r['pred_best'][m] == 0) & (y[m] == 1)).sum()
            print(f"  {s:40s} pos={npos:4d} neg={nneg:4d}  "
                  f"TP={tp:4d} FP={fp:4d} FN={fn:4d}")

    best = max(results, key=lambda d: d['f1@best'])
    print(f"\nBest model: {best['name']}  F1={best['f1@best']:.3f}  "
          f"threshold={best['best_threshold']:.2f}")

    # ── Post-processing grid sweep (smooth_w × min_run × threshold) ──
    # Full grid search over post-processing params on raw CV proba.
    # CLI --smooth_w / --min_run override and skip the sweep when both set
    # explicitly (default sentinels = -1 mean "sweep").
    raw_proba = best['proba'].copy()
    ts = np.linspace(0.05, 0.95, 91)
    sweep_smooth = (1, 3, 5, 7, 9, 11, 15, 21)
    sweep_min_run = (1, 3, 5, 7, 9, 13, 17, 25)
    sweep_results = []
    print("\n── Post-processing grid sweep (smooth_w × min_run) ──")
    print("  smooth_w  min_run    F1     P     R    threshold")
    for w in sweep_smooth:
        sm = (smooth_proba_along_trace(raw_proba, trace_idx, src, window=w)
              if w > 1 else raw_proba)
        for mr in sweep_min_run:
            best_f1_pp = -1.0; t_pp = 0.5; p_pp = r_pp = 0.0
            for t in ts:
                pred_t = (sm >= t).astype(int)
                if mr > 1:
                    pred_t = apply_min_run(pred_t, trace_idx, src, min_run=mr)
                f1 = f1_score(y, pred_t, zero_division=0)
                if f1 > best_f1_pp:
                    best_f1_pp = f1
                    t_pp = t
                    p_pp = precision_score(y, pred_t, zero_division=0)
                    r_pp = recall_score(y, pred_t, zero_division=0)
            sweep_results.append((best_f1_pp, w, mr, t_pp, p_pp, r_pp))
            print(f"  {w:8d}  {mr:7d}  {best_f1_pp:.3f}  {p_pp:.3f}  "
                  f"{r_pp:.3f}    {t_pp:.2f}")

    sweep_results.sort(key=lambda x: -x[0])
    print("\n── Top 10 (smooth_w, min_run) by F1 ──")
    for f1v, w, mr, t, pv, rv in sweep_results[:10]:
        print(f"  smooth_w={w:2d}  min_run={mr:2d}  F1={f1v:.4f}  "
              f"P={pv:.3f}  R={rv:.3f}  t={t:.2f}")

    if args.smooth_w >= 0 and args.min_run >= 0:
        best_pp_w = args.smooth_w
        best_min_run = args.min_run
        # find best threshold for the forced (w, mr)
        sm_final = (smooth_proba_along_trace(raw_proba, trace_idx, src,
                                              window=best_pp_w)
                    if best_pp_w > 1 else raw_proba)
        best_f1 = -1.0; t_final = 0.5
        for t in ts:
            pred_t = (sm_final >= t).astype(int)
            if best_min_run > 1:
                pred_t = apply_min_run(pred_t, trace_idx, src,
                                        min_run=best_min_run)
            f1 = f1_score(y, pred_t, zero_division=0)
            if f1 > best_f1:
                best_f1, t_final = f1, t
        print(f"\n── Forced post-proc: smooth_w={best_pp_w}, "
              f"min_run={best_min_run}, threshold={t_final:.2f}, "
              f"F1={best_f1:.4f} ──")
    else:
        best_f1, best_pp_w, best_min_run, t_final, _, _ = sweep_results[0]
        sm_final = (smooth_proba_along_trace(raw_proba, trace_idx, src,
                                              window=best_pp_w)
                    if best_pp_w > 1 else raw_proba)
        print(f"\n── Auto-selected: smooth_w={best_pp_w}, "
              f"min_run={best_min_run}, threshold={t_final:.2f}, "
              f"F1={best_f1:.4f} ──")

    pred_final = (sm_final >= t_final).astype(int)
    if best_min_run > 1:
        pred_final = apply_min_run(pred_final, trace_idx, src,
                                    min_run=best_min_run)
    f1_final = best_f1

    # per-image breakdown for final
    print("Per-image breakdown (final, post-processed):")
    for s in unique_src:
        m = src == s
        npos = y[m].sum()
        nneg = (y[m] == 0).sum()
        tp = ((pred_final[m] == 1) & (y[m] == 1)).sum()
        fp = ((pred_final[m] == 1) & (y[m] == 0)).sum()
        rec = tp / npos if npos else float('nan')
        print(f"  {s:40s} pos={npos:4d} neg={nneg:4d}  "
              f"TP={tp:4d} FP={fp:4d}  recall={rec:.3f}")

    # overwrite best['proba'], best['pred_best'] etc for downstream saving
    best['proba'] = sm_final
    best['pred_best'] = pred_final
    best['best_threshold'] = t_final
    best['f1@best'] = f1_final
    best['post_process'] = {'smooth_w': best_pp_w,
                             'min_run': best_min_run}

    cm = confusion_matrix(y, best['pred_best'])
    print("Confusion matrix [best model @ best threshold]:")
    print("           pred_neg  pred_pos")
    print(f"true_neg   {cm[0,0]:>8d}  {cm[0,1]:>8d}")
    print(f"true_pos   {cm[1,0]:>8d}  {cm[1,1]:>8d}")

    # per-image breakdown
    print("\nPer-image recall / FP breakdown (best @ best threshold):")
    for s in unique_src:
        m = src == s
        npos = y[m].sum()
        nneg = (y[m] == 0).sum()
        tp = ((best['pred_best'][m] == 1) & (y[m] == 1)).sum()
        fp = ((best['pred_best'][m] == 1) & (y[m] == 0)).sum()
        rec = tp / npos if npos else float('nan')
        print(f"  {s:40s}  pos={npos:4d} neg={nneg:4d}  "
              f"TP={tp:4d} FP={fp:4d}  recall={rec:.3f}")

    # retrain best on all data + save
    final_pipe = models[best['name']]
    final_pipe.fit(X, y)
    input_type = 'pixel' if args.pixel else 'nm_resampled'
    pp_save = best.get('post_process', {})
    if input_type == 'nm_resampled':
        # Auto-convert best smooth_w/min_run (in trace-px) to nm using the
        # median nm/px of the training images, so predict.py (which expects
        # smooth_nm / min_run_nm keys) reproduces the trained post-processing.
        median_nmpx = float(np.median(list(pxnm_config.values())))
        pp_save.setdefault(
            'smooth_nm', round(pp_save.get('smooth_w', 7) * median_nmpx, 3))
        pp_save.setdefault(
            'min_run_nm', round(pp_save.get('min_run', 4) * median_nmpx, 3))
        if args.smooth_nm is not None:
            pp_save['smooth_nm'] = args.smooth_nm
        if args.min_run_nm is not None:
            pp_save['min_run_nm'] = args.min_run_nm
    dump({'pipeline': final_pipe,
          'feature_names': feat_names,
          'threshold': float(best['best_threshold']),
          'model_name': best['name'],
          'post_process': pp_save,
          'input_type': input_type,
          'nm_range': NM_RANGE, 'nm_samples': NM_SAMPLES}, out_model)
    print(f"\nSaved model -> {out_model}")

    clf = final_pipe.named_steps['clf']
    if hasattr(clf, 'feature_importances_'):
        imps = clf.feature_importances_
        order = np.argsort(imps)[::-1]
        print("\nFeature importances (top 10):")
        for i in order[:10]:
            print(f"  {feat_names[i]:25s} {imps[i]:.4f}")

    with open(out_report, 'w') as f:
        f.write("Profile classifier report\n")
        f.write("=========================\n")
        f.write(f"Total profiles: {P.shape[0]} "
                f"({y.sum()} pos, {(y==0).sum()} neg) "
                f"across {n_img} images\n")
        f.write(f"Features ({len(feat_names)}): {feat_names}\n\n")
        for r in results:
            f.write(f"[{r['name']}]\n")
            f.write(f"  ROC-AUC={r['roc_auc']:.3f}  AP={r['ap']:.3f}\n")
            f.write(f"  @0.5 F1={r['f1@0.5']:.3f} P={r['precision@0.5']:.3f} "
                    f"R={r['recall@0.5']:.3f}\n")
            f.write(f"  @best(t={r['best_threshold']:.2f}) "
                    f"F1={r['f1@best']:.3f} P={r['precision@best']:.3f} "
                    f"R={r['recall@best']:.3f}\n\n")
        f.write(f"Best: {best['name']}\n")
        f.write(f"Confusion matrix [best @ best threshold]:\n{cm}\n\n")
        f.write("Classification report [best @ best threshold]:\n")
        f.write(classification_report(y, best['pred_best'],
                                      target_names=['non-yellow', 'yellow'],
                                      zero_division=0))
    print(f"Saved report -> {out_report}")

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = 'predictions'
    ws.append(['Image', 'Trace_idx', 'true_yellow', 'proba',
               'pred@best_threshold'])
    for i in range(len(y)):
        ws.append([str(src[i]), int(trace_idx[i]), int(y[i]),
                   float(best['proba'][i]), int(best['pred_best'][i])])

    # ── edge width per positive run (true_yellow and pred) per image ──
    # Uses the loaded profile matrix P directly — profiles are already
    # per-trace-point perpendicular cross-sections.
    from dna_edge_width import measure_edge_width_subpixel

    W_prof = P.shape[1]
    if input_type == 'nm_resampled':
        distances = np.linspace(-args.nm_range, args.nm_range, W_prof)
        width_unit = 'nm'
    else:
        distances = (np.arange(W_prof) - W_prof // 2).astype(float)
        width_unit = 'px'

    pred_final = best['pred_best']

    runs_ws = wb.create_sheet('runs')
    header = [
        'Image', 'signal_type', 'Run_idx',
        'Start_trace_idx', 'End_trace_idx', 'Length', 'N_valid_profiles',
        f'Edge_width_{width_unit}',
    ]
    if width_unit != 'nm':
        header.append('Edge_width_nm')
    header += [f'Left_edge_{width_unit}', f'Right_edge_{width_unit}', 'pxnm']
    runs_ws.append(header)

    def _runs(binary):
        b = np.concatenate([[0], binary.astype(int), [0]])
        d = np.diff(b)
        starts = np.where(d == 1)[0]
        ends = np.where(d == -1)[0]
        return list(zip(starts.tolist(), ends.tolist()))

    n_runs_written = 0
    for img_name in np.unique(src):
        mask = src == img_name
        order = np.argsort(trace_idx[mask])
        ti_img = trace_idx[mask][order].astype(int)
        y_img = y[mask][order].astype(int)
        pr_img = pred_final[mask][order].astype(int)
        P_img = P[mask][order]

        px_nm_img = None
        if pxnm_config is not None:
            px_nm_img = pxnm_config.get(str(img_name))
            if px_nm_img is None:
                raise SystemExit(
                    f"Error: no entry for image '{img_name}' in "
                    f"pxum_config.json")

        for sig_name, binary in (('true', y_img), ('pred', pr_img)):
            for r_idx, (s, e) in enumerate(_runs(binary)):
                seg = P_img[s:e]
                valid = ~np.any(np.isnan(seg), axis=1)
                n_valid = int(valid.sum())
                w_u = le_u = re_u = None
                if n_valid >= 3:
                    avg_prof = seg[valid].mean(axis=0)
                    # σ=1 sample (~3.25 nm) Gaussian-smooth before derivative
                    # (DoG edge detection) — suppresses noise-driven far-out
                    # gradient extrema on short-run averaged profiles.
                    w_u, le_u, re_u, _ = measure_edge_width_subpixel(
                        distances, avg_prof, smooth_sigma=1)

                if w_u is not None:
                    if width_unit == 'nm':
                        w_nm = w_u
                    elif px_nm_img is not None:
                        w_nm = w_u * px_nm_img
                    else:
                        w_nm = None
                else:
                    w_nm = None

                row = [
                    str(img_name), sig_name, r_idx,
                    int(ti_img[s]), int(ti_img[e - 1]), int(e - s), n_valid,
                    round(float(w_u), 4) if w_u is not None else None,
                ]
                if width_unit != 'nm':
                    row.append(round(float(w_nm), 4) if w_nm is not None else None)
                row += [
                    round(float(le_u), 4) if le_u is not None else None,
                    round(float(re_u), 4) if re_u is not None else None,
                    px_nm_img,
                ]
                runs_ws.append(row)
                n_runs_written += 1

    wb.save(out_pred)
    print(f"Saved predictions -> {out_pred}")
    print(f"  runs sheet: {n_runs_written} rows (true + pred positive runs)")

    # ── Generate per-image overlay PNGs for each training image ──
    # Uses the freshly-saved joblib + the heatmap labels (true_yellow) so the
    # overlays show TP/FP/FN of the trained model on each training image.
    print("\n── Generating training overlays (TP/FP/FN per image) ──")
    try:
        import predict as _predict_mod
        from joblib import load as _joblib_load
        model_info = _joblib_load(out_model)
        ep_cache = _predict_mod.load_endpoints_cache()
        tif_dir = os.path.join(HERE, 'tif')
        overlay_dir = os.path.join(HERE, 'overlay')
        os.makedirs(overlay_dir, exist_ok=True)

        n_made = 0
        for img_name in np.unique(src):
            tif_path = os.path.join(tif_dir, f'{img_name}.tif')
            if not os.path.exists(tif_path):
                print(f"  ! {img_name}: tif not found in {tif_dir}, skip")
                continue
            mask = src == img_name
            order = np.argsort(trace_idx[mask])
            ti_img = trace_idx[mask][order].astype(int)
            y_img = y[mask][order].astype(int)
            truth = (ti_img, y_img)

            px_nm = pxnm_config.get(str(img_name)) if pxnm_config else None
            ep = ep_cache.get(str(img_name))
            try:
                _predict_mod.process(
                    tif_path, model_info,
                    pxnm=px_nm, out_dir=overlay_dir,
                    truth=truth, endpoints=ep)
                n_made += 1
            except Exception as exc:
                print(f"  ! {img_name}: overlay failed — {exc}")
        print(f"  saved {n_made} training overlay PNGs -> {overlay_dir}")
    except Exception as exc:
        print(f"  overlay generation skipped: {exc}")


if __name__ == '__main__':
    main()
