"""
DNA Perpendicular Profile Heatmap
- x-axis: trace position along DNA
- y-axis: perpendicular distance from center
- color: intensity

Usage:
    python dna_profile_heatmap.py image.tif --hfw 4.14
    python dna_profile_heatmap.py ML_study/ --hfw 4.14
"""

import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dna_backbone_trace import (
    compute_line_strength, auto_select_trace, dp_trace, mask_round_blobs,
    get_or_pick_endpoints, ensure_pxum_entries,
)
from dna_edge_width import extract_perpendicular_profiles


def make_heatmap(gray, cx, cy, half_width, px_size, image_path,
                 output_dir=None, x_valid=None):
    """Generate and save profile heatmap for one image."""
    profiles, distances = extract_perpendicular_profiles(
        gray, cx, cy, half_width=half_width
    )

    # NaN out profiles outside endpoint range
    if x_valid is not None:
        x_start, x_end = x_valid
        for i in range(len(cx)):
            if cx[i] < x_start or cx[i] > x_end:
                profiles[i, :] = np.nan

    valid_mask = ~np.any(np.isnan(profiles), axis=1)
    profiles_valid = profiles.copy()
    n_valid = valid_mask.sum()
    print(f"  Valid profiles: {n_valid} / {len(cx)}")

    # NaN rows -> fill with column mean for display
    col_mean = np.nanmean(profiles, axis=0)
    for i in range(profiles_valid.shape[0]):
        if np.any(np.isnan(profiles_valid[i])):
            profiles_valid[i] = col_mean

    # x-axis: trace position, y-axis: perpendicular distance
    # profiles_valid shape: (n_trace, n_perp)
    # Transpose so y-axis = perpendicular distance
    data = profiles_valid.T  # (n_perp, n_trace)

    basename = os.path.splitext(os.path.basename(image_path))[0]
    if output_dir:
        out_base = os.path.join(output_dir, basename)
    else:
        out_base = os.path.splitext(image_path)[0]

    # ── Figure ──
    fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                             gridspec_kw={'height_ratios': [4, 1.5, 1.5]})

    # 1) Heatmap
    ax = axes[0]
    if px_size:
        extent = [0, len(cx) * px_size * 1000,
                  distances[-1] * px_size * 1000,
                  distances[0] * px_size * 1000]
        xlabel = "Trace position (nm)"
        ylabel = "Perpendicular distance (nm)"
    else:
        extent = [0, len(cx), distances[-1], distances[0]]
        xlabel = "Trace position (px)"
        ylabel = "Perpendicular distance (px)"

    im = ax.imshow(data, aspect='auto', cmap='inferno', extent=extent,
                   interpolation='bilinear')
    ax.axhline(0, color='cyan', linewidth=0.5, alpha=0.7, linestyle='--')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Perpendicular Profile Heatmap — {basename}")
    plt.colorbar(im, ax=ax, label="Intensity", shrink=0.8)

    # 2) Center intensity along trace
    ax2 = axes[1]
    center_idx = half_width  # center of profile
    center_intensity = profiles_valid[:, center_idx]
    x_positions = np.arange(len(cx))
    if px_size:
        x_nm = x_positions * px_size * 1000
        ax2.plot(x_nm, center_intensity, color='orange', linewidth=0.5)
        ax2.set_xlabel(xlabel)
    else:
        ax2.plot(x_positions, center_intensity, color='orange', linewidth=0.5)
        ax2.set_xlabel(xlabel)
    ax2.set_ylabel("Center intensity")
    ax2.set_title("Intensity along DNA center")

    # 3) Local width along trace (from gradient edge detection)
    ax3 = axes[2]
    from dna_edge_width import measure_edge_width_subpixel
    window = 50  # rolling window for local width
    local_widths = []
    x_centers = []
    for i in range(0, n_valid - window, window // 2):
        seg = profiles[valid_mask][i:i + window].mean(axis=0)
        w, le, re, _ = measure_edge_width_subpixel(distances, seg, smooth_sigma=1)
        if w is not None:
            local_widths.append(w)
            x_centers.append(i + window // 2)

    if local_widths:
        local_widths = np.array(local_widths)
        x_centers = np.array(x_centers)
        if px_size:
            ax3.plot(x_centers * px_size * 1000, local_widths * px_size * 1000,
                     color='lime', linewidth=1)
            ax3.set_ylabel("Edge width (nm)")
            ax3.set_xlabel(xlabel)
        else:
            ax3.plot(x_centers, local_widths, color='lime', linewidth=1)
            ax3.set_ylabel("Edge width (px)")
            ax3.set_xlabel(xlabel)
    ax3.set_title("Local edge width along trace")

    plt.tight_layout()
    png_path = f"{out_base}_profile_heatmap.png"
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {png_path}")

    # ── Excel: raw profiles ──
    import openpyxl
    xlsx_path = f"{out_base}_profile_heatmap.xlsx"
    wb = openpyxl.Workbook()

    # Sheet 1: full profiles
    ws = wb.active
    ws.title = "Profiles"
    header = ["Trace_idx"]
    if px_size:
        header.append("Trace_pos_nm")
    for d in distances:
        if px_size:
            header.append(f"{d * px_size * 1000:.1f}nm")
        else:
            header.append(f"{d:.0f}px")
    ws.append(header)

    for i in range(len(cx)):
        row = [int(cx[i])]
        if px_size:
            row.append(round(cx[i] * px_size * 1000, 2))
        for j in range(len(distances)):
            val = profiles[i, j]
            row.append(round(float(val), 4) if not np.isnan(val) else "")
        ws.append(row)

    # Sheet 2: summary
    ws2 = wb.create_sheet("Summary")
    ws2.append(["Image", basename])
    ws2.append(["N trace points", len(cx)])
    ws2.append(["N valid profiles", int(n_valid)])
    ws2.append(["Half width (px)", half_width])
    if px_size:
        ws2.append(["Pixel size (nm)", round(px_size * 1000, 2)])

    # Sheet 3: local widths
    if local_widths is not None and len(local_widths) > 0:
        ws3 = wb.create_sheet("Local_Width")
        if px_size:
            ws3.append(["Trace_pos_nm", "Edge_width_nm"])
            for xc, w in zip(x_centers, local_widths):
                ws3.append([round(xc * px_size * 1000, 2), round(w * px_size * 1000, 2)])
        else:
            ws3.append(["Trace_pos_px", "Edge_width_px"])
            for xc, w in zip(x_centers, local_widths):
                ws3.append([round(float(xc), 2), round(float(w), 4)])

    wb.save(xlsx_path)
    print(f"  Saved: {xlsx_path}")


def load_endpoints_cache(base_dir=None):
    """Load endpoints.json from project root."""
    if base_dir is None:
        base_dir = os.path.dirname(__file__)
    ep_path = os.path.join(base_dir, 'endpoints.json')
    if not os.path.exists(ep_path):
        return {}
    import json
    with open(ep_path) as f:
        data = json.load(f)
    out = {}
    for k, v in data.items():
        name = os.path.splitext(k)[0]
        out[name] = (v['start'], v['end'])
    return out


def process_image(image_path, args, endpoints=None, nm_per_px=None):
    """Process a single image."""
    print(f"\n{'='*60}")
    print(f"  {os.path.basename(image_path)}")
    print(f"{'='*60}")

    img = Image.open(image_path)
    gray = np.array(img.convert('L')).astype(float)
    h, w = gray.shape

    # px_size is in µm/pixel for downstream code (heatmap nm conversion etc.)
    if nm_per_px is None:
        raise SystemExit(f"Internal error: nm_per_px required for {image_path}")
    px_size = nm_per_px / 1000.0  # nm/px -> µm/px

    # Trace
    line_strength = compute_line_strength(gray)
    trace_local, _ = auto_select_trace(
        gray, line_strength, 0, h - 1,
        max_step=2, template_sigma=1.5, continuity_weight=1.0, x_avg=1)
    trace_y = trace_local

    # Full trace across entire image
    cx = np.arange(w).astype(float)
    cy = trace_y.astype(float)

    # Apply endpoints: mark valid range, NaN outside
    if endpoints is not None:
        start_pt, end_pt = endpoints
        x_start = min(int(start_pt[0]), int(end_pt[0]))
        x_end = max(int(start_pt[0]), int(end_pt[0]))
        print(f"  endpoints: ({start_pt[0]},{start_pt[1]}) -> "
              f"({end_pt[0]},{end_pt[1]})  x=[{x_start}, {x_end}]")
    else:
        x_start, x_end = 0, w - 1

    make_heatmap(gray, cx, cy, args.half_width, px_size, image_path,
                 output_dir=args.output_dir, x_valid=(x_start, x_end))


def main():
    parser = argparse.ArgumentParser(description="DNA Perpendicular Profile Heatmap")
    parser.add_argument("image", help="Input image (.tif) or folder")
    parser.add_argument("pxum", type=float, nargs="?", default=None,
                        help="Default px/µm to apply to any image missing "
                             "from pxum_config.json (auto-saved).")
    parser.add_argument("--half_width", type=int, default=20,
                        help="Half-width of perpendicular profile (default: 20)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: same as input)")
    parser.add_argument("--no-interactive", dest="interactive",
                        action="store_false", default=True,
                        help="Disable interactive endpoint picker for images "
                             "not in endpoints.json (use full-range instead)")
    args = parser.parse_args()

    ep_cache = load_endpoints_cache()
    if ep_cache:
        print(f"Loaded endpoints for {len(ep_cache)} images")

    def _resolve_endpoints(tif_path, name):
        if name in ep_cache:
            return ep_cache[name]
        img_gray = np.array(Image.open(tif_path).convert('L')).astype(float)
        ep = get_or_pick_endpoints(
            img_gray, name, interactive=args.interactive, title=name)
        ep_cache[name] = ep
        return ep

    if os.path.isdir(args.image):
        tifs = sorted(glob.glob(os.path.join(args.image, "*.tif")))
        if not tifs:
            print(f"No .tif files found in {args.image}")
            sys.exit(1)
        print(f"Found {len(tifs)} .tif files in {args.image}")
        if args.output_dir is None:
            args.output_dir = args.image
        names = [os.path.splitext(os.path.basename(t))[0] for t in tifs]
        pxum_cfg = ensure_pxum_entries(names, interactive=args.interactive,
                                        default_pxum=args.pxum)
        for t in tifs:
            name = os.path.splitext(os.path.basename(t))[0]
            process_image(t, args, endpoints=_resolve_endpoints(t, name),
                          nm_per_px=pxum_cfg[name])
    else:
        name = os.path.splitext(os.path.basename(args.image))[0]
        pxum_cfg = ensure_pxum_entries([name], interactive=args.interactive,
                                       default_pxum=args.pxum)
        process_image(args.image, args,
                      endpoints=_resolve_endpoints(args.image, name),
                      nm_per_px=pxum_cfg[name])

    print("\nDone!")


if __name__ == "__main__":
    main()
