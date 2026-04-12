"""
DNA Edge-to-Edge Width Measurement from SEM/TEM images
- Uses the same DNA trace as dna_backbone_trace.py
- Measures width by finding max-gradient edges (steepest intensity change)
  on each side of the DNA backbone, rather than Gaussian FWHM
- More consistent with visual edge perception

Usage:
    python dna_edge_width.py <image_path> [options]

Example:
    python dna_edge_width.py image.tif
    python dna_edge_width.py image.tif --hfw 4.14
"""

import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter1d, map_coordinates
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import openpyxl


from dna_backbone_trace import (
    compute_line_strength, dp_trace, dp_trace_intensity,
    auto_select_trace, trim_edge_artifacts, measure_intensity, get_pxnm,
)
def gaussian(x, amp, mu, sigma, baseline):
    """Inverted Gaussian: baseline - amp * exp(-(x-mu)^2 / (2*sigma^2))"""
    return baseline - amp * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


def fit_gaussian_single(distances, profile, half_width_bound=15.0):
    """Fit inverted Gaussian to a cross-section profile.
    Returns (popt, fwhm) or (None, None)."""
    from scipy.optimize import curve_fit
    baseline0 = np.mean([profile[:3].mean(), profile[-3:].mean()])
    amp0 = max(baseline0 - profile.min(), 0.5)
    mu0 = distances[np.argmin(profile)]
    sigma0 = 2.0
    try:
        popt, _ = curve_fit(
            gaussian, distances, profile,
            p0=[amp0, mu0, sigma0, baseline0],
            bounds=([0, distances[0], 0.3, 0], [300, distances[-1], half_width_bound, 300]),
            maxfev=5000,
        )
        fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * abs(popt[2])
        return popt, fwhm
    except (RuntimeError, ValueError):
        return None, None


def _fit_gaussian_profile(distances, profile, half_width_bound=15.0):
    """Fit inverted Gaussian. Returns (popt, fitted_curve) or (None, None)."""
    popt, _ = fit_gaussian_single(distances, profile, half_width_bound)
    if popt is not None:
        return popt, gaussian(distances, *popt)
    return None, None


def extract_perpendicular_profiles(gray, cx, cy, half_width=20):
    """Extract perpendicular intensity profiles at each trace point."""
    n = len(cx)
    profile_len = 2 * half_width + 1
    profiles = np.full((n, profile_len), np.nan)
    distances = np.arange(-half_width, half_width + 1, dtype=float)

    for i in range(n):
        i_left = max(0, i - 1)
        i_right = min(n - 1, i + 1)
        dx = cx[i_right] - cx[i_left]
        dy = cy[i_right] - cy[i_left]

        length = np.sqrt(dx ** 2 + dy ** 2)
        if length < 1e-6:
            continue
        nx = -dy / length
        ny = dx / length

        sample_x = cx[i] + distances * nx
        sample_y = cy[i] + distances * ny

        h, w = gray.shape
        valid = (sample_x >= 0) & (sample_x < w - 1) & (sample_y >= 0) & (sample_y < h - 1)
        if not valid.all():
            continue

        coords = np.array([sample_y, sample_x])
        profiles[i, :] = map_coordinates(gray, coords, order=1, mode='reflect')

    return profiles, distances



def measure_edge_width_subpixel(distances, profile, smooth_sigma=0):
    """
    Measure DNA width with sub-pixel accuracy by fitting parabolas
    around the gradient extrema.

    Returns:
        width: edge-to-edge width in pixels (sub-pixel)
        left_edge: left edge position (px, sub-pixel)
        right_edge: right edge position (px, sub-pixel)
        gradient: smoothed gradient array
    """
    smoothed = gaussian_filter1d(profile, sigma=smooth_sigma) if smooth_sigma > 0 else profile.copy()
    gradient = np.gradient(smoothed, distances)

    center_idx = np.argmin(smoothed)

    # Left edge: most negative gradient
    left_region = gradient[:center_idx]
    if len(left_region) < 3:
        return None, None, None, gradient
    left_peak_idx = np.argmin(left_region)

    # Sub-pixel refinement via parabola fit around the peak
    left_edge = _refine_peak(distances, gradient, left_peak_idx)

    # Right edge: most positive gradient
    right_region = gradient[center_idx:]
    if len(right_region) < 3:
        return None, None, None, gradient
    right_peak_idx = center_idx + np.argmax(right_region)

    right_edge = _refine_peak(distances, gradient, right_peak_idx)

    if left_edge is None or right_edge is None:
        return None, None, None, gradient

    width = right_edge - left_edge
    return width, left_edge, right_edge, gradient


def _refine_peak(distances, gradient, peak_idx):
    """Refine peak position with parabolic interpolation."""
    if peak_idx <= 0 or peak_idx >= len(gradient) - 1:
        return distances[peak_idx]

    x0, x1, x2 = distances[peak_idx - 1], distances[peak_idx], distances[peak_idx + 1]
    y0, y1, y2 = gradient[peak_idx - 1], gradient[peak_idx], gradient[peak_idx + 1]

    denom = 2 * (y0 - 2 * y1 + y2)
    if abs(denom) < 1e-10:
        return x1

    refined = x1 - (y2 - y0) * (x2 - x1) / denom
    # Sanity check: must be between x0 and x2
    if refined < x0 or refined > x2:
        return x1
    return refined


def create_edge_width_output(distances, avg_profile, avg_gradient,
                              width, left_edge, right_edge,
                              px_size=None, output_path="edge_width_profile.png",
                              segment_profiles=None, segment_widths=None,
                              segment_labels=None, smooth_sigma=0):
    """Create output plot showing profile, gradient, and edge positions."""
    if px_size:
        dist_plot = distances * px_size * 1000
        x_label = "Distance from backbone center (nm)"
        width_nm = width * px_size * 1000
        left_nm = left_edge * px_size * 1000
        right_nm = right_edge * px_size * 1000
        width_label = f"Edge width = {width:.1f} px ({width_nm:.1f} nm)"
    else:
        dist_plot = distances
        x_label = "Distance from backbone center (px)"
        left_nm = left_edge
        right_nm = right_edge
        width_label = f"Edge width = {width:.1f} px"

    has_segments = (segment_profiles is not None and len(segment_profiles) > 0)

    if has_segments:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        ax_prof, ax_grad = axes[0]
        ax_seg, ax_seg_w = axes[1]
    else:
        fig, (ax_prof, ax_grad) = plt.subplots(1, 2, figsize=(14, 5))

    # 1) Average profile with edge markers
    ax_prof.plot(dist_plot, avg_profile, 'ko', markersize=2, alpha=0.6, label='Average profile')
    smoothed = gaussian_filter1d(avg_profile, sigma=smooth_sigma) if smooth_sigma > 0 else avg_profile.copy()
    ax_prof.plot(dist_plot, smoothed, 'b-', linewidth=1.5, alpha=0.5, label='Smoothed')
    ax_prof.axvline(left_nm, color='red', linestyle='--', linewidth=1.5, label=f'Left edge')
    ax_prof.axvline(right_nm, color='red', linestyle='--', linewidth=1.5, label=f'Right edge')
    ax_prof.axvline(0, color='gray', linestyle=':', alpha=0.4)

    # Shade the DNA width region
    ax_prof.axvspan(left_nm, right_nm, alpha=0.15, color='red', label=width_label)

    ax_prof.set_xlabel(x_label, fontsize=11)
    ax_prof.set_ylabel("Intensity (8-bit)", fontsize=11)
    ax_prof.set_title("DNA Cross-Section Profile (edge-to-edge)", fontsize=13, fontweight='bold')
    ax_prof.legend(loc='best', fontsize=8, framealpha=0.9)
    ax_prof.grid(True, alpha=0.3)

    # 2) Gradient plot
    if px_size:
        grad_plot = avg_gradient / (px_size * 1000)  # per nm
        grad_label = "dI/dx (per nm)"
    else:
        grad_plot = avg_gradient
        grad_label = "dI/dx (per px)"

    ax_grad.plot(dist_plot, grad_plot, 'k-', linewidth=1.5)
    ax_grad.axvline(left_nm, color='red', linestyle='--', linewidth=1.5)
    ax_grad.axvline(right_nm, color='red', linestyle='--', linewidth=1.5)
    ax_grad.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax_grad.axvline(0, color='gray', linestyle=':', alpha=0.4)
    ax_grad.set_xlabel(x_label, fontsize=11)
    ax_grad.set_ylabel(grad_label, fontsize=11)
    ax_grad.set_title("Intensity Gradient (edges = extrema)", fontsize=13, fontweight='bold')
    ax_grad.grid(True, alpha=0.3)

    # 3) Segment profiles
    if has_segments:
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(segment_profiles)))
        for i, (sp, sw, sl) in enumerate(zip(segment_profiles, segment_widths, segment_labels)):
            if px_size and sw is not None:
                label = f"{sl}: {sw:.1f}px ({sw * px_size * 1000:.1f}nm)"
            elif sw is not None:
                label = f"{sl}: {sw:.1f}px"
            else:
                label = f"{sl}: failed"
            ax_seg.plot(dist_plot, sp, color=colors[i], linewidth=1.2, label=label)
        ax_seg.set_xlabel(x_label, fontsize=11)
        ax_seg.set_ylabel("Intensity (8-bit)", fontsize=11)
        ax_seg.set_title("Segment Cross-Sections", fontsize=13, fontweight='bold')
        ax_seg.legend(loc='best', fontsize=7, framealpha=0.9)
        ax_seg.grid(True, alpha=0.3)

        # 4) Segment widths bar chart
        valid_segs = [(sl, sw) for sl, sw in zip(segment_labels, segment_widths) if sw is not None]
        if valid_segs:
            seg_names = [s[0] for s in valid_segs]
            seg_vals = [s[1] for s in valid_segs]
            if px_size:
                seg_vals_plot = [v * px_size * 1000 for v in seg_vals]
                ylabel = "Edge width (nm)"
            else:
                seg_vals_plot = seg_vals
                ylabel = "Edge width (px)"
            ax_seg_w.bar(range(len(seg_names)), seg_vals_plot, color=colors[:len(seg_names)])
            ax_seg_w.set_xticks(range(len(seg_names)))
            ax_seg_w.set_xticklabels(seg_names, fontsize=8, rotation=30)
            ax_seg_w.set_ylabel(ylabel, fontsize=11)
            ax_seg_w.set_title("Segment Edge Widths", fontsize=13, fontweight='bold')
            ax_seg_w.grid(True, alpha=0.3, axis='y')
            # Overall average line
            avg_w = np.mean(seg_vals_plot)
            ax_seg_w.axhline(avg_w, color='red', linestyle='--', alpha=0.7,
                            label=f'Mean={avg_w:.1f}')
            ax_seg_w.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return output_path


def create_overlay_image(img_rgb, cx, cy, width, px_size=None,
                          y_offset=15, output_path="edge_width_overlay.png"):
    """Create overlay showing trace + edge width band."""
    canvas = img_rgb.copy()
    pil_out = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_out)

    if width is not None:
        hw = width / 2.0

        overlay = Image.new('RGBA', pil_out.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        for i in range(len(cx)):
            y_top = int(cy[i] + y_offset - hw)
            y_bot = int(cy[i] + y_offset + hw)
            overlay_draw.line([(int(cx[i]), y_top), (int(cx[i]), y_bot)],
                              fill=(0, 100, 255, 40), width=1)
        pil_out = Image.alpha_composite(pil_out.convert('RGBA'), overlay)
        draw = ImageDraw.Draw(pil_out)

    # Red trace
    for i in range(len(cx) - 1):
        draw.line(
            [(int(cx[i]), int(cy[i] + y_offset)),
             (int(cx[i + 1]), int(cy[i + 1] + y_offset))],
            fill=(255, 0, 0), width=2
        )

    # Use matplotlib's bundled DejaVu Sans for cross-platform compatibility
    _font_path = os.path.join(
        os.path.dirname(matplotlib.__file__),
        "mpl-data", "fonts", "ttf", "DejaVuSans.ttf")
    try:
        font = ImageFont.truetype(_font_path, 14)
    except Exception:
        font = ImageFont.load_default()

    if px_size and width is not None:
        label = f"Edge width: {width:.1f} px ({width * px_size * 1000:.1f} nm)"
    elif width is not None:
        label = f"Edge width: {width:.1f} px"
    else:
        label = "Edge detection failed"
    draw.text((10, 10), label, fill=(255, 255, 0), font=font)

    pil_out.convert('RGB').save(output_path)
    return output_path


def export_edge_width_excel(distances, avg_profile, avg_gradient,
                             width, left_edge, right_edge,
                             px_size=None, n_profiles=0, smooth_sigma=0,
                             segment_widths=None, segment_labels=None,
                             segment_profiles=None, segment_left_edges=None,
                             segment_right_edges=None,
                             output_path="edge_width_data.xlsx"):
    """Export edge width measurement data to Excel."""
    wb = openpyxl.Workbook()

    # Sheet 1: Cross-section profile + gradient + Gaussian fit
    ws_prof = wb.active
    ws_prof.title = "Cross_Section"
    headers = ["Distance (px)", "Intensity", "Intensity (smoothed)",
               "Gaussian Fit", "Gradient"]
    if px_size:
        headers.insert(1, "Distance (nm)")
    ws_prof.append(headers)

    smoothed_profile = gaussian_filter1d(avg_profile, sigma=smooth_sigma) if smooth_sigma > 0 else avg_profile.copy()
    hw_bound = float(len(distances) // 2)
    _, avg_fit = _fit_gaussian_profile(distances, avg_profile, hw_bound)

    for i in range(len(distances)):
        col = 1
        ws_prof.cell(row=i + 2, column=col, value=round(float(distances[i]), 4)); col += 1
        if px_size:
            ws_prof.cell(row=i + 2, column=col, value=round(float(distances[i] * px_size * 1000), 4)); col += 1
        ws_prof.cell(row=i + 2, column=col, value=round(float(avg_profile[i]), 4)); col += 1
        ws_prof.cell(row=i + 2, column=col, value=round(float(smoothed_profile[i]), 4)); col += 1
        ws_prof.cell(row=i + 2, column=col,
                     value=round(float(avg_fit[i]), 4) if avg_fit is not None else ""); col += 1
        ws_prof.cell(row=i + 2, column=col, value=round(float(avg_gradient[i]), 4))

    # Sheet 2: Results
    ws_res = wb.create_sheet("Results")
    ws_res.append(["Parameter", "Value"])
    ws_res.append(["Method", "Edge-to-edge (max gradient)"])
    ws_res.append(["Smooth sigma (px)", round(smooth_sigma, 2)])
    ws_res.append(["N profiles averaged", n_profiles])
    ws_res.append([])
    if width is not None:
        ws_res.append(["Edge width (px)", round(width, 4)])
        if px_size:
            ws_res.append(["Edge width (nm)", round(width * px_size * 1000, 4)])
        ws_res.append(["Left edge (px)", round(left_edge, 4)])
        ws_res.append(["Right edge (px)", round(right_edge, 4)])
        if px_size:
            ws_res.append(["Left edge (nm)", round(left_edge * px_size * 1000, 4)])
            ws_res.append(["Right edge (nm)", round(right_edge * px_size * 1000, 4)])
    else:
        ws_res.append(["Edge detection failed", ""])

    # Sheet 3: Segments
    if segment_widths is not None and segment_profiles is not None:
        ws_seg = wb.create_sheet("Segments")

        # Data columns
        headers_seg = ["Distance (px)"]
        if px_size:
            headers_seg.append("Distance (nm)")
        for sl in segment_labels:
            headers_seg.append(f"{sl} Data")
            headers_seg.append(f"{sl} Smoothed")
            headers_seg.append(f"{sl} Gaussian Fit")
            headers_seg.append(f"{sl} Gradient")
        ws_seg.append(headers_seg)

        # Compute segment smoothed, Gaussian fits, and gradients
        seg_smoothed = []
        seg_fits = []
        seg_gradients = []
        for sp in segment_profiles:
            smoothed = gaussian_filter1d(sp, sigma=smooth_sigma) if smooth_sigma > 0 else sp.copy()
            seg_smoothed.append(smoothed)
            seg_gradients.append(np.gradient(smoothed, distances))
            _, seg_fit = _fit_gaussian_profile(distances, sp, hw_bound)
            seg_fits.append(seg_fit)

        for i in range(len(distances)):
            col = 1
            ws_seg.cell(row=i + 2, column=col, value=round(float(distances[i]), 4)); col += 1
            if px_size:
                ws_seg.cell(row=i + 2, column=col, value=round(float(distances[i] * px_size * 1000), 4)); col += 1
            for s_idx in range(len(segment_profiles)):
                ws_seg.cell(row=i + 2, column=col, value=round(float(segment_profiles[s_idx][i]), 4)); col += 1
                ws_seg.cell(row=i + 2, column=col, value=round(float(seg_smoothed[s_idx][i]), 4)); col += 1
                ws_seg.cell(row=i + 2, column=col,
                             value=round(float(seg_fits[s_idx][i]), 4) if seg_fits[s_idx] is not None else ""); col += 1
                ws_seg.cell(row=i + 2, column=col, value=round(float(seg_gradients[s_idx][i]), 4)); col += 1

        # Width summary
        summary_row = len(distances) + 3
        ws_seg.cell(row=summary_row, column=1, value="Edge Width Summary")
        summary_row += 1
        w_headers = ["Segment", "Width (px)", "Left edge (px)", "Right edge (px)"]
        if px_size:
            w_headers.extend(["Width (nm)", "Left edge (nm)", "Right edge (nm)"])
        for c, h in enumerate(w_headers, 1):
            ws_seg.cell(row=summary_row, column=c, value=h)
        summary_row += 1
        for s_idx, sl in enumerate(segment_labels):
            sw = segment_widths[s_idx]
            sl_e = segment_left_edges[s_idx] if segment_left_edges else None
            sr_e = segment_right_edges[s_idx] if segment_right_edges else None
            ws_seg.cell(row=summary_row, column=1, value=sl)
            ws_seg.cell(row=summary_row, column=2, value=round(sw, 4) if sw is not None else "failed")
            ws_seg.cell(row=summary_row, column=3, value=round(sl_e, 4) if sl_e is not None else "")
            ws_seg.cell(row=summary_row, column=4, value=round(sr_e, 4) if sr_e is not None else "")
            if px_size:
                ws_seg.cell(row=summary_row, column=5, value=round(sw * px_size * 1000, 4) if sw is not None else "")
                ws_seg.cell(row=summary_row, column=6, value=round(sl_e * px_size * 1000, 4) if sl_e is not None else "")
                ws_seg.cell(row=summary_row, column=7, value=round(sr_e * px_size * 1000, 4) if sr_e is not None else "")
            summary_row += 1

    wb.save(output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="DNA Edge-to-Edge Width Measurement")
    parser.add_argument("image", help="Input image path (.tif)")
    parser.add_argument("pxum", type=float, nargs="?", default=None,
                        help="Default px/µm to apply if image missing from "
                             "pxum_config.json (auto-saved).")
    parser.add_argument("--y_band", type=int, nargs=2, default=None)
    parser.add_argument("--x_range", type=int, nargs=2, default=None)
    parser.add_argument("--trace_method", type=str, default="auto",
                        choices=["auto", "dp", "intensity"])
    parser.add_argument("--max_step", type=int, default=2)
    parser.add_argument("--template_sigma", type=float, default=1.5)
    parser.add_argument("--continuity_weight", type=float, default=1.0)
    parser.add_argument("--x_avg", type=int, default=1)
    parser.add_argument("--y_offset", type=int, default=15)
    parser.add_argument("--half_width", type=int, default=20,
                        help="Half-width of perpendicular profile in pixels (default: 20)")
    parser.add_argument("--smooth_sigma", type=float, default=0,
                        help="Gaussian smoothing sigma for gradient (default: 0, no smoothing)")
    parser.add_argument("--seg_size", type=int, default=100,
                        help="Segment size in x-pixels for regional analysis (default: 100)")
    parser.add_argument("--start_pt", type=int, nargs=2, default=None,
                        help="DNA start point [x y]")
    parser.add_argument("--end_pt", type=int, nargs=2, default=None,
                        help="DNA end point [x y]")
    parser.add_argument("--no-interactive", dest="interactive",
                        action="store_false", default=True,
                        help="Error instead of prompting when an image's "
                             "px/µm is missing from pxum_config.json")
    args = parser.parse_args()

    # Load image
    img = Image.open(args.image)
    img_rgb = np.array(img.convert('RGB'))
    gray = np.array(img.convert('L')).astype(float)
    h_full, w_full = gray.shape
    print(f"Image: {args.image} ({w_full} x {h_full})")

    sem = gray
    sem_h = h_full

    # Pixel scale: looked up from pxum_config.json (prompts if missing)
    image_name = os.path.splitext(os.path.basename(args.image))[0]
    nm_per_px = get_pxnm(image_name, interactive=args.interactive,
                         default_pxum=args.pxum)
    px_size = nm_per_px / 1000.0  # µm/pixel
    print(f"Pixel scale: {nm_per_px:.2f} nm/pixel "
          f"(from pxum_config.json)")

    # Compute trace
    print("Computing line strength map...")
    line_strength = compute_line_strength(sem)

    if args.start_pt is not None and args.end_pt is not None:
        sx, sy = args.start_pt
        ex, ey = args.end_pt
        if sx > ex:
            sx, sy, ex, ey = ex, ey, sx, sy
        y_margin = max(abs(sy - ey), 30)
        y_lo = max(0, min(sy, ey) - y_margin)
        y_hi = min(sem_h, max(sy, ey) + y_margin)
        x_start, x_end = sx, ex
        print(f"Start/End points: ({sx},{sy}) -> ({ex},{ey})")
        print(f"Y search band (from endpoints): [{y_lo}, {y_hi}]")
    else:
        if args.y_band is not None:
            y_lo, y_hi = args.y_band
        else:
            y_lo, y_hi = 0, sem_h - 1
        print(f"Y search band: [{y_lo}, {y_hi}]")

    if args.trace_method == "auto":
        print("Auto-selecting trace method...")
        trace_local, _ = auto_select_trace(
            gray, line_strength, y_lo, y_hi,
            max_step=args.max_step, template_sigma=args.template_sigma,
            continuity_weight=args.continuity_weight, x_avg=args.x_avg)
    elif args.trace_method == "intensity":
        print("Running intensity trace...")
        gray_band = gray[y_lo:y_hi, :]
        trace_local = dp_trace_intensity(gray_band,
                                          template_sigma=args.template_sigma,
                                          continuity_weight=args.continuity_weight,
                                          x_avg=args.x_avg)
    else:
        print("Running DP trace...")
        band = line_strength[y_lo:y_hi, :]
        trace_local = dp_trace(band, max_step=args.max_step)
    trace_y_full = trace_local + y_lo

    if args.start_pt is None or args.end_pt is None:
        if args.x_range is not None:
            x_start, x_end = args.x_range
        else:
            x_start, x_end = 0, w_full - 1
    print(f"X range: [{x_start}, {x_end}]")

    cx = np.arange(x_start, x_end + 1).astype(float)
    cy = trace_y_full[x_start:x_end + 1].astype(float)

    print(f"Trace length: {len(cx)} px")

    # Extract perpendicular profiles
    print("Extracting perpendicular cross-sections...")
    profiles, distances = extract_perpendicular_profiles(
        gray, cx, cy, half_width=args.half_width
    )

    valid_mask = ~np.any(np.isnan(profiles), axis=1)
    profiles_valid = profiles[valid_mask]
    n_valid = profiles_valid.shape[0]
    print(f"Valid cross-sections: {n_valid} / {len(cx)}")

    # Average profile
    avg_profile = profiles_valid.mean(axis=0)

    # Edge-to-edge width measurement
    print(f"Measuring edge-to-edge width (smooth_sigma={args.smooth_sigma})...")
    width, left_edge, right_edge, avg_gradient = measure_edge_width_subpixel(
        distances, avg_profile, smooth_sigma=args.smooth_sigma
    )

    if width is not None:
        if px_size:
            print(f"Edge width = {width:.2f} px ({width * px_size * 1000:.2f} nm)")
        else:
            print(f"Edge width = {width:.2f} px")
        print(f"  Left edge: {left_edge:.2f} px, Right edge: {right_edge:.2f} px")
    else:
        print("Edge detection failed.")

    # Segment analysis
    segment_profiles = []
    segment_widths = []
    segment_labels = []
    segment_left_edges = []
    segment_right_edges = []
    seg_size = args.seg_size
    n_segments = (n_valid + seg_size - 1) // seg_size  # ceiling division
    if n_segments > 1 and n_valid > seg_size:
        print(f"\nSegment analysis ({n_segments} segments, {seg_size}px each):")
        for s in range(n_segments):
            i_start = s * seg_size
            i_end = min(i_start + seg_size, n_valid)
            seg_avg = profiles_valid[i_start:i_end].mean(axis=0)
            seg_w, seg_le, seg_re, _ = measure_edge_width_subpixel(
                distances, seg_avg, smooth_sigma=args.smooth_sigma
            )
            segment_profiles.append(seg_avg)
            segment_widths.append(seg_w)
            segment_left_edges.append(seg_le)
            segment_right_edges.append(seg_re)
            label = f"Seg {s + 1} (n={i_end - i_start})"
            segment_labels.append(label)
            if seg_w is not None:
                if px_size:
                    print(f"  {label}: width = {seg_w:.2f} px ({seg_w * px_size * 1000:.2f} nm)")
                else:
                    print(f"  {label}: width = {seg_w:.2f} px")
            else:
                print(f"  {label}: failed")

    # Output files
    base = os.path.splitext(args.image)[0]

    profile_path = f"{base}_edge_width_profile.png"
    create_edge_width_output(
        distances, avg_profile, avg_gradient,
        width, left_edge, right_edge,
        px_size=px_size, output_path=profile_path,
        segment_profiles=segment_profiles if segment_profiles else None,
        segment_widths=segment_widths if segment_widths else None,
        segment_labels=segment_labels if segment_labels else None,
        smooth_sigma=args.smooth_sigma,
    )
    print(f"\nSaved: {profile_path}")

    overlay_path = f"{base}_edge_width_overlay.png"
    create_overlay_image(
        img_rgb, cx, cy, width,
        px_size=px_size, y_offset=args.y_offset,
        output_path=overlay_path,
    )
    print(f"Saved: {overlay_path}")

    excel_path = f"{base}_edge_width_data.xlsx"
    export_edge_width_excel(
        distances, avg_profile, avg_gradient,
        width, left_edge, right_edge,
        px_size=px_size, n_profiles=n_valid,
        smooth_sigma=args.smooth_sigma,
        segment_widths=segment_widths if segment_widths else None,
        segment_labels=segment_labels if segment_labels else None,
        segment_profiles=segment_profiles if segment_profiles else None,
        segment_left_edges=segment_left_edges if segment_left_edges else None,
        segment_right_edges=segment_right_edges if segment_right_edges else None,
        output_path=excel_path,
    )
    print(f"Saved: {excel_path}")

    # Summary
    print(f"\n{'=' * 50}")
    print(f"  DNA Edge Width Measurement Results")
    print(f"{'=' * 50}")
    if width is not None:
        print(f"  Edge width:          {width:.2f} px", end="")
        if px_size:
            print(f" = {width * px_size * 1000:.2f} nm")
        else:
            print()
        print(f"  Left edge:           {left_edge:.2f} px", end="")
        if px_size:
            print(f" = {left_edge * px_size * 1000:.2f} nm")
        else:
            print()
        print(f"  Right edge:          {right_edge:.2f} px", end="")
        if px_size:
            print(f" = {right_edge * px_size * 1000:.2f} nm")
        else:
            print()
        print(f"  N profiles averaged: {n_valid}")
    else:
        print("  Edge detection failed.")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
