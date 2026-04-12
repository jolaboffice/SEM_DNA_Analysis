"""
DNA Backbone Tracer for SEM/TEM images
- CLAHE enhancement → line strength map → Dynamic Programming trace
- Ridge detection → graph → Dijkstra trace (handles folds/kinks)
- Extracts intensity (Mean, SD, Min, Max) from ORIGINAL image
- Background sampled from above/below the trace
- Outputs: overlay image with data table

Usage:
    python dna_backbone_trace.py <image_path> [options]

Example:
    python dna_backbone_trace.py image.tif
    python dna_backbone_trace.py image.tif --hfw 4.14
    python dna_backbone_trace.py image.tif --y_band 370 450 --x_range 70 1060
    python dna_backbone_trace.py image.tif --trace_method ridge
"""

import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter, uniform_filter, uniform_filter1d, label, binary_dilation
from skimage import exposure
from skimage.filters import meijering
from skimage.morphology import disk
from skimage.measure import regionprops
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import openpyxl


def mask_round_blobs(gray, dark_thresh_percentile=3, min_area=80,
                     max_eccentricity=0.70, dilate_radius=3):
    """
    Detect and mask dark, round blobs (non-DNA noise) in the image.
    Replaces blob pixels with global median (background level).
    Blobs touching image edges are excluded.

    Returns:
        gray_clean: image with blobs replaced by median
        n_blobs: number of blobs detected
    """
    thresh = np.percentile(gray, dark_thresh_percentile)
    dark_mask = gray < thresh
    labeled_arr, n = label(dark_mask)
    blob_mask = np.zeros_like(dark_mask)
    h, w = gray.shape
    props = regionprops(labeled_arr)
    n_blobs = 0
    for p in props:
        r0, c0, r1, c1 = p.bbox
        if r0 == 0 or c0 == 0 or r1 == h or c1 == w:
            continue
        if p.area >= min_area and p.eccentricity < max_eccentricity:
            blob_mask[labeled_arr == p.label] = True
            n_blobs += 1
    if dilate_radius > 0 and np.any(blob_mask):
        blob_mask = binary_dilation(blob_mask, structure=disk(dilate_radius))
    gray_clean = gray.copy()
    gray_clean[blob_mask] = np.median(gray)
    return gray_clean, n_blobs



def compute_line_strength(sem, clahe_clip=0.03, offsets=(3, 4, 5), continuity_window=25):
    """
    Compute line strength map using CLAHE enhancement + vertical profile comparison.
    Dark horizontal lines get high scores.
    Uses continuity weighting: sustained horizontal lines are boosted over point noise.
    """
    sem_uint8 = sem.astype(np.uint8)
    enhanced = exposure.equalize_adapthist(sem_uint8, clip_limit=clahe_clip)
    enh = enhanced * 255.0

    enh_vs = gaussian_filter(enh, sigma=[0.5, 1])
    line_strength = np.zeros_like(enh_vs)

    for off in offsets:
        above = np.roll(enh_vs, -off, axis=0)
        below = np.roll(enh_vs, off, axis=0)
        darkness = ((above - enh_vs) + (below - enh_vs)) / 2.0
        line_strength += darkness

    margin = max(offsets) + 5
    line_strength[:margin, :] = 0
    line_strength[-margin:, :] = 0

    # Continuity weighting: horizontal sliding average rewards sustained lines
    ls_continuity = uniform_filter(line_strength, size=[1, continuity_window])

    ls_norm = line_strength / (line_strength.std() + 1e-6)
    cont_norm = ls_continuity / (ls_continuity.std() + 1e-6)
    combined = 0.4 * ls_norm + 0.6 * cont_norm

    return combined


def compute_line_strength_meijering(sem, sigmas=range(1, 5), continuity_window=25):
    """
    Compute line strength map using Meijering ridge filter + continuity weighting.
    Responds to line structures (DNA), not blobs (noise).
    """
    gray_norm = (sem - sem.min()) / (sem.max() - sem.min() + 1e-6)
    gray_inv = 1.0 - gray_norm  # dark lines → bright
    ridge = meijering(gray_inv, sigmas=sigmas, black_ridges=False)
    ridge = ridge / (ridge.max() + 1e-6)
    # Horizontal continuity weighting: sustained lines boosted over point noise
    ridge_cont = uniform_filter(ridge, size=[1, continuity_window])
    r_norm = ridge / (ridge.std() + 1e-6)
    c_norm = ridge_cont / (ridge_cont.std() + 1e-6)
    combined = 0.4 * r_norm + 0.6 * c_norm
    return combined


def _dp_one_direction(score_band, max_step, jump_penalty, reverse=False):
    """Run single-direction DP. If reverse=True, trace right-to-left."""
    if reverse:
        score_band = score_band[:, ::-1]

    band_h, band_w = score_band.shape
    cost = np.full((band_h, band_w), -1e9)
    back = np.zeros((band_h, band_w), dtype=int)
    cost[:, 0] = score_band[:, 0]

    for x in range(1, band_w):
        for y in range(band_h):
            y_lo = max(0, y - max_step)
            y_hi = min(band_h, y + max_step + 1)
            candidates = cost[y_lo:y_hi, x - 1]
            best_local = np.argmax(candidates)
            best_prev_y = y_lo + best_local
            penalty = -jump_penalty * abs(y - best_prev_y)
            cost[y, x] = score_band[y, x] + candidates[best_local] + penalty
            back[y, x] = best_prev_y

    # Backtrace
    trace = np.zeros(band_w, dtype=int)
    trace[-1] = np.argmax(cost[:, -1])
    for x in range(band_w - 2, -1, -1):
        trace[x] = back[trace[x + 1], x + 1]

    if reverse:
        trace = trace[::-1]
    return trace


def dp_trace(score_band, max_step=2, jump_penalty=0.5):
    """
    Bidirectional Dynamic Programming: average of left→right and right→left.
    Reduces local noise artifacts by combining both directions.
    """
    trace_fwd = _dp_one_direction(score_band, max_step, jump_penalty, reverse=False)
    trace_rev = _dp_one_direction(score_band, max_step, jump_penalty, reverse=True)
    trace_avg = np.round((trace_fwd.astype(float) + trace_rev.astype(float)) / 2).astype(int)
    return trace_avg


def dp_trace_intensity(gray_band, template_sigma=1.5, continuity_weight=1.0, x_avg=1):
    """
    Trace DNA using intensity-based Gaussian profile matching + continuity.
    Better for high-contrast images and sharp bends.

    Args:
        gray_band: 2D intensity array (y_band region of image)
        template_sigma: σ of Gaussian template in pixels (default: 1.5)
        continuity_weight: penalty per pixel of y-distance (default: 1.0)
        x_avg: pixels to average on each side of x (default: 1, total 3px)
    """
    trace_fwd = _intensity_trace_one_dir(gray_band, template_sigma,
                                          continuity_weight, x_avg, reverse=False)
    trace_rev = _intensity_trace_one_dir(gray_band, template_sigma,
                                          continuity_weight, x_avg, reverse=True)
    trace_avg = np.round((trace_fwd.astype(float) + trace_rev.astype(float)) / 2).astype(int)
    return trace_avg


def _intensity_trace_one_dir(gray_band, template_sigma, continuity_weight,
                             x_avg=1, reverse=False):
    """Single-direction intensity-based trace."""
    band_h, band_w = gray_band.shape
    if reverse:
        gray_band = gray_band[:, ::-1]

    half_t = int(template_sigma * 3)
    t = np.arange(-half_t, half_t + 1, dtype=float)
    template = np.exp(-(t ** 2) / (2 * template_sigma ** 2))
    template = template / template.sum()

    match_score = np.zeros((band_h, band_w))
    for x in range(band_w):
        x_lo = max(0, x - x_avg)
        x_hi = min(band_w, x + x_avg + 1)
        col = gray_band[:, x_lo:x_hi].mean(axis=1).astype(float)
        col_inv = col.max() - col
        for y in range(band_h):
            y_lo = max(0, y - half_t)
            y_hi = min(band_h, y + half_t + 1)
            t_lo = y_lo - (y - half_t)
            t_hi = len(template) - ((y + half_t + 1) - y_hi)
            if t_hi > t_lo:
                match_score[y, x] = np.sum(col_inv[y_lo:y_hi] * template[t_lo:t_hi])

    trace = np.zeros(band_w, dtype=int)
    trace[0] = np.argmax(match_score[:, 0])

    for x in range(1, band_w):
        prev_y = trace[x - 1]
        scores = np.full(band_h, -np.inf)
        for y in range(band_h):
            scores[y] = match_score[y, x] - continuity_weight * abs(y - prev_y)
        trace[x] = np.argmax(scores)

    if reverse:
        trace = trace[::-1]
    return trace


def _build_ridge_map(band, meijering_sigmas=range(1, 4), continuity_window=25):
    """
    Build enhanced ridge map: Meijering filter + horizontal continuity weighting.
    Dark lines on bright background → high ridge score.
    """
    band_norm = (band - band.min()) / (band.max() - band.min() + 1e-6)
    band_inv = 1.0 - band_norm
    ridge_raw = meijering(band_inv, sigmas=meijering_sigmas, black_ridges=False)
    ridge_raw = ridge_raw / (ridge_raw.max() + 1e-6)

    # Horizontal continuity: boost sustained ridges over point noise
    ridge_cont = uniform_filter(ridge_raw, size=[1, continuity_window])
    r_norm = ridge_raw / (ridge_raw.std() + 1e-6)
    c_norm = ridge_cont / (ridge_cont.std() + 1e-6)
    combined = 0.4 * r_norm + 0.6 * c_norm
    combined = combined / (combined.max() + 1e-6)
    return combined


def _find_ridge_endpoints(ridge_map, seed_percentile=90, min_seed_size=20):
    """
    Find DNA trace endpoints from high-ridge seed regions.
    Uses top percentile of ridge map, removes small objects,
    then finds the two seed pixels with maximum Euclidean distance
    (orientation-agnostic: works for horizontal, vertical, or diagonal DNA).
    Returns (endpoint_a, endpoint_b) as (y, x) tuples, or None if failed.
    """
    from skimage.morphology import remove_small_objects
    from scipy.spatial import cKDTree

    nonzero = ridge_map[ridge_map > 0.01]
    if len(nonzero) == 0:
        return None

    thresh = np.percentile(nonzero, seed_percentile)
    seeds = ridge_map > thresh
    seeds = remove_small_objects(seeds, min_size=min_seed_size)

    seed_ys, seed_xs = np.where(seeds)
    if len(seed_ys) < 2:
        return None

    coords = np.column_stack([seed_ys, seed_xs])

    # Find the two farthest seed pixels (orientation-agnostic)
    # Step 1: pick one extreme via convex hull or simple max-distance search
    # Use BFS-like approach: pick arbitrary point, find farthest, then from that find farthest
    tree = cKDTree(coords)
    # Start from first point, find farthest
    dists_0 = np.sqrt(np.sum((coords - coords[0])**2, axis=1))
    idx_a = np.argmax(dists_0)
    # From that point, find farthest
    dists_a = np.sqrt(np.sum((coords - coords[idx_a])**2, axis=1))
    idx_b = np.argmax(dists_a)

    ep_a = (int(coords[idx_a, 0]), int(coords[idx_a, 1]))
    ep_b = (int(coords[idx_b, 0]), int(coords[idx_b, 1]))

    dist = np.sqrt((ep_a[0] - ep_b[0])**2 + (ep_a[1] - ep_b[1])**2)
    if dist < 10:
        return None

    return ep_a, ep_b


def ridge_trace(gray, y_lo=None, y_hi=None, x_lo=None, x_hi=None,
                meijering_sigmas=range(1, 4), smooth_sigma=2.0):
    """
    Ridge detection → MCP (Minimum Cost Path) trace.
    Handles DNA that doesn't span the full image and folds/kinks
    where the same x has multiple y values.

    Pipeline:
        1. Meijering filter + continuity weighting → ridge map
        2. Top-percentile seed regions → endpoint detection
        3. MCP (route_through_array) on inverse ridge cost surface
        4. Gaussian smoothing

    Args:
        gray: 2D grayscale image (float)
        y_lo, y_hi: Y search band (default: full image)
        x_lo, x_hi: X search range (default: full image)
        meijering_sigmas: σ range for Meijering filter (default: 1-3)
        smooth_sigma: σ for final trace smoothing (default: 2.0)

    Returns:
        cx, cy: arrays of trace coordinates (float), arc-length ordered.
                Duplicate x values are allowed.
    """
    from skimage.graph import route_through_array

    h_full, w_full = gray.shape
    if y_lo is None:
        y_lo = 0
    if y_hi is None:
        y_hi = h_full
    if x_lo is None:
        x_lo = 0
    if x_hi is None:
        x_hi = w_full

    band = gray[y_lo:y_hi, x_lo:x_hi]
    band_h, band_w = band.shape

    # Step 1: Enhanced ridge map (Meijering + continuity)
    ridge_map = _build_ridge_map(band, meijering_sigmas)

    # Step 2: Find endpoints from high-ridge seed regions
    endpoints = _find_ridge_endpoints(ridge_map)
    if endpoints is None:
        print("  Ridge trace: could not find endpoints, returning empty trace")
        return np.array([]), np.array([])

    ep_a, ep_b = endpoints
    print(f"  Endpoints: ({ep_a[1]+x_lo},{ep_a[0]+y_lo}) → "
          f"({ep_b[1]+x_lo},{ep_b[0]+y_lo})")

    # Step 3: MCP on inverse ridge cost surface
    cost_map = 1.0 / (ridge_map + 0.01)
    path, cost = route_through_array(cost_map, ep_a, ep_b, fully_connected=True)
    path = np.array(path)
    print(f"  MCP path: {len(path)} points, cost={cost:.1f}")

    cy_local = path[:, 0].astype(float)
    cx_local = path[:, 1].astype(float)

    # Step 4: Smooth the trace
    if smooth_sigma > 0 and len(cx_local) > 5:
        cx_smooth = gaussian_filter(cx_local, sigma=smooth_sigma)
        cy_smooth = gaussian_filter(cy_local, sigma=smooth_sigma)
    else:
        cx_smooth = cx_local
        cy_smooth = cy_local

    # Convert back to full image coordinates
    cx_out = cx_smooth + x_lo
    cy_out = cy_smooth + y_lo

    arc_len = np.sum(np.sqrt(np.diff(cx_out)**2 + np.diff(cy_out)**2))
    print(f"  Ridge trace: {len(cx_out)} points, arc length = {arc_len:.1f} px")

    return cx_out, cy_out


def auto_select_trace(gray, line_strength, y_lo, y_hi,
                      max_step=2, **kwargs):
    """
    Auto-select best trace by comparing CLAHE DP vs Meijering DP.
    Applies blob masking before tracing to avoid round noise artifacts.
    Returns (trace_local, method_name).
    """
    # Blob masking: remove round dark noise before tracing
    gray_clean, n_blobs = mask_round_blobs(gray)
    if n_blobs > 0:
        print(f"  Blob masking: removed {n_blobs} round blob(s)")
        # Recompute line_strength on clean image
        line_strength = compute_line_strength(gray_clean)

    band_clahe = line_strength[y_lo:y_hi, :]
    trace_clahe = dp_trace(band_clahe, max_step=max_step)

    ls_meij = compute_line_strength_meijering(gray_clean)
    band_meij = ls_meij[y_lo:y_hi, :]
    trace_meij = dp_trace(band_meij, max_step=max_step)

    def _eval(trace_local, bg_gap=8, bg_window=25):
        ty = trace_local + y_lo
        h_img = gray.shape[0]
        dna_sum, bg_sum, n = 0, 0, 0
        for x in range(0, len(trace_local), max(1, len(trace_local) // 100)):
            y = int(ty[x])
            y_lo_s = max(0, y - 1)
            y_hi_s = min(h_img, y + 2)
            dna_sum += gray[y_lo_s:y_hi_s, x].mean()
            ya_end = y - bg_gap
            ya_start = max(0, ya_end - bg_window)
            yb_start = y + bg_gap
            yb_end = min(h_img, yb_start + bg_window)
            bg_above = gray[ya_start:ya_end, x].mean() if ya_start < ya_end else np.nan
            bg_below = gray[yb_start:yb_end, x].mean() if yb_start < yb_end else np.nan
            if not np.isnan(bg_above) and not np.isnan(bg_below):
                bg_sum += (bg_above + bg_below) / 2
            elif not np.isnan(bg_above):
                bg_sum += bg_above
            elif not np.isnan(bg_below):
                bg_sum += bg_below
            else:
                continue
            n += 1
        return (bg_sum - dna_sum) / max(n, 1)

    score_clahe = _eval(trace_clahe)
    score_meij = _eval(trace_meij)
    print(f"  CLAHE DP score: {score_clahe:.2f}, Meijering DP score: {score_meij:.2f}")

    if score_meij >= score_clahe:
        print(f"  → Selected: meijering")
        return trace_meij, "meijering"
    else:
        print(f"  → Selected: clahe")
        return trace_clahe, "clahe"


def measure_intensity(gray, cx, cy, bg_gap=8, bg_extent=1.5, bg_side="both"):
    """
    Measure DNA backbone and background intensity from original image.
    - DNA: 3px vertical window centered on trace
    - Background above: bg_extent px (default 1.5) outward from gap edge above trace
    - Background below: bg_extent px (default 1.5) outward from gap edge below trace
    - bg_side: "both" (default), "above", or "below"
    """
    h = gray.shape[0]
    dna_vals = []
    bg_vals = []

    for i in range(len(cx)):
        x, y = int(cx[i]), int(cy[i])

        # DNA intensity (3px vertical window)
        y_lo = max(0, y - 1)
        y_hi = min(h, y + 2)
        dna_vals.append(gray[y_lo:y_hi, x].mean())

        # Background above: 1px outward from gap edge
        ya_end = y - bg_gap
        ya_start = max(0, ya_end - 1)
        bg_above = gray[ya_start:ya_end, x].mean() if ya_start < ya_end else np.nan

        # Background below: 2px outward from gap edge
        yb_start = y + bg_gap
        yb_end = min(h, yb_start + 2)
        bg_below = gray[yb_start:yb_end, x].mean() if yb_start < yb_end else np.nan

        # Always append 2 values per point (above, below) for consistent reshape
        if bg_side == "both":
            bg_vals.append(bg_above)
            bg_vals.append(bg_below)
        elif bg_side == "above":
            bg_vals.append(bg_above)
            bg_vals.append(bg_above)  # duplicate to keep shape
        elif bg_side == "below":
            bg_vals.append(bg_below)
            bg_vals.append(bg_below)  # duplicate to keep shape

    return np.array(dna_vals), np.array(bg_vals)


def create_output_image(img_rgb, cx, cy, dna_vals, bg_vals, px_size, y_offset=15):
    """Create output image: original + red trace (offset) + data table below."""
    h_orig, w_orig = img_rgb.shape[:2]
    text_area_h = 160
    canvas = np.ones((h_orig + text_area_h, w_orig, 3), dtype=np.uint8) * 40
    canvas[:h_orig, :, :] = img_rgb

    pil_out = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_out)

    # Red trace with y offset
    for i in range(len(cx) - 1):
        draw.line(
            [(int(cx[i]), int(cy[i] + y_offset)),
             (int(cx[i + 1]), int(cy[i + 1] + y_offset))],
            fill=(255, 0, 0), width=2
        )

    # Endpoints
    r = 6
    draw.ellipse(
        [int(cx[0]) - r, int(cy[0] + y_offset) - r,
         int(cx[0]) + r, int(cy[0] + y_offset) + r],
        fill=(255, 255, 0), outline=(255, 255, 255), width=2
    )
    draw.ellipse(
        [int(cx[-1]) - r, int(cy[-1] + y_offset) - r,
         int(cx[-1]) + r, int(cy[-1] + y_offset) + r],
        fill=(255, 255, 0), outline=(255, 255, 255), width=2
    )

    # Measurements
    total_length = sum(
        np.sqrt(1.0 + (cy[i + 1] - cy[i]) ** 2) for i in range(len(cx) - 1)
    )
    end2end = np.sqrt((float(cx[-1] - cx[0])) ** 2 + (cy[-1] - cy[0]) ** 2)

    # Fonts (use matplotlib's bundled DejaVu Sans for cross-platform compatibility)
    _font_path = os.path.join(
        os.path.dirname(matplotlib.__file__),
        "mpl-data", "fonts", "ttf", "DejaVuSans.ttf")
    _font_bold_path = os.path.join(
        os.path.dirname(matplotlib.__file__),
        "mpl-data", "fonts", "ttf", "DejaVuSans-Bold.ttf")
    try:
        font = ImageFont.truetype(_font_path, 14)
        font_bold = ImageFont.truetype(_font_bold_path, 15)
        font_data = ImageFont.truetype(_font_path, 13)
    except Exception:
        font = font_bold = font_data = ImageFont.load_default()

    # Top label (show in nm if px_size known, else px)
    if px_size:
        label = (f"Contour length: {total_length * px_size * 1000:.0f} nm   "
                 f"|   End-to-end: {end2end * px_size * 1000:.0f} nm")
    else:
        label = (f"Contour length: {total_length:.0f} px   "
                 f"|   End-to-end: {end2end:.0f} px")
    draw.text((10, 10), label, fill=(255, 255, 0), font=font)

    # Data table below image
    y_text = h_orig + 10
    col1, col2 = 30, 350

    draw.text((col1, y_text), "DNA Backbone Intensity", fill=(255, 200, 100), font=font_bold)
    draw.text((col2, y_text), "Background Intensity", fill=(200, 200, 255), font=font_bold)
    y_text += 24

    lines_left = [
        f"Mean:     {dna_vals.mean():.2f}",
        f"SD:       {dna_vals.std():.2f}",
        f"Min:      {dna_vals.min():.2f}",
        f"Max:      {dna_vals.max():.2f}",
        f"N pixels: {len(dna_vals)}",
    ]
    lines_right = [
        f"Mean:     {bg_vals.mean():.2f}",
        f"SD:       {bg_vals.std():.2f}",
        f"Min:      {bg_vals.min():.2f}",
        f"Max:      {bg_vals.max():.2f}",
        f"N pixels: {len(bg_vals)}",
    ]

    for i, (ll, lr) in enumerate(zip(lines_left, lines_right)):
        draw.text((col1, y_text + i * 20), ll, fill=(255, 255, 255), font=font_data)
        draw.text((col2, y_text + i * 20), lr, fill=(255, 255, 255), font=font_data)

    y_bottom = y_text + len(lines_left) * 20 + 5
    diff = dna_vals.mean() - bg_vals.mean()
    label = "darker" if diff < 0 else "brighter"
    draw.text(
        (col1, y_bottom),
        f"DNA - Background = {diff:.2f}  (DNA is {label})",
        fill=(255, 100, 100), font=font_data
    )

    return pil_out, total_length, end2end


def create_intensity_profile(dna_vals, bg_vals, cx, px_size=None, output_path="profile.png"):
    """
    Create intensity profile plot along the DNA backbone.
    - DNA intensity per position (with smoothed line)
    - Background mean ± SD band
    """
    # Cumulative contour distance for x-axis
    positions = np.arange(len(dna_vals)).astype(float)
    if px_size:
        positions = positions * px_size * 1000  # nm
        x_label = "Position along backbone (nm)"
    else:
        x_label = "Position along backbone (px)"

    # Background: reshape to (N, 2) since we sampled above+below per point
    bg_per_point = bg_vals.reshape(-1, 2).mean(axis=1)

    # Smoothed DNA intensity
    window = max(5, len(dna_vals) // 50)
    dna_smooth = uniform_filter1d(dna_vals, size=window)
    bg_smooth = uniform_filter1d(bg_per_point, size=window)

    fig, ax = plt.subplots(figsize=(10, 4))

    # DNA raw + smoothed
    ax.plot(positions, dna_vals, color='#FF6B6B', alpha=0.25, linewidth=0.5, label='DNA (raw)')
    ax.plot(positions, dna_smooth, color='#CC0000', linewidth=1.5, label='DNA (smoothed)')

    # Background smoothed + SD band
    bg_std_smooth = uniform_filter1d(
        np.abs(bg_per_point - bg_smooth), size=window
    ) * np.sqrt(np.pi / 2)  # approximate rolling SD
    ax.plot(positions, bg_smooth, color='#4488CC', linewidth=1.5, label='Background (smoothed)')
    ax.fill_between(positions, bg_smooth - bg_std_smooth, bg_smooth + bg_std_smooth,
                     color='#4488CC', alpha=0.15, label='Background ± SD')

    # Mean lines
    ax.axhline(dna_vals.mean(), color='#CC0000', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.axhline(bg_vals.mean(), color='#4488CC', linestyle='--', linewidth=0.8, alpha=0.6)

    # Annotations
    ax.text(positions[-1] * 0.98, dna_vals.mean() + 1, f'DNA mean: {dna_vals.mean():.1f}',
            ha='right', va='bottom', fontsize=8, color='#CC0000')
    ax.text(positions[-1] * 0.98, bg_vals.mean() + 1, f'BG mean: {bg_vals.mean():.1f}',
            ha='right', va='bottom', fontsize=8, color='#4488CC')

    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel("Intensity (8-bit)", fontsize=11)
    ax.set_title("Intensity Profile along DNA Backbone", fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.set_xlim(positions[0], positions[-1])
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return output_path


def create_separate_profiles(dna_vals, bg_vals, cx, px_size=None, base_path=""):
    """
    Create separate intensity profile plots for DNA backbone and background.
    """
    positions = np.arange(len(dna_vals)).astype(float)
    if px_size:
        positions = positions * px_size * 1000  # nm
        x_label = "Position along backbone (nm)"
    else:
        x_label = "Position along backbone (px)"

    bg_per_point = bg_vals.reshape(-1, 2).mean(axis=1)
    bg_above = bg_vals.reshape(-1, 2)[:, 0]
    bg_below = bg_vals.reshape(-1, 2)[:, 1]
    window = max(5, len(dna_vals) // 50)
    dna_smooth = uniform_filter1d(dna_vals, size=window)
    bg_smooth = uniform_filter1d(bg_per_point, size=window)
    bg_above_smooth = uniform_filter1d(bg_above, size=window)
    bg_below_smooth = uniform_filter1d(bg_below, size=window)

    # --- DNA backbone intensity profile ---
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(positions, dna_vals, color='#FF6B6B', alpha=0.25, linewidth=0.5, label='DNA (raw)')
    ax.plot(positions, dna_smooth, color='#CC0000', linewidth=1.5, label='DNA (smoothed)')
    ax.axhline(dna_vals.mean(), color='#CC0000', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.text(positions[-1] * 0.98, dna_vals.mean() + 1, f'Mean: {dna_vals.mean():.1f}',
            ha='right', va='bottom', fontsize=9, color='#CC0000')
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel("Intensity (8-bit)", fontsize=11)
    ax.set_title("DNA Backbone Intensity Profile", fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.set_xlim(positions[0], positions[-1])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    dna_profile_path = f"{base_path}_dna_profile.png"
    fig.savefig(dna_profile_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    # --- Background intensity profile ---
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(positions, bg_above, color='#88BBEE', alpha=0.2, linewidth=0.5, label='BG above (raw)')
    ax.plot(positions, bg_below, color='#AADDAA', alpha=0.2, linewidth=0.5, label='BG below (raw)')
    ax.plot(positions, bg_above_smooth, color='#2266AA', linewidth=1.2, label='BG above (smoothed)')
    ax.plot(positions, bg_below_smooth, color='#228844', linewidth=1.2, label='BG below (smoothed)')
    ax.plot(positions, bg_smooth, color='#4488CC', linewidth=1.5, linestyle='--', label='BG mean (smoothed)')
    bg_std_smooth = uniform_filter1d(
        np.abs(bg_per_point - bg_smooth), size=window
    ) * np.sqrt(np.pi / 2)
    ax.fill_between(positions, bg_smooth - bg_std_smooth, bg_smooth + bg_std_smooth,
                     color='#4488CC', alpha=0.12, label='BG mean ± SD')
    ax.axhline(bg_per_point.mean(), color='#4488CC', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.text(positions[-1] * 0.98, bg_per_point.mean() + 1, f'Mean: {bg_per_point.mean():.1f}',
            ha='right', va='bottom', fontsize=9, color='#4488CC')
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel("Intensity (8-bit)", fontsize=11)
    ax.set_title("Background Intensity Profile", fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.set_xlim(positions[0], positions[-1])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    bg_profile_path = f"{base_path}_bg_profile.png"
    fig.savefig(bg_profile_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    return dna_profile_path, bg_profile_path


def trim_edge_artifacts(dna_vals, bg_vals, cx, cy, window=30, threshold_sigma=1.5):
    """
    Trim edge regions where intensity rises due to image boundary effects.
    Uses background intensity: finds inward from each edge until the smoothed
    background drops below (median + threshold_sigma * MAD).
    Returns trimmed arrays.
    """
    n = len(dna_vals)
    if n < window * 4:
        return dna_vals, bg_vals, cx, cy

    bg_per_point = bg_vals.reshape(-1, 2).mean(axis=1)
    bg_smooth = uniform_filter1d(bg_per_point, size=window)

    # Use central 50% as reference for stable background level
    q1, q3 = n // 4, 3 * n // 4
    center_median = np.median(bg_smooth[q1:q3])
    center_mad = np.median(np.abs(bg_smooth[q1:q3] - center_median))
    cutoff = center_median + threshold_sigma * max(center_mad, 0.5)

    # Scan from left edge inward
    trim_left = 0
    for i in range(n // 4):
        if bg_smooth[i] > cutoff:
            trim_left = i + 1
        else:
            break

    # Scan from right edge inward
    trim_right = n
    for i in range(n - 1, 3 * n // 4, -1):
        if bg_smooth[i] > cutoff:
            trim_right = i
        else:
            break

    if trim_left >= trim_right:
        return dna_vals, bg_vals, cx, cy

    # Reshape bg_vals for trimming (N, 2)
    bg_2d = bg_vals.reshape(-1, 2)
    dna_trimmed = dna_vals[trim_left:trim_right]
    bg_trimmed = bg_2d[trim_left:trim_right].flatten()
    cx_trimmed = cx[trim_left:trim_right]
    cy_trimmed = cy[trim_left:trim_right]

    trimmed_total = (trim_left) + (n - trim_right)
    if trimmed_total > 0:
        print(f"Edge trimming: removed {trim_left} px from left, {n - trim_right} px from right "
              f"({trimmed_total} total, BG cutoff={cutoff:.1f})")

    return dna_trimmed, bg_trimmed, cx_trimmed, cy_trimmed


def export_intensity_excel(dna_vals, bg_vals, cx, cy, px_size=None, output_path="intensity.xlsx"):
    """
    Export DNA backbone and background intensity profiles to Excel.
    Sheet 1: DNA backbone intensity (raw + smoothed)
    Sheet 2: Background intensity (raw + smoothed)
    Sheet 3: Summary statistics
    """
    wb = openpyxl.Workbook()

    positions_px = np.arange(len(dna_vals)).astype(float)
    bg_per_point = bg_vals.reshape(-1, 2).mean(axis=1)
    bg_above = bg_vals.reshape(-1, 2)[:, 0]
    bg_below = bg_vals.reshape(-1, 2)[:, 1]

    if px_size:
        positions_nm = positions_px * px_size * 1000
    else:
        positions_nm = [None] * len(positions_px)

    # Compute smoothed values (same as profile plots)
    window = max(5, len(dna_vals) // 50)
    dna_smooth = uniform_filter1d(dna_vals, size=window)
    bg_smooth = uniform_filter1d(bg_per_point, size=window)
    bg_above_smooth = uniform_filter1d(bg_above, size=window)
    bg_below_smooth = uniform_filter1d(bg_below, size=window)

    # --- Sheet 1: DNA backbone ---
    ws_dna = wb.active
    ws_dna.title = "DNA_Backbone"
    headers_dna = ["Index", "X (px)", "Y (px)", "Position (px)",
                   "Intensity", "Intensity (smoothed)"]
    if px_size:
        headers_dna.append("Position (nm)")
    ws_dna.append(headers_dna)
    for i in range(len(dna_vals)):
        row = [i, int(cx[i]), int(cy[i]), positions_px[i],
               round(dna_vals[i], 2), round(float(dna_smooth[i]), 2)]
        if px_size:
            row.append(round(positions_nm[i], 2))
        ws_dna.append(row)

    # --- Sheet 2: Background ---
    ws_bg = wb.create_sheet("Background")
    headers_bg = ["Index", "X (px)", "Position (px)",
                  "BG_Above", "BG_Above (smoothed)",
                  "BG_Below", "BG_Below (smoothed)",
                  "BG_Mean", "BG_Mean (smoothed)"]
    if px_size:
        headers_bg.append("Position (nm)")
    ws_bg.append(headers_bg)
    for i in range(len(bg_per_point)):
        row = [i, int(cx[i]), positions_px[i],
               round(bg_above[i], 2), round(float(bg_above_smooth[i]), 2),
               round(bg_below[i], 2), round(float(bg_below_smooth[i]), 2),
               round(bg_per_point[i], 2), round(float(bg_smooth[i]), 2)]
        if px_size:
            row.append(round(positions_nm[i], 2))
        ws_bg.append(row)

    # --- Sheet 3: Summary ---
    ws_sum = wb.create_sheet("Summary")
    ws_sum.append(["Metric", "DNA Backbone", "Background"])
    ws_sum.append(["Mean", round(dna_vals.mean(), 2), round(bg_per_point.mean(), 2)])
    ws_sum.append(["SD", round(dna_vals.std(), 2), round(bg_per_point.std(), 2)])
    ws_sum.append(["Min", round(dna_vals.min(), 2), round(bg_per_point.min(), 2)])
    ws_sum.append(["Max", round(dna_vals.max(), 2), round(bg_per_point.max(), 2)])
    ws_sum.append(["N pixels", len(dna_vals), len(bg_per_point)])
    ws_sum.append(["DNA - BG", round(dna_vals.mean() - bg_per_point.mean(), 2), ""])
    if px_size:
        ws_sum.append([])
        ws_sum.append(["Pixel scale (nm/px)", round(px_size * 1000, 2)])

    wb.save(output_path)
    return output_path


ENDPOINTS_FILE = "endpoints.json"
PXUM_CONFIG_FILE = "pxum_config.json"


def _project_root():
    """Project root containing pxum_config.json and endpoints.json."""
    return os.path.dirname(os.path.abspath(__file__))


def _pxum_path(base_dir=None):
    if base_dir is None:
        base_dir = _project_root()
    return os.path.join(base_dir, PXUM_CONFIG_FILE)


def _load_pxum_raw(base_dir=None):
    """Load raw {key: px/um} dict from disk. Creates an empty file if absent."""
    import json
    path = _pxum_path(base_dir)
    if not os.path.exists(path):
        with open(path, 'w') as f:
            json.dump({"_comment": "px per um (pixels per micrometer)"}, f, indent=2)
        return {}
    with open(path) as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items()
            if not str(k).startswith('_') and k != 'default'}


def _save_pxum_entry(image_name, pxum_value, base_dir=None):
    """Append/update one image's px/um entry to disk."""
    import json
    path = _pxum_path(base_dir)
    if os.path.exists(path):
        with open(path) as f:
            raw = json.load(f)
    else:
        raw = {"_comment": "px per um (pixels per micrometer)"}
    raw[image_name] = float(pxum_value)
    with open(path, 'w') as f:
        json.dump(raw, f, indent=2)


def load_pxum_config(base_dir=None):
    """Load px/um config and return {image_name: nm_per_px} (auto-converted).

    User stores px/µm (easier to read off SEM software); internal code uses
    nm/px so this loader does the conversion (nm/px = 1000 / px/µm).
    """
    raw = _load_pxum_raw(base_dir)
    out = {}
    for k, v in raw.items():
        if not isinstance(v, (int, float)) or v <= 0:
            raise SystemExit(
                f"Error: invalid px/µm value for '{k}' in pxum_config.json: {v}")
        out[k] = 1000.0 / float(v)
    return out


def prompt_pxum_for(image_name, base_dir=None):
    """Interactively ask the user for an image's px/µm and persist it.

    Returns the entered px/µm value (float). Re-prompts on bad input.
    """
    print(f"\n  No px/µm entry for image '{image_name}' in pxum_config.json.")
    while True:
        try:
            raw = input(f"  Enter px/µm for '{image_name}': ").strip()
        except EOFError:
            raise SystemExit(
                f"Error: stdin closed; cannot prompt for '{image_name}'. "
                f"Add the entry to pxum_config.json manually.")
        try:
            val = float(raw)
        except ValueError:
            print("  → not a number, try again")
            continue
        if val <= 0:
            print("  → must be positive, try again")
            continue
        _save_pxum_entry(image_name, val, base_dir)
        print(f"  → saved \"{image_name}\": {val} to pxum_config.json")
        return val


def ensure_pxum_entries(image_names, base_dir=None, interactive=True,
                        default_pxum=None):
    """Make sure every image in `image_names` has a px/µm entry.

    Resolution order for missing images:
      1. If `default_pxum` is given, use that value for all missing images
         (no prompts). Each gets saved to pxum_config.json.
      2. Else if `interactive=True`, prompt for each missing image and save.
      3. Else raise SystemExit.

    Returns {image_name: nm_per_px} dict covering all requested images.
    """
    raw = _load_pxum_raw(base_dir)
    missing = [n for n in image_names if n not in raw]
    if missing:
        if default_pxum is not None:
            if not isinstance(default_pxum, (int, float)) or default_pxum <= 0:
                raise SystemExit(
                    f"Error: invalid default px/µm value: {default_pxum}")
            print(f"\npxum_config.json is missing {len(missing)} image(s); "
                  f"auto-applying CLI value {default_pxum} px/µm to each:")
            for nm in missing:
                _save_pxum_entry(nm, default_pxum, base_dir)
                raw[nm] = default_pxum
                print(f"  → saved \"{nm}\": {default_pxum}")
        elif not interactive:
            raise SystemExit(
                f"Error: pxum_config.json missing entries for: {missing}")
        else:
            print(f"\npxum_config.json is missing {len(missing)} image(s); "
                  f"please provide px/µm for each:")
            for nm in missing:
                val = prompt_pxum_for(nm, base_dir)
                raw[nm] = val
    return {n: 1000.0 / float(raw[n]) for n in image_names}


def get_pxnm(image_name, base_dir=None, interactive=True, default_pxum=None):
    """Look up nm/px for image; auto-save default or prompt if missing."""
    raw = _load_pxum_raw(base_dir)
    if image_name in raw:
        return 1000.0 / float(raw[image_name])
    if default_pxum is not None:
        if not isinstance(default_pxum, (int, float)) or default_pxum <= 0:
            raise SystemExit(
                f"Error: invalid default px/µm value: {default_pxum}")
        _save_pxum_entry(image_name, default_pxum, base_dir)
        print(f"  Saved \"{image_name}\": {default_pxum} px/µm "
              f"to pxum_config.json (from CLI)")
        return 1000.0 / float(default_pxum)
    if not interactive:
        raise SystemExit(
            f"Error: no entry for image '{image_name}' in pxum_config.json")
    val = prompt_pxum_for(image_name, base_dir)
    return 1000.0 / val


def _endpoints_path(base_dir=None):
    """Return the path to the endpoints cache file."""
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, ENDPOINTS_FILE)


def load_endpoints(base_dir=None):
    """Load saved endpoints from JSON. Returns dict of {image_name: {start, end}}."""
    import json
    path = _endpoints_path(base_dir)
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}


def save_endpoint(image_name, start_pt, end_pt, base_dir=None):
    """Save a single image's endpoints to JSON cache."""
    import json
    path = _endpoints_path(base_dir)
    data = load_endpoints(base_dir)
    data[image_name] = {"start": list(start_pt), "end": list(end_pt)}
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def get_endpoints(gray, image_name, interactive=True, base_dir=None, title=None):
    """Get endpoints for an image: load from cache, or ask interactively and save.

    Returns (start_pt, end_pt) or (None, None) if cancelled.
    """
    # Try loading from cache first
    cached = load_endpoints(base_dir)
    if image_name in cached:
        ep = cached[image_name]
        start_pt, end_pt = ep["start"], ep["end"]
        print(f"  Loaded cached endpoints for {image_name}: "
              f"start=({start_pt[0]},{start_pt[1]}), end=({end_pt[0]},{end_pt[1]})")
        return start_pt, end_pt

    # Interactive selection
    if not interactive:
        return None, None

    display_title = title or image_name
    start_pt, end_pt = pick_endpoints_interactive(gray, title=display_title)

    # Save if selected
    if start_pt is not None and end_pt is not None:
        save_endpoint(image_name, start_pt, end_pt, base_dir)
        print(f"  Saved endpoints for {image_name}")

    return start_pt, end_pt


def get_or_pick_endpoints(gray, image_name, interactive=True,
                          title=None, base_dir=None):
    """Return (start_pt, end_pt) for image_name, using endpoints.json cache.

    Behavior:
      - If cached: return cached values immediately.
      - Else if interactive=True: show GUI for 2-point click.
          * User clicks 2 points  -> save those points, return them.
          * User closes window    -> save full-range fallback
                                     ([0,h//2] -> [w-1,h//2]), return it.
      - Else (non-interactive):   return full-range fallback without saving.
    """
    cached = load_endpoints(base_dir)
    if image_name in cached:
        ep = cached[image_name]
        start_pt, end_pt = ep['start'], ep['end']
        print(f"  Loaded cached endpoints for {image_name}: "
              f"({start_pt[0]},{start_pt[1]}) -> ({end_pt[0]},{end_pt[1]})")
        return start_pt, end_pt

    h, w = gray.shape
    full_start = [0, h // 2]
    full_end = [w - 1, h // 2]

    if not interactive:
        print(f"  No cached endpoints for {image_name}, using full range "
              f"(non-interactive)")
        return full_start, full_end

    start_pt, end_pt = pick_endpoints_interactive(gray, title or image_name)
    if start_pt is None:
        save_endpoint(image_name, full_start, full_end, base_dir)
        print(f"  Skipped (no click): saved full-range endpoints "
              f"[0,{h//2}] -> [{w-1},{h//2}] to endpoints.json")
        return full_start, full_end

    save_endpoint(image_name, start_pt, end_pt, base_dir)
    print(f"  Saved endpoints to endpoints.json: "
          f"({start_pt[0]},{start_pt[1]}) -> ({end_pt[0]},{end_pt[1]})")
    return start_pt, end_pt


def pick_endpoints_interactive(gray, title=""):
    """
    Show the image and let user click two endpoints (start, end) of the DNA.
    Returns (start_pt, end_pt) as [x, y] lists, or (None, None) if cancelled.
    """
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    pts = []

    fig, ax = plt.subplots(figsize=(14, max(4, 14 * gray.shape[0] / gray.shape[1])))
    ax.imshow(gray, cmap='gray')
    ax.set_title(f"{title}\nClick 2 points: DNA start and end. Close window to cancel.")
    ax.set_xlabel("Click start point, then end point")
    markers = []

    def onclick(event):
        if event.inaxes != ax:
            return
        x, y = int(round(event.xdata)), int(round(event.ydata))
        pts.append([x, y])
        color = 'lime' if len(pts) == 1 else 'red'
        label = 'Start' if len(pts) == 1 else 'End'
        m = ax.plot(x, y, 'o', color=color, markersize=8, markeredgecolor='white',
                    markeredgewidth=1.5)
        ax.annotate(f' {label} ({x},{y})', (x, y), color=color, fontsize=10,
                    fontweight='bold')
        markers.append(m)
        fig.canvas.draw()
        if len(pts) >= 2:
            plt.close(fig)

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.tight_layout()
    plt.show()

    # Restore Agg backend for subsequent non-interactive plots
    matplotlib.use('Agg')

    if len(pts) >= 2:
        print(f"Selected endpoints: start=({pts[0][0]},{pts[0][1]}), "
              f"end=({pts[1][0]},{pts[1][1]})")
        return pts[0], pts[1]
    else:
        print("Endpoint selection cancelled (less than 2 points clicked)")
        return None, None


def process_single_image(image_path, args, start_pt=None, end_pt=None,
                         nm_per_px=None):
    """
    Run the full backbone trace pipeline on a single image.

    Args:
        image_path: path to .tif file
        args: argparse namespace with trace parameters
        start_pt: [x, y] start point override (from interactive selection)
        end_pt: [x, y] end point override (from interactive selection)
    """
    img = Image.open(image_path)
    img_rgb = np.array(img.convert('RGB'))
    gray = np.array(img.convert('L')).astype(float)
    h_full, w_full = gray.shape
    print(f"Image: {image_path} ({w_full} x {h_full})")

    # Use overrides if provided, else fall back to args
    spt = start_pt if start_pt is not None else getattr(args, 'start_pt', None)
    ept = end_pt if end_pt is not None else getattr(args, 'end_pt', None)

    sem = gray
    sem_h = h_full

    # Pixel scale (looked up from pxum_config.json by main; falls back to
    # legacy --hfw if explicitly given on the CLI for backward compatibility).
    if nm_per_px is None:
        hfw = getattr(args, 'hfw', None)
        if hfw:
            nm_per_px = hfw * 1000.0 / w_full  # µm * 1000 / px = nm/px
    if nm_per_px is not None:
        px_size = nm_per_px / 1000.0  # µm/pixel
        print(f"Pixel scale: {nm_per_px:.2f} nm/pixel")
    else:
        px_size = None
        print("Pixel scale: not provided. Reporting in pixels only.")

    # Compute line strength map
    print("Computing line strength map...")
    line_strength = compute_line_strength(sem)

    # If start/end points provided, derive y_band and x_range from them
    if spt is not None and ept is not None:
        sx, sy = spt
        ex, ey = ept
        if sx > ex:
            sx, sy, ex, ey = ex, ey, sx, sy
        y_span = abs(sy - ey)
        y_margin = max(y_span, 30)
        y_lo = max(0, min(sy, ey) - y_margin)
        y_hi = min(sem_h, max(sy, ey) + y_margin)
        x_start, x_end = sx, ex
        print(f"Start/End points: ({sx},{sy}) -> ({ex},{ey})")
        print(f"Y search band (from endpoints): [{y_lo}, {y_hi}]")
        print(f"X range (from endpoints): [{x_start}, {x_end}]")
    else:
        y_band = getattr(args, 'y_band', None)
        if y_band is not None:
            y_lo, y_hi = y_band
        else:
            y_lo, y_hi = 0, sem_h - 1
        print(f"Y search band: [{y_lo}, {y_hi}]")

    # Trace
    trace_method = getattr(args, 'trace_method', 'auto')
    max_step = getattr(args, 'max_step', 2)
    template_sigma = getattr(args, 'template_sigma', 1.5)
    continuity_weight = getattr(args, 'continuity_weight', 1.0)
    x_avg = getattr(args, 'x_avg', 1)

    if trace_method == "auto":
        print("Auto-selecting trace method...")
        trace_local, _ = auto_select_trace(
            gray, line_strength, y_lo, y_hi,
            max_step=max_step, template_sigma=template_sigma,
            continuity_weight=continuity_weight, x_avg=x_avg)
    elif trace_method == "intensity":
        print("Running intensity trace...")
        gray_band = gray[y_lo:y_hi, :]
        trace_local = dp_trace_intensity(gray_band,
                                          template_sigma=template_sigma,
                                          continuity_weight=continuity_weight,
                                          x_avg=x_avg)
    else:
        print("Running DP trace...")
        band = line_strength[y_lo:y_hi, :]
        trace_local = dp_trace(band, max_step=max_step)
    trace_y_full = trace_local + y_lo

    if spt is None or ept is None:
        x_range = getattr(args, 'x_range', None)
        if x_range is not None:
            x_start, x_end = x_range
        else:
            x_start, x_end = 0, w_full - 1
    print(f"X range: [{x_start}, {x_end}]")

    cx = np.arange(x_start, x_end + 1).astype(float)
    cy = trace_y_full[x_start:x_end + 1].astype(float)

    # Measure intensity from ORIGINAL image
    bg_side = getattr(args, 'bg_side', 'both')
    print("Measuring intensity from original image...")
    dna_vals, bg_vals = measure_intensity(gray, cx, cy, bg_side=bg_side)

    # Create output image (px_size = µm/pixel; pass None if unknown so the
    # label correctly falls back to pixel units instead of bogus nm)
    y_offset = getattr(args, 'y_offset', 15)
    pil_out, total_length, end2end = create_output_image(
        img_rgb, cx, cy, dna_vals, bg_vals,
        px_size,
        y_offset=y_offset
    )

    # Save outputs
    base = os.path.splitext(image_path)[0]
    output = getattr(args, 'output', None)
    if output:
        out_path = output
    else:
        out_path = f"{base}_backbone.png"
    pil_out.save(out_path)
    print(f"Saved: {out_path}")

    profile_path = f"{base}_profile.png"
    create_intensity_profile(dna_vals, bg_vals, cx, px_size, output_path=profile_path)
    print(f"Saved: {profile_path}")

    dna_prof, bg_prof = create_separate_profiles(dna_vals, bg_vals, cx, px_size, base_path=base)
    print(f"Saved: {dna_prof}")
    print(f"Saved: {bg_prof}")

    excel_path = f"{base}_intensity.xlsx"
    export_intensity_excel(dna_vals, bg_vals, cx, cy, px_size, output_path=excel_path)
    print(f"Saved: {excel_path}")

    # Print results
    print(f"\n{'='*50}")
    print(f"  DNA Backbone Analysis Results")
    print(f"{'='*50}")
    if px_size:
        print(f"  Contour length : {total_length:.1f} px = {total_length*px_size*1000:.1f} nm ({total_length*px_size:.3f} µm)")
        print(f"  End-to-end     : {end2end:.1f} px = {end2end*px_size*1000:.1f} nm ({end2end*px_size:.3f} µm)")
    else:
        print(f"  Contour length : {total_length:.1f} px")
        print(f"  End-to-end     : {end2end:.1f} px")
    print(f"  N trace pixels : {len(cx)}")
    print()
    print(f"  DNA Backbone Intensity (original image)")
    print(f"    Mean : {dna_vals.mean():.2f}")
    print(f"    SD   : {dna_vals.std():.2f}")
    print(f"    Min  : {dna_vals.min():.2f}")
    print(f"    Max  : {dna_vals.max():.2f}")
    print(f"    N    : {len(dna_vals)}")
    print()
    print(f"  Background Intensity")
    print(f"    Mean : {bg_vals.mean():.2f}")
    print(f"    SD   : {bg_vals.std():.2f}")
    print(f"    Min  : {bg_vals.min():.2f}")
    print(f"    Max  : {bg_vals.max():.2f}")
    print(f"    N    : {len(bg_vals)}")
    print()
    print(f"  DNA - Background = {dna_vals.mean() - bg_vals.mean():.2f}")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description="DNA Backbone Tracer for SEM/TEM images")
    parser.add_argument("image", help="Input image path (.tif) or folder containing .tif files")
    parser.add_argument("pxum", type=float, nargs="?", default=None,
                        help="Default px/µm to apply to any image missing "
                             "from pxum_config.json (auto-saved). If omitted, "
                             "the script prompts interactively.")
    parser.add_argument("--y_band", type=int, nargs=2, default=None,
                        help="Y search band [lo hi] for DNA (auto-detect if omitted)")
    parser.add_argument("--x_range", type=int, nargs=2, default=None,
                        help="X range [start end] for DNA extent (auto-detect if omitted)")
    parser.add_argument("--hfw", type=float, default=None,
                        help="Horizontal Field Width in µm (auto-read from filename if possible)")
    parser.add_argument("--trace_method", type=str, default="auto",
                        choices=["auto", "dp", "intensity"],
                        help="Trace method: auto (select best), dp (line strength), "
                             "or intensity (Gaussian matching)")
    parser.add_argument("--max_step", type=int, default=2,
                        help="Max vertical step per pixel in DP (default: 2)")
    parser.add_argument("--template_sigma", type=float, default=1.5,
                        help="Gaussian template σ for intensity method (default: 1.5)")
    parser.add_argument("--continuity_weight", type=float, default=1.0,
                        help="Continuity weight for intensity method (default: 1.0)")
    parser.add_argument("--x_avg", type=int, default=1,
                        help="X averaging half-width for intensity method (default: 1)")
    parser.add_argument("--y_offset", type=int, default=15,
                        help="Y offset for trace display (default: 15)")
    parser.add_argument("--bg_side", type=str, default="both",
                        choices=["both", "above", "below"],
                        help="Which side to measure background: both (default), above, or below")
    parser.add_argument("--start_pt", type=int, nargs=2, default=None,
                        help="DNA start point [x y] (for low-mag images where DNA ends are visible)")
    parser.add_argument("--end_pt", type=int, nargs=2, default=None,
                        help="DNA end point [x y] (for low-mag images where DNA ends are visible)")
    parser.add_argument("--no-interactive", dest="interactive",
                        action="store_false", default=True,
                        help="Disable interactive DNA start/end point selection")
    parser.add_argument("--output", type=str, default=None,
                        help="Output filename (default: <input>_backbone.png)")
    args = parser.parse_args()

    import glob

    # Check if input is a directory
    if os.path.isdir(args.image):
        tif_files = sorted(glob.glob(os.path.join(args.image, "*.tif")))
        if not tif_files:
            print(f"No .tif files found in {args.image}")
            return
        print(f"Found {len(tif_files)} .tif files in {args.image}\n")

        # Ensure every image has a px/µm entry; prompts for missing ones,
        # or uses CLI default_pxum if provided. Returns {name: nm/px}.
        names = [os.path.splitext(os.path.basename(t))[0] for t in tif_files]
        nmpx_map = ensure_pxum_entries(
            names, interactive=args.interactive, default_pxum=args.pxum)

        for i, tif_path in enumerate(tif_files):
            name = os.path.basename(tif_path)
            stem = os.path.splitext(name)[0]
            print(f"\n{'='*60}")
            print(f"  [{i+1}/{len(tif_files)}] {name}")
            print(f"{'='*60}")

            start_pt, end_pt = None, None
            if args.interactive:
                img_gray = np.array(Image.open(tif_path).convert('L')).astype(float)
                start_pt, end_pt = pick_endpoints_interactive(
                    img_gray, f"[{i+1}/{len(tif_files)}] {name}")
                if start_pt is None:
                    h, w = img_gray.shape
                    start_pt = [0, h // 2]
                    end_pt = [w - 1, h // 2]
                    save_endpoint(name, start_pt, end_pt)
                    print(f"Skipped (no click): saved full-range endpoints "
                          f"[0,{h//2}] -> [{w-1},{h//2}] to endpoints.json")
                    continue

            process_single_image(tif_path, args,
                                 start_pt=start_pt, end_pt=end_pt,
                                 nm_per_px=nmpx_map[stem])

        print(f"\n{'='*60}")
        print(f"  All done: {len(tif_files)} images processed")
        print(f"{'='*60}")

    else:
        # Single image
        single_name = os.path.splitext(os.path.basename(args.image))[0]
        nmpx_map = ensure_pxum_entries(
            [single_name], interactive=args.interactive,
            default_pxum=args.pxum)
        start_pt, end_pt = None, None
        if args.interactive:
            img_gray = np.array(Image.open(args.image).convert('L')).astype(float)
            start_pt, end_pt = pick_endpoints_interactive(img_gray, args.image)
            if start_pt is None:
                h, w = img_gray.shape
                start_pt = [0, h // 2]
                end_pt = [w - 1, h // 2]
                save_endpoint(os.path.basename(args.image), start_pt, end_pt)
                print(f"Skipped (no click): saved full-range endpoints "
                      f"[0,{h//2}] -> [{w-1},{h//2}] to endpoints.json")
                return

        process_single_image(args.image, args,
                             start_pt=start_pt, end_pt=end_pt,
                             nm_per_px=nmpx_map[single_name])


if __name__ == "__main__":
    main()
