# DNA-Protein Binding Detection from SEM/TEM Images

Automated pipeline for tracing DNA backbones, measuring edge widths, and detecting protein-bound regions from scanning/transmission electron microscopy images.

## Pipeline Overview

```
[SEM/TEM .tif image]
       |
       v
(1) dna_backbone_trace.py    -- Trace DNA backbone via DP + auto method selection
       |
       v
(2) dna_edge_width.py        -- Measure edge-to-edge width (gradient extrema)
       |
       v
(3) dna_profile_heatmap.py   -- Generate perpendicular profile heatmaps for labeling
       |                         (user labels protein-bound rows yellow in Excel)
       v
(4) ml_profile_classifier.py -- Train classifier (CalibratedRF) on labeled profiles
       |
       v
(5) predict.py               -- Predict binding on new images + overlay visualization
```

## Requirements

Python 3.8+ with the following packages:

```bash
pip install -r requirements.txt
```

## Setup

1. **Scale configuration**: Copy `pxum_config.example.json` to `pxum_config.json` and add your image names with their px/um (pixels per micrometer) values. Every image must have an explicit entry.

2. **Endpoint selection**: On first run, each script will show the image and prompt you to click the DNA start and end points. These are cached in `endpoints.json` for subsequent runs. Close the window without clicking to use the full image range.

## Usage

### Step 1: Trace DNA backbone
```bash
python dna_backbone_trace.py image.tif           # single image
python dna_backbone_trace.py tif/ 288             # folder + default px/um
```
Outputs: `*_backbone.png`, `*_profile.png`, `*_intensity.xlsx`

### Step 2: Measure edge width
```bash
python dna_edge_width.py image.tif
```
Outputs: `*_edge_width_profile.png`, `*_edge_width_overlay.png`, `*_edge_width_data.xlsx`

### Step 3: Generate profile heatmaps for labeling
```bash
python dna_profile_heatmap.py tif/
```
Outputs: `*_profile_heatmap.xlsx`, `*_profile_heatmap.png`

Open each `*_profile_heatmap.xlsx` and highlight protein-bound rows in yellow (`#FFFF00`). These labels are used as ground truth for ML training.

### Step 4: Train classifier
```bash
cd ML_study
python ml_profile_classifier.py --dir heatmap/short --suffix _short --force_model CalibratedRF
```
Outputs: `profile_classifier_short.joblib`, `*_predictions.xlsx`, `*_report.txt`, per-image overlay PNGs

### Step 5: Predict binding on new images
```bash
python predict.py tif/ --model short
python predict.py tif/ --model continuous
```
Outputs: `short_predictions.xlsx` (summary + runs + per-trace predictions), overlay PNGs with 3-panel layout

## Output Excel Structure

### predictions xlsx (from predict.py)
- **summary**: per-image N_pos, N_runs, run lengths
- **runs**: per positive run edge width (px, nm), left/right edge positions
- **predictions**: per-trace-point proba_raw, proba_smoothed, pred (0/1)
- **{image}**: per-image detail sheets

### profile_classifier xlsx (from ml_profile_classifier.py)
- **predictions**: CV per-trace-point true_yellow, proba, pred@best_threshold
- **runs**: per positive run edge width, signal_type (true/pred)

## Overlay PNG Layout

Three-panel visualization:
1. **Top**: Original image + scatter (TP/FP/FN for training; Predicted for inference)
2. **Middle**: Smoothed probability curve + threshold line
3. **Bottom**: DNA trace colored by binding (green) / non-binding (gray) with segment lengths in nm

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `half_width` | 20 px | Perpendicular profile extraction radius |
| `NM_RANGE` | 65 nm | Common nm grid half-range for profile resampling |
| `NM_SAMPLES` | 41 | Number of samples on the nm grid |
| `smooth_w` | 7 | Post-processing smoothing window (trace-px) |
| `min_run` | 5 | Minimum positive run length (trace-px) |

## Scale Configuration

All scripts require `pxum_config.json` with per-image pixel scale in **px/um** (pixels per micrometer). Example:

```json
{
  "Alu1": 288.18,
  "rrvt_17": 69.06
}
```

If an image is missing from the config, the script will interactively prompt for the value and save it automatically. Pass `--no-interactive` to error instead.

## Citation

If you use this code, please cite:

> [Your paper reference here]
