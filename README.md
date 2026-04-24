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
(4) ml_profile_classifier.py -- Train single CalibratedRF classifier on
                                 labeled profiles (nm-resampled features)
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

## Quick start with the bundled example

A single Alu image (`example/Alu4.tif`) and its labeled heatmap (`example/Alu4_profile_heatmap.xlsx`) are bundled so the full pipeline can be verified without any external data.

```bash
# 1. Initialize the scale config from the shipped example (already contains Alu4)
cp pxum_config.example.json pxum_config.json

# 2. Train a classifier from the labeled heatmap
python ML_study/ml_profile_classifier.py --dir example/

# 3. Predict on the same tif
python ML_study/predict.py example/Alu4.tif --out_dir example_out --no-interactive

# 4. Generate the feature-analysis figure
python ML_study/plot_feature_analysis.py --image Alu4 --xlsx_dir example/ \
    --out_path example_out/feature_analysis.png
```

Expected outputs:
* `ML_study/profile_classifier.joblib` — trained model (CV F1 ≈ 0.93 on this single image)
* `example_out/Alu4_predict.png` — 3-panel overlay
* `example_out/predictions.xlsx` — per-trace predictions + run summaries
* `example_out/feature_analysis.png` — two-panel feature figure

## Full workflow (with your own data)

The bundled Alu4 example exercises only step 4 and later. For end-to-end processing from raw SEM/TEM images, run the steps below in order.

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
python ml_profile_classifier.py --dir path/to/heatmap_xlsx/
```
Trains a single CalibratedRF pipeline on all `*_profile_heatmap.xlsx` files in the given directory. Profile cross-sections are nm-resampled to a common ±65 nm / 41-sample grid so images at different magnifications can be mixed. After CV, a grid sweep over `(smooth_w × min_run × threshold)` auto-selects the best post-processing.

Outputs: `profile_classifier.joblib`, `profile_classifier_predictions.xlsx`, `profile_classifier_report.txt`, per-image overlay PNGs.

### Step 5: Predict binding on new images
```bash
python predict.py tif/
python predict.py tif/ --no-interactive    # skip endpoint GUI picker
```
Outputs: `predictions.xlsx` (summary + runs + per-trace predictions), overlay PNGs with 3-panel layout.

### (Optional) Feature-analysis figure
```bash
python plot_feature_analysis.py --image Alu4 --xlsx_dir path/to/heatmap_xlsx/
```
Produces a two-panel figure: (a) mean bare vs protein-bound profile for the chosen image with SEM bands, (b) top feature importances of the trained classifier.

## Output Excel Structure

### predictions.xlsx (from predict.py)
- **summary**: per-image N_pos, N_runs, run lengths
- **runs**: per positive run edge width (px, nm), left/right edge positions
- **predictions**: per-trace-point proba_raw, proba_smoothed, pred (0/1)
- **{image}**: per-image detail sheets

### profile_classifier_predictions.xlsx (from ml_profile_classifier.py)
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
| `smooth_w` | auto | Post-processing smoothing window (trace-px, swept over {1,3,5,7,9,11,15,21}) |
| `min_run` | auto | Minimum positive run length (trace-px, swept over {1,3,5,7,9,13,17,25}) |
| `threshold` | auto | Decision threshold (grid-searched over 0.05–0.95 by F1) |

## Scale Configuration

All scripts require `pxum_config.json` with per-image pixel scale in **px/um** (pixels per micrometer). Example:

```json
{
  "Alu1": 288.18
}
```

If an image is missing from the config, the script will interactively prompt for the value and save it automatically. Pass `--no-interactive` to error instead.

## License

This project is released under the MIT License — see [LICENSE](LICENSE) for details.

## Citation

If you use this code, please cite:

> Chanyoung Noh *et al.* "DNA-protein binding detection from SEM/TEM images using perpendicular profile classification." Manuscript in preparation, 2026.

Update this entry with the final journal reference and DOI once published.

## Contact

Questions, issues, or pull requests are welcome via the [issue tracker](https://github.com/ChanyoungNoh/SEM_DNA_Analysis/issues).
