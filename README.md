# Proteomics Analysis Pipeline

Simple pipeline to compare **formate vs phosphate** proteomics data.

## Workflow

1. Clean raw data
2. Run analysis
3. Check outputs in `analysis_outputs/`

## Inputs

Place raw files in `data/`:
- `formate_dataset.xlsx`
- `phospate_dataset.xlsx`

## Preprocessing (required)

```bash
python clean_formate_data.py
python clean_phosphate_data.py
```

Creates:
- `data/formate_dataset_cleaned.csv`
- `data/phosphate_dataset_cleaned.csv`

## Run

```bash
python main_analysis.py
```

## Outputs

All results are saved in:
```
analysis_outputs/
```

### Key files

**Core results**
- `formate_bio_level.csv`
- `phosphate_bio_level.csv`
- `formate_xs_vs_s.csv`
- `phosphate_xs_vs_i.csv`
- `shared_significant_genes_between_individual_analyses.csv`
- `formate_vs_phosphate_full.csv`
- `formate_vs_phosphate_growth_controlled_matched_summary.csv`
- `formate_vs_phosphate_growth_controlled_matched_per_rate.csv`
- `formate_vs_phosphate_growth_controlled_model.csv`

**Plots**
- Volcano plots (all comparisons)
- Heatmaps in `analysis_outputs/heatmaps/`

**Pathway + enrichment (if run)**
- Methanogenesis summary + plot
- STRING enrichment CSV/TSV + plots

## Notes

- Always run cleaning scripts first
- Main analysis uses cleaned CSVs
- Primary result = **growth-controlled model comparison**

