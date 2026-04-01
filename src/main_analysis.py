import numpy as np
import pandas as pd
from pathlib import Path
from utils import load_and_prepare, average_tech_reps, GENE_COL
from stats import (
    compare_growth_rates,
    compare_datasets_full,
    compare_datasets_growth_controlled_matched,
    compare_datasets_growth_controlled_model,
)
from plots import (
    plot_top_expressed_genes_heatmap,
    volcano_plot,
    plot_formate_vs_phosphate_heatmap_from_model,
    plot_formate_vs_phosphate_simple_heatmap,
)
from stringdb import run_string_enrichment, make_enrichment_plots
from pathways import METHANOGENESIS_GENES, summarize_pathway_across_datasets, plot_pathway_log2fc, plot_pathway_growth_trends


def shared_significant_hits(df_a, df_b, label_a, label_b, fc_col="log2FC", fdr_col="fdr",
                            fdr_thresh=0.05, fc_thresh=0.5):
    req_cols = {GENE_COL, fc_col, fdr_col}
    if not req_cols.issubset(df_a.columns) or not req_cols.issubset(df_b.columns):
        return pd.DataFrame()

    a_sig = df_a[(df_a[fdr_col] < fdr_thresh) & (df_a[fc_col].abs() >= fc_thresh)].copy()
    b_sig = df_b[(df_b[fdr_col] < fdr_thresh) & (df_b[fc_col].abs() >= fc_thresh)].copy()

    merged = a_sig[[GENE_COL, fc_col, fdr_col]].merge(
        b_sig[[GENE_COL, fc_col, fdr_col]],
        on=GENE_COL,
        suffixes=(f"_{label_a}", f"_{label_b}")
    )

    if merged.empty:
        return merged

    merged["same_direction"] = (
        np.sign(merged[f"{fc_col}_{label_a}"]) == np.sign(merged[f"{fc_col}_{label_b}"])
    )

    return merged.sort_values([f"{fdr_col}_{label_a}", f"{fdr_col}_{label_b}"])


def main(show_plots=False, fc_thresh=0.5):
    file1 = "data/formate_dataset_cleaned.csv"
    file2 = "data/phosphate_dataset_cleaned.csv"

    name1 = "formate"
    name2 = "phosphate"

    outdir = Path("analysis_outputs")
    outdir.mkdir(exist_ok=True)

    df1 = load_and_prepare(file1, name1)
    df2 = load_and_prepare(file2, name2)

    df1_bio = average_tech_reps(df1)
    df2_bio = average_tech_reps(df2)

    plot_top_expressed_genes_heatmap(
        df1_bio,
        dataset_name=name1,
        outpath=outdir / f"heatmaps/{name1}_top_expressed_genes_heatmap.png",
        top_n=30,
        center_by_gene=True
    )

    plot_top_expressed_genes_heatmap(
        df2_bio,
        dataset_name=name2,
        outpath=outdir / f"heatmaps/{name2}_top_expressed_genes_heatmap.png",
        top_n=30,
        center_by_gene=True
    )

    df1_bio.to_csv(outdir / f"{name1}_bio_level.csv", index=False)
    df2_bio.to_csv(outdir / f"{name2}_bio_level.csv", index=False)

    rates1 = sorted(df1_bio["growth_rate"].dropna().unique())
    rates2 = sorted(df2_bio["growth_rate"].dropna().unique())
    common_rates = sorted(set(rates1).intersection(set(rates2)))

    print(f"{name1} growth rates: {rates1}")
    print(f"{name2} growth rates: {rates2}")
    print(f"Shared growth rates: {common_rates}")

    comparisons_1 = [("xs", "s")]
    comparisons_2 = [("xs", "i")]

    individual_results = {}

    for g1, g2 in comparisons_1:
        res = compare_growth_rates(df1_bio, g1, g2)
        individual_results[f"{name1}_{g1}_vs_{g2}"] = res

        out_csv = outdir / f"{name1}_{g1}_vs_{g2}.csv"
        res.to_csv(out_csv, index=False)

        print(f"\n{name1}: {g1} vs {g2}")
        print(res.head())

        if not res.empty:
            volcano_plot(
                res,
                x_col="log2FC",
                p_col="fdr",
                title=f"Volcano Plot – {name1} {g1.upper()} vs {g2.upper()}",
                outpath=outdir / f"{name1}_{g1}_vs_{g2}_volcano.png",
                fc_thresh=fc_thresh,
                sig_thresh=0.05,
                label_top_n=10
                , show=show_plots
            )

    for g1, g2 in comparisons_2:
        res = compare_growth_rates(df2_bio, g1, g2)
        individual_results[f"{name2}_{g1}_vs_{g2}"] = res

        out_csv = outdir / f"{name2}_{g1}_vs_{g2}.csv"
        res.to_csv(out_csv, index=False)

        print(f"\n{name2}: {g1} vs {g2}")
        print(res.head())

        if not res.empty:
            volcano_plot(
                res,
                x_col="log2FC",
                p_col="fdr",
                title=f"Volcano Plot – {name2} {g1.upper()} vs {g2.upper()}",
                outpath=outdir / f"{name2}_{g1}_vs_{g2}_volcano.png",
                fc_thresh=fc_thresh,
                sig_thresh=0.05,
                label_top_n=10
                , show=show_plots
            )

    key1 = f"{name1}_xs_vs_s"
    key2 = f"{name2}_xs_vs_i"

    if key1 in individual_results and key2 in individual_results:
        overlap = shared_significant_hits(
            individual_results[key1],
            individual_results[key2],
            label_a=key1,
            label_b=key2,
            fc_col="log2FC",
            fdr_col="fdr",
            fdr_thresh=0.05,
            fc_thresh=fc_thresh
        )
        overlap.to_csv(outdir / "shared_significant_genes_between_individual_analyses.csv", index=False)
        print("\nShared significant genes between individual analyses:")
        print(overlap.head())

    full_compare = compare_datasets_full(df1_bio, df2_bio, name1=name1, name2=name2)
    full_compare.to_csv(outdir / f"{name1}_vs_{name2}_full.csv", index=False)

    print(f"\nFull comparison: {name1} vs {name2}")
    print(full_compare.head())

    if not full_compare.empty:
        volcano_plot(
            full_compare,
            x_col="log2FC",
            p_col="fdr",
            title=f"Volcano Plot – {name1} vs {name2} (full)",
            outpath=outdir / f"{name1}_vs_{name2}_full_volcano.png",
            fc_thresh=fc_thresh,
            sig_thresh=0.05,
            label_top_n=10
            , show=show_plots
        )

    controlled_summary, controlled_per_rate = compare_datasets_growth_controlled_matched(
        df1_bio, df2_bio, name1=name1, name2=name2
    )

    controlled_summary.to_csv(
        outdir / f"{name1}_vs_{name2}_growth_controlled_matched_summary.csv",
        index=False
    )
    controlled_per_rate.to_csv(
        outdir / f"{name1}_vs_{name2}_growth_controlled_matched_per_rate.csv",
        index=False
    )

    if not controlled_summary.empty:
        save_cols = controlled_summary.rename(columns={"avg_log2FC": "log2FC", "summary_fdr": "fdr"})
        save_cols.to_csv(outdir / f"{name1}_vs_{name2}_growth_controlled_matched_top_hits.csv", index=False)

    print(f"\nGrowth-controlled matched comparison: {name1} vs {name2}")
    print(controlled_summary.head())

    model_compare = compare_datasets_growth_controlled_model(
        df1_bio, df2_bio, name1=name1, name2=name2
    )

    plot_formate_vs_phosphate_heatmap_from_model(
        df1_bio,
        df2_bio,
        model_compare,
        name1=name1,
        name2=name2,
        outpath=outdir / f"heatmaps/{name1}_vs_{name2}_model_heatmap.png",
        top_n=30,
        center_by_gene=True,
        sort_by="fdr"
        , show=show_plots
    )

    plot_formate_vs_phosphate_simple_heatmap(
        df1_bio,
        df2_bio,
        model_compare,
        name1=name1,
        name2=name2,
        outpath=outdir / f"heatmaps/{name1}_vs_{name2}_simple_heatmap.png",
        top_n=30,
        center_by_gene=False
        , show=show_plots
    )

    model_compare.to_csv(
        outdir / f"{name1}_vs_{name2}_growth_controlled_model.csv",
        index=False
    )

    print(f"\nGrowth-controlled model comparison: {name1} vs {name2}")
    print(model_compare.head())

    if not model_compare.empty:
        volcano_plot(
            model_compare.rename(columns={"dataset_effect_log2FC": "log2FC"}),
            x_col="log2FC",
            p_col="fdr",
            title=f"Volcano Plot – {name1} vs {name2} (model growth-rate controlled)",
            outpath=outdir / f"{name1}_vs_{name2}_growth_controlled_model_volcano.png",
            fc_thresh=fc_thresh,
            sig_thresh=0.05,
            label_top_n=10
            , show=show_plots
        )

    # Track targeted methanogenesis pathway genes
    try:
        print("Summarizing methanogenesis pathway genes...")
        pathway_summary = summarize_pathway_across_datasets(df1_bio, df2_bio, METHANOGENESIS_GENES, name1, name2, outdir)
        pathway_csv = outdir / f"methanogenesis_pathway_summary_{name1}_vs_{name2}.csv"
        pathway_summary.to_csv(pathway_csv, index=False)

        plot_pathway_log2fc(pathway_summary, outpath=outdir / f"methanogenesis_pathway_log2fc_{name1}_vs_{name2}.png", show=show_plots)

        plot_pathway_growth_trends(df1_bio, df2_bio, METHANOGENESIS_GENES, name1, name2, outpath=outdir / f"methanogenesis_pathway_growth_trends_{name1}_vs_{name2}.png", show=show_plots)
        print(f"Methanogenesis pathway summary and plots saved to {outdir}")
    except Exception as e_path:
        print(f"Failed to produce pathway summaries/plots: {e_path}")

    # Run STRING-db enrichment on top model-selected genes (best FDR + effect)
    try:
        top_genes = model_compare.sort_values(["fdr", "dataset_effect_log2FC"], ascending=[True, False])[GENE_COL].head(100).tolist()
        if top_genes:
            print(f"Running STRING enrichment on top {len(top_genes)} model genes...")
            tsv_path = outdir / "string_enrichment_top_model_genes.tsv"
            enrich_df = run_string_enrichment(top_genes, species=2187, limit=500, output_path=str(tsv_path))
            enrich_df.to_csv(outdir / "string_enrichment_top_model_genes.csv", index=False)
            print("STRING enrichment saved to analysis_outputs/")

            # automatically generate enrichment plots (bar + dotplot)
            try:
                print("Generating enrichment plots...")
                made = make_enrichment_plots(str(tsv_path), outdir=outdir, prefix=tsv_path.stem, show=show_plots)
                for p in made:
                    print(f"Saved enrichment plot: {p}")
            except Exception as e_plot:
                print(f"Failed to create enrichment plots: {e_plot}")
    except Exception as e:
        print(f"STRING enrichment failed: {e}")

    print(f"\nDone. Outputs saved in: {outdir.resolve()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run modular expression analysis")
    parser.add_argument("--show-plots", action="store_true", help="Display plots interactively in addition to saving them")
    parser.add_argument("--fc-thresh", type=float, default=0.5, help="Fold-change cutoff (absolute log2) for volcano and shared-hit filtering; default 0.5")
    args = parser.parse_args()

    main(show_plots=args.show_plots, fc_thresh=args.fc_thresh)
