import numpy as np
import pandas as pd
from pathlib import Path
from typing import Iterable, Optional
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
from stringdb import (
    run_string_enrichment,
    make_enrichment_plots,
    run_directional_string_enrichment,
    plot_pathway_venn_from_enrichment,
    save_pathway_overlap_table_from_enrichment,
)
from pathways import METHANOGENESIS_GENES, summarize_pathway_across_datasets, plot_pathway_log2fc


def shared_significant_hits(df_a, df_b, label_a, label_b, fc_col="log2FC", fdr_col="fdr",
                            fdr_thresh=0.1, fc_thresh=0.5):
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

def print_analysis_summary(
    df1_bio,
    df2_bio,
    full_compare,
    controlled_summary,
    model_compare,
    gene_col=GENE_COL,
    fc_thresh=0.5,
    fdr_thresh=0.05,
    enrich_outputs=None,
    name1="formate",
    name2="phosphate",
):
    """
    Print a broad end-of-run summary of quantified proteins and differential abundance.
    """
    print("\n" + "=" * 80)
    print("OVERALL ANALYSIS SUMMARY")
    print("=" * 80)

    # -------------------------
    # Quantified proteins
    # -------------------------
    q1 = set(df1_bio[gene_col].dropna().astype(str).unique()) if gene_col in df1_bio.columns else set()
    q2 = set(df2_bio[gene_col].dropna().astype(str).unique()) if gene_col in df2_bio.columns else set()

    union_q = q1.union(q2)
    overlap_q = q1.intersection(q2)

    print("\nQuantified proteins")
    print("-" * 80)
    print(f"{name1}: {len(q1)} unique quantified proteins")
    print(f"{name2}: {len(q2)} unique quantified proteins")
    print(f"Combined total (union): {len(union_q)}")
    print(f"Quantified in both datasets: {len(overlap_q)}")

    # -------------------------
    # Full comparison summary
    # -------------------------
    print("\nDifferential abundance: full comparison")
    print("-" * 80)
    if full_compare is not None and not full_compare.empty:
        fc_col = "log2FC"
        sig_col = "fdr"

        tested_full = full_compare[gene_col].dropna().nunique() if gene_col in full_compare.columns else len(full_compare)
        sig_full = full_compare[
            full_compare[sig_col].lt(fdr_thresh) & full_compare[fc_col].abs().ge(fc_thresh)
        ].copy()

        up_1_full = sig_full[sig_full[fc_col] > 0]
        up_2_full = sig_full[sig_full[fc_col] < 0]

        print(f"Proteins tested: {tested_full}")
        print(f"Significant at FDR < {fdr_thresh} and |log2FC| >= {fc_thresh}: {len(sig_full)}")
        print(f"  Higher in {name1}: {len(up_1_full)}")
        print(f"  Higher in {name2}: {len(up_2_full)}")
    else:
        print("No full comparison results available.")

    # -------------------------
    # Growth-controlled matched summary
    # -------------------------
    print("\nDifferential abundance: growth-controlled matched summary")
    print("-" * 80)
    if controlled_summary is not None and not controlled_summary.empty:
        tmp = controlled_summary.copy()
        fc_col = "avg_log2FC"
        sig_col = "summary_fdr"

        tested_ctrl = tmp[gene_col].dropna().nunique() if gene_col in tmp.columns else len(tmp)
        sig_ctrl = tmp[
            tmp[sig_col].lt(fdr_thresh) & tmp[fc_col].abs().ge(fc_thresh)
        ].copy()

        up_1_ctrl = sig_ctrl[sig_ctrl[fc_col] > 0]
        up_2_ctrl = sig_ctrl[sig_ctrl[fc_col] < 0]

        print(f"Proteins tested: {tested_ctrl}")
        print(f"Significant at FDR < {fdr_thresh} and |avg_log2FC| >= {fc_thresh}: {len(sig_ctrl)}")
        print(f"  Higher in {name1}: {len(up_1_ctrl)}")
        print(f"  Higher in {name2}: {len(up_2_ctrl)}")
    else:
        print("No growth-controlled matched summary available.")

    # -------------------------
    # Growth-controlled model summary
    # -------------------------
    print("\nDifferential abundance: growth-controlled model")
    print("-" * 80)
    if model_compare is not None and not model_compare.empty:
        fc_col = "dataset_effect_log2FC"
        sig_col = "fdr"

        tested_model = model_compare[gene_col].dropna().nunique() if gene_col in model_compare.columns else len(model_compare)
        sig_model = model_compare[
            model_compare[sig_col].lt(fdr_thresh) & model_compare[fc_col].abs().ge(fc_thresh)
        ].copy()

        up_1_model = sig_model[sig_model[fc_col] > 0]
        up_2_model = sig_model[sig_model[fc_col] < 0]

        print(f"Proteins tested: {tested_model}")
        print(f"Significant at FDR < {fdr_thresh} and |dataset_effect_log2FC| >= {fc_thresh}: {len(sig_model)}")
        print(f"  Higher in {name1}: {len(up_1_model)}")
        print(f"  Higher in {name2}: {len(up_2_model)}")

        # top hits preview
        print("\nTop significant model hits")
        top_hits = model_compare.sort_values(["fdr", "dataset_effect_log2FC"], ascending=[True, False]).head(10)
        preview_cols = [c for c in [gene_col, "dataset_effect_log2FC", "fdr"] if c in top_hits.columns]
        if not top_hits.empty and preview_cols:
            print(top_hits[preview_cols].to_string(index=False))
    else:
        print("No growth-controlled model results available.")

    if enrich_outputs is not None:
        print("\nDirectional pathway enrichment")
        print("-" * 80)

        if "formate" in enrich_outputs:
            n_formate = len(enrich_outputs["formate"].get("genes", []))
            print(f"Genes sent to STRING for {name1}-up enrichment: {n_formate}")

        if "phosphate" in enrich_outputs:
            n_phosphate = len(enrich_outputs["phosphate"].get("genes", []))
            print(f"Genes sent to STRING for {name2}-up enrichment: {n_phosphate}")

        if "formate" not in enrich_outputs and "phosphate" not in enrich_outputs:
            print("No directional enrichment outputs generated.")

    print("\nInterpretation")
    print("-" * 80)
    print(
        f"This analysis quantified {len(union_q)} total proteins across both datasets, "
        f"with {len(overlap_q)} observed in both. The growth-controlled model is the best overall "
        f"summary for {name1} vs {name2} because it adjusts for growth-rate effects while estimating "
        f"the dataset effect directly."
    )

    print("=" * 80 + "\n")

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
        sort_by="fdr",
        aggregate_replicates=True,
        show_significance=True,
        show=show_plots
    )

    plot_formate_vs_phosphate_simple_heatmap(
        df1_bio,
        df2_bio,
        model_compare,
        name1=name1,
        name2=name2,
        outpath=outdir / f"heatmaps/{name1}_vs_{name2}_simple_heatmap.png",
        top_n=30,
        center_by_gene=False,
        show_significance=True,
        show_direction=True,
        show=show_plots
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
            label_top_n=10
            , show=show_plots
        )
    enrich_outputs = {}
    # Track targeted methanogenesis pathway genes
    try:
        print("Summarizing methanogenesis pathway genes...")
        pathway_summary = summarize_pathway_across_datasets(df1_bio, df2_bio, METHANOGENESIS_GENES, name1, name2, outdir)
        pathway_csv = outdir / f"methanogenesis_pathway_summary_{name1}_vs_{name2}.csv"
        pathway_summary.to_csv(pathway_csv, index=False)

        plot_pathway_log2fc(pathway_summary, outpath=outdir / f"methanogenesis_pathway_log2fc_{name1}_vs_{name2}.png", show=show_plots)

        # plot_pathway_growth_trends(df1_bio, df2_bio, METHANOGENESIS_GENES, name1, name2, outpath=outdir / f"methanogenesis_pathway_growth_trends_{name1}_vs_{name2}.png", show=show_plots)
        print(f"Methanogenesis pathway summary and plots saved to {outdir}")
    except Exception as e_path:
        print(f"Failed to produce pathway summaries/plots: {e_path}")

    # Run STRING-db enrichment on top model-selected genes (best FDR + effect)
        # Run STRING-db enrichment on top model-selected genes (best FDR + effect)
      # Run STRING-db enrichment separately for genes up in formate vs up in phosphate
    try:
        print("Running directional STRING enrichment...")

        enrich_outputs = run_directional_string_enrichment(
            model_compare=model_compare,
            gene_col=GENE_COL,
            effect_col="dataset_effect_log2FC",
            fdr_col="fdr",
            fdr_threshold=0.1,      # optional; set to None to use all genes
            effect_threshold=0.0,   # optional minimum abs log2FC cutoff
            top_n=100,              # number of genes per direction
            species=2187,
            outdir=outdir,
            prefix="string_enrichment_model_genes"
        )

        if "formate" in enrich_outputs:
            print(f"Generating enrichment plots for genes up in {name1}...")
            made_formate = make_enrichment_plots(
                enrich_outputs["formate"]["tsv"],
                outdir=outdir,
                prefix="string_enrichment_up_in_formate",
                show=show_plots
            )
            for p in made_formate:
                print(f"Saved formate enrichment plot: {p}")

        if "phosphate" in enrich_outputs:
            print(f"Generating enrichment plots for genes up in {name2}...")
            made_phosphate = make_enrichment_plots(
                enrich_outputs["phosphate"]["tsv"],
                outdir=outdir,
                prefix="string_enrichment_up_in_phosphate",
                show=show_plots
            )
            for p in made_phosphate:
                print(f"Saved phosphate enrichment plot: {p}")

        if not enrich_outputs:
            print("No genes met the directional enrichment criteria.")
        if "formate" in enrich_outputs and "phosphate" in enrich_outputs:
                    
            plot_pathway_venn_from_enrichment(
                enrich_outputs["formate"]["tsv"],
                enrich_outputs["phosphate"]["tsv"],
                outpath=outdir / "string_enrichment_formate_vs_phosphate_venn.png",
                pval_threshold=0.05,      # or 0.1 if you want looser significance
                use_fdr=False,            # set True to use FDR instead
                use_description_only=False,
                show=show_plots
            )
            print("Saved pathway Venn diagram.")

            overlap_table = save_pathway_overlap_table_from_enrichment(
                enrich_outputs["formate"]["tsv"],
                enrich_outputs["phosphate"]["tsv"],
                outpath=outdir / "string_enrichment_formate_vs_phosphate_venn_table.csv",
                pval_threshold=0.05,
                use_fdr=False,
                use_description_only=False
            )
            print(f"Saved pathway overlap table with {len(overlap_table)} rows.")
            print_analysis_summary(
                df1_bio=df1_bio,
                df2_bio=df2_bio,
                full_compare=full_compare,
                controlled_summary=controlled_summary,
                model_compare=model_compare,
                gene_col=GENE_COL,
                fc_thresh=fc_thresh,
                fdr_thresh=0.05,
                enrich_outputs=enrich_outputs,
                name1=name1,
                name2=name2,
            )

            print(f"\nDone. Outputs saved in: {outdir.resolve()}")
    except Exception as e:
        print(f"Directional STRING enrichment failed: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run modular expression analysis")
    parser.add_argument("--show-plots", action="store_true", help="Display plots interactively in addition to saving them")
    parser.add_argument("--fc-thresh", type=float, default=0.5, help="Fold-change cutoff (absolute log2) for volcano and shared-hit filtering; default 0.5")
    args = parser.parse_args()

    main(show_plots=args.show_plots, fc_thresh=args.fc_thresh)
