import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from utils import GENE_COL, safe_neglog10
from typing import Iterable, Optional
# Visual theming: seaborn base style + a small set of pretty colors
sns.set(style="whitegrid")
POS_COLOR = "#E64B35"   # warm orange/red for positive effects
NEG_COLOR = "#4A90E2"   # cool blue for negative effects
NEUTRAL_COLOR = "#9EA7AA"  # muted grey for non-significant
DIV_CMAP = sns.diverging_palette(240, 10, as_cmap=True)


def volcano_plot(
    results,
    x_col,
    p_col,
    title,
    outpath=None,
    fc_thresh=0.5,
    sig_thresh=0.1,
    label_top_n=0,
    gene_col=GENE_COL,
    show=False
):
    """
    Make a volcano plot from any results table with effect size and p/FDR columns.
    """
    plot_df = results.copy()
    plot_df = plot_df[plot_df[x_col].notna() & plot_df[p_col].notna()].copy()

    if plot_df.empty:
        print(f"Skipping plot '{title}' because there are no plottable rows.")
        return

    plot_df["neglog10_p"] = safe_neglog10(plot_df[p_col])

    # mark points passing both effect size and significance thresholds
    sig_mask = (plot_df[p_col] < sig_thresh) & (plot_df[x_col].abs() >= fc_thresh)

    plt.figure(figsize=(8, 6))
    pos_sig = sig_mask & (plot_df[x_col] > 0)
    neg_sig = sig_mask & (plot_df[x_col] < 0)
    non_sig = ~sig_mask

    plt.scatter(plot_df.loc[non_sig, x_col], plot_df.loc[non_sig, "neglog10_p"], s=14, alpha=0.5, color=NEUTRAL_COLOR)
    plt.scatter(plot_df.loc[neg_sig, x_col], plot_df.loc[neg_sig, "neglog10_p"], s=28, alpha=0.9, color=NEG_COLOR)
    plt.scatter(plot_df.loc[pos_sig, x_col], plot_df.loc[pos_sig, "neglog10_p"], s=28, alpha=0.95, color=POS_COLOR)

    # reference lines
    plt.axvline(0, linewidth=1, color=NEUTRAL_COLOR)
    plt.axvline(fc_thresh, linestyle="--", linewidth=1, color=NEUTRAL_COLOR)
    plt.axvline(-fc_thresh, linestyle="--", linewidth=1, color=NEUTRAL_COLOR)
    plt.axhline(-np.log10(sig_thresh), linestyle="--", linewidth=1, color=NEUTRAL_COLOR)

    plt.xlabel("log2 Fold Change")
    plt.ylabel(f"-log10({p_col})")
    plt.title(title)
    plt.tight_layout()

    # optionally label a small number of top hits
    if label_top_n > 0 and gene_col in plot_df.columns:
        top_hits = (
            plot_df.sort_values([p_col, x_col], ascending=[True, False])
            .head(label_top_n)
        )
        for _, row in top_hits.iterrows():
            plt.text(row[x_col], row["neglog10_p"], str(row[gene_col]), fontsize=7)

    if outpath is not None:
        plt.savefig(outpath, dpi=300)

    # only display the figure when explicitly requested; otherwise close it so it
    # doesn't pop up in interactive environments
    if show:
        plt.show()
    else:
        plt.close()


def plot_top_expressed_genes_heatmap(
    df_bio,
    dataset_name,
    outpath=None,
    top_n=30,
    center_by_gene=True,
    gene_col=GENE_COL,
    show=False
):
    """
    Make a heatmap of the top most expressed genes based on mean log2 expression.
    """
    if df_bio.empty:
        print(f"Skipping heatmap for {dataset_name}: empty dataframe.")
        return

    # find top genes by average expression across all samples
    top_genes = (
        df_bio.groupby(gene_col)["log2_ratio"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )

    heat_df = df_bio[df_bio[gene_col].isin(top_genes)].copy()
    if heat_df.empty:
        print(f"Skipping heatmap for {dataset_name}: no genes available after filtering.")
        return

    # create a sample-like column label
    heat_df["sample"] = (
        heat_df["growth_rate"].astype(str) + "_rep" + heat_df["bio_rep"].astype(str)
    )

    # pivot to genes x samples
    heatmap_data = heat_df.pivot_table(
        index=gene_col,
        columns="sample",
        values="log2_ratio",
        aggfunc="mean"
    )

    # order genes by overall mean expression
    gene_order = heat_df.groupby(gene_col)["log2_ratio"].mean().sort_values(ascending=False).index
    heatmap_data = heatmap_data.loc[[g for g in gene_order if g in heatmap_data.index]]

    # optional row-centering to emphasize relative differences across samples
    if center_by_gene:
        heatmap_plot = heatmap_data.sub(heatmap_data.mean(axis=1), axis=0)
        colorbar_label = "Centered log2 expression"
    else:
        heatmap_plot = heatmap_data.copy()
        colorbar_label = "log2 expression"

    plt.figure(figsize=(max(8, heatmap_plot.shape[1] * 0.6), max(8, heatmap_plot.shape[0] * 0.3)))
    # use blue-red diverging colormap and symmetric scaling when centered
    cmap = DIV_CMAP
    try:
        cmap.set_bad("black")
    except Exception:
        pass
    masked = np.ma.masked_invalid(heatmap_plot.values)
    if center_by_gene:
        max_abs = np.nanmax(np.abs(heatmap_plot.values)) if heatmap_plot.size else None
        if max_abs is None:
            im = plt.imshow(masked, aspect="auto", cmap=cmap)
        else:
            im = plt.imshow(masked, aspect="auto", cmap=cmap, vmin=-max_abs, vmax=max_abs)
    else:
        im = plt.imshow(masked, aspect="auto", cmap=cmap)
    plt.colorbar(im, label=colorbar_label)

    plt.xticks(range(len(heatmap_plot.columns)), heatmap_plot.columns, rotation=90)
    plt.yticks(range(len(heatmap_plot.index)), heatmap_plot.index)

    plt.xlabel("Sample")
    plt.ylabel("Gene")
    plt.title(f"Top {top_n} Most Expressed Genes – {dataset_name}")
    plt.tight_layout()

    # ensure heatmaps go into analysis_outputs/heatmaps by default
    if outpath is None:
        outp = Path("analysis_outputs") / "heatmaps" / f"top_expressed_{dataset_name}.png"
    else:
        outp = Path(outpath)
    outp.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outp, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

def plot_formate_vs_phosphate_heatmap_from_model(
    df1_bio,
    df2_bio,
    model_results,
    name1="formate",
    name2="phosphate",
    outpath=None,
    top_n=30,
    gene_col=GENE_COL,
    center_by_gene=True,
    sort_by="fdr",
    aggregate_replicates=True,
    show_significance=True,
    show=False
):
    """
    Heatmap comparing formate vs phosphate across shared growth rates.

    Changes:
    - aggregates biological replicates across each dataset/growth_rate/gene
    - annotates gene labels with significance from model_results FDR
    """
    if model_results.empty:
        print("Skipping formate/phosphate heatmap: model_results is empty.")
        return

    x1 = df1_bio.copy()
    x2 = df2_bio.copy()
    x1["dataset"] = name1
    x2["dataset"] = name2

    # keep only shared growth rates
    common_rates = sorted(set(x1["growth_rate"]).intersection(set(x2["growth_rate"])))
    x1 = x1[x1["growth_rate"].isin(common_rates)].copy()
    x2 = x2[x2["growth_rate"].isin(common_rates)].copy()

    if x1.empty or x2.empty:
        print("Skipping formate/phosphate heatmap: no shared growth rates.")
        return

    combined = pd.concat([x1, x2], ignore_index=True)

    # rank genes from model results
    rank_df = model_results.copy().dropna(subset=[gene_col])
    if "dataset_effect_log2FC" not in rank_df.columns or "fdr" not in rank_df.columns:
        print("Skipping formate/phosphate heatmap: model_results missing required columns.")
        return

    if sort_by == "effect":
        rank_df = rank_df.reindex(
            rank_df["dataset_effect_log2FC"].abs().sort_values(ascending=False).index
        )
    else:
        rank_df = rank_df.sort_values(
            ["fdr", "dataset_effect_log2FC"],
            ascending=[True, False],
            na_position="last"
        )

    top_genes = rank_df[gene_col].head(top_n).tolist()

    heat_df = combined[combined[gene_col].isin(top_genes)].copy()
    if heat_df.empty:
        print("Skipping formate/phosphate heatmap: no matching genes found in expression table.")
        return

    # -------- AGGREGATE REPLICATES --------
    if aggregate_replicates:
        heat_df = (
            heat_df.groupby([gene_col, "dataset", "growth_rate"], as_index=False)["log2_ratio"]
            .mean()
        )
        heat_df["sample"] = (
            heat_df["dataset"].astype(str) + "_" +
            heat_df["growth_rate"].astype(str)
        )
    else:
        heat_df["sample"] = (
            heat_df["dataset"].astype(str) + "_" +
            heat_df["growth_rate"].astype(str) + "_rep" +
            heat_df["bio_rep"].astype(str)
        )

    heatmap_data = heat_df.pivot_table(
        index=gene_col,
        columns="sample",
        values="log2_ratio",
        aggfunc="mean"
    )

    # preserve model-based ranking order
    gene_order = [g for g in top_genes if g in heatmap_data.index]
    heatmap_data = heatmap_data.loc[gene_order]

    # optional row-centering
    if center_by_gene:
        heatmap_plot = heatmap_data.sub(heatmap_data.mean(axis=1), axis=0)
        cbar_label = "Centered log2 expression"
    else:
        heatmap_plot = heatmap_data.copy()
        cbar_label = "Mean log2 expression"

    # -------- SIGNIFICANCE LABELS --------
    def sig_stars(fdr):
        if pd.isna(fdr):
            return ""
        elif fdr < 0.001:
            return "***"
        elif fdr < 0.01:
            return "**"
        elif fdr < 0.05:
            return "*"
        elif fdr < 0.1:
            return "·"
        return ""

    if show_significance:
        fdr_map = rank_df.drop_duplicates(subset=[gene_col]).set_index(gene_col)["fdr"].to_dict()
        y_labels = [f"{g} {sig_stars(fdr_map.get(g, np.nan))}" for g in heatmap_plot.index]
    else:
        y_labels = list(heatmap_plot.index)

    plt.figure(figsize=(max(10, heatmap_plot.shape[1] * 0.6), max(8, heatmap_plot.shape[0] * 0.3)))

    cmap = DIV_CMAP
    try:
        cmap.set_bad("black")
    except Exception:
        pass

    masked = np.ma.masked_invalid(heatmap_plot.values)

    if center_by_gene:
        max_abs = np.nanmax(np.abs(heatmap_plot.values)) if heatmap_plot.size else None
        if max_abs is None or np.isnan(max_abs):
            im = plt.imshow(masked, aspect="auto", cmap=cmap)
        else:
            im = plt.imshow(masked, aspect="auto", cmap=cmap, vmin=-max_abs, vmax=max_abs)
    else:
        im = plt.imshow(masked, aspect="auto", cmap=cmap)

    plt.colorbar(im, label=cbar_label)
    plt.xticks(range(len(heatmap_plot.columns)), heatmap_plot.columns, rotation=90)
    plt.yticks(range(len(heatmap_plot.index)), y_labels)

    plt.xlabel("Condition")
    plt.ylabel("Gene")
    plt.title(f"Top {len(gene_order)} Model-Based Genes: {name1} vs {name2}")

    if show_significance:
        plt.suptitle("* FDR<0.05, ** FDR<0.01, *** FDR<0.001, · FDR<0.1", y=1.02, fontsize=9)

    plt.tight_layout()

    if outpath is None:
        outp = Path("analysis_outputs") / "heatmaps" / f"model_heatmap_{name1}_vs_{name2}.png"
    else:
        outp = Path(outpath)

    outp.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outp, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()
        
def plot_formate_vs_phosphate_simple_heatmap(
    df1_bio,
    df2_bio,
    model_results,
    name1="formate",
    name2="phosphate",
    outpath=None,
    top_n=30,
    gene_col=GENE_COL,
    center_by_gene=False,
    show_significance=True,
    show_direction=True,
    show=False
):
    """
    2-row heatmap:
    Rows = [formate, phosphate]
    Columns = genes (top from model)
    Values = mean log2 expression per dataset

    Reference:
    - Model selection: dataset_effect_log2FC = formate - phosphate
      so positive effect means higher in formate, negative means higher in phosphate.
    - Heatmap colors:
        * center_by_gene=False -> absolute mean log2 expression
        * center_by_gene=True  -> centered within each gene across the two conditions
    """

    if model_results.empty:
        print("Skipping heatmap: model_results empty.")
        return

    req_cols = {gene_col, "dataset_effect_log2FC", "fdr"}
    if not req_cols.issubset(model_results.columns):
        print("Skipping heatmap: model_results missing required columns.")
        return

    rank_df = (
        model_results.copy()
        .dropna(subset=[gene_col, "dataset_effect_log2FC", "fdr"])
        .assign(abs_effect=lambda d: d["dataset_effect_log2FC"].abs())
        .sort_values(["fdr", "abs_effect"], ascending=[True, False])
    )

    top_genes = rank_df[gene_col].head(top_n).tolist()
    if not top_genes:
        print("Skipping heatmap: no top genes found.")
        return

    mean1 = (
        df1_bio[df1_bio[gene_col].isin(top_genes)]
        .groupby(gene_col)["log2_ratio"]
        .mean()
        .rename(name1)
    )

    mean2 = (
        df2_bio[df2_bio[gene_col].isin(top_genes)]
        .groupby(gene_col)["log2_ratio"]
        .mean()
        .rename(name2)
    )

    heatmap_data = pd.concat([mean1, mean2], axis=1).T

    if heatmap_data.empty:
        print("Skipping heatmap: no overlapping genes.")
        return

    ordered_genes = [g for g in top_genes if g in heatmap_data.columns]
    heatmap_data = heatmap_data[ordered_genes]

    if center_by_gene:
        heatmap_plot = heatmap_data - heatmap_data.mean(axis=0)
        cbar_label = "Centered log2 expression"
    else:
        heatmap_plot = heatmap_data.copy()
        cbar_label = "Mean log2 expression"

    def sig_stars(fdr):
        if pd.isna(fdr):
            return ""
        if fdr < 0.001:
            return "***"
        if fdr < 0.01:
            return "**"
        if fdr < 0.05:
            return "*"
        if fdr < 0.1:
            return "·"
        return ""

    fdr_map = rank_df.drop_duplicates(subset=[gene_col]).set_index(gene_col)["fdr"].to_dict()
    effect_map = rank_df.drop_duplicates(subset=[gene_col]).set_index(gene_col)["dataset_effect_log2FC"].to_dict()

    x_labels = []
    for g in heatmap_plot.columns:
        label = g
        if show_direction:
            effect = effect_map.get(g, np.nan)
            if pd.notna(effect):
                label += " ↑F" if effect > 0 else " ↑P"
        if show_significance:
            label += sig_stars(fdr_map.get(g, np.nan))
        x_labels.append(label)

    plt.figure(figsize=(max(10, len(heatmap_plot.columns) * 0.45), 3.2))

    cmap = DIV_CMAP
    try:
        cmap.set_bad("black")
    except Exception:
        pass

    masked = np.ma.masked_invalid(heatmap_plot.values)

    if center_by_gene:
        max_abs = np.nanmax(np.abs(heatmap_plot.values)) if heatmap_plot.size else None
        if max_abs is None or np.isnan(max_abs):
            im = plt.imshow(masked, aspect="auto", cmap=cmap)
        else:
            im = plt.imshow(masked, aspect="auto", cmap=cmap, vmin=-max_abs, vmax=max_abs)
    else:
        im = plt.imshow(masked, aspect="auto", cmap=cmap)

    plt.colorbar(im, label=cbar_label)
    plt.xticks(range(len(heatmap_plot.columns)), x_labels, rotation=90)
    plt.yticks(range(len(heatmap_plot.index)), heatmap_plot.index)

    plt.xlabel("Gene")
    plt.ylabel("Condition")
    plt.title(f"{name1} vs {name2} (Model-selected genes)")

    if show_significance or show_direction:
        plt.suptitle(
            "↑F: higher in formate, ↑P: higher in phosphate; · FDR<0.1, * FDR<0.05, ** FDR<0.01, *** FDR<0.001",
            y=1.04,
            fontsize=9
        )

    plt.tight_layout()

    if outpath is None:
        outp = Path("analysis_outputs") / "heatmaps" / f"simple_heatmap_{name1}_vs_{name2}.png"
    else:
        outp = Path(outpath)

    outp.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outp, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

def splot_formate_vs_phosphate_simple_heatmap(
    df1_bio,
    df2_bio,
    model_results,
    name1="formate",
    name2="phosphate",
    outpath=None,
    top_n=30,
    gene_col=GENE_COL,
    center_by_gene=True,
    show=False
):
    """
    2-row heatmap:
    Rows = [formate, phosphate]
    Columns = genes (top from model)
    Values = mean log2 expression per dataset
    """

    if model_results.empty:
        print("Skipping heatmap: model_results empty.")
        return

    # -------------------------
    # Select top genes from model
    # -------------------------
    top_genes = (
        model_results
        .assign(abs_effect=model_results["dataset_effect_log2FC"].abs())
        .sort_values(["fdr", "abs_effect"], ascending=[True, False])
        [gene_col]
        .head(top_n)
        .tolist()
    )

    # -------------------------
    # Compute mean expression per gene per dataset
    # -------------------------
    mean1 = (
        df1_bio[df1_bio[gene_col].isin(top_genes)]
        .groupby(gene_col)["log2_ratio"]
        .mean()
        .rename(name1)
    )

    mean2 = (
        df2_bio[df2_bio[gene_col].isin(top_genes)]
        .groupby(gene_col)["log2_ratio"]
        .mean()
        .rename(name2)
    )

    heatmap_data = pd.concat([mean1, mean2], axis=1).T

    if heatmap_data.empty:
        print("Skipping heatmap: no overlapping genes.")
        return

    # keep gene order from model ranking
    heatmap_data = heatmap_data[[g for g in top_genes if g in heatmap_data.columns]]

    # -------------------------
    # Optional centering
    # -------------------------
    if center_by_gene:
        heatmap_plot = heatmap_data - heatmap_data.mean(axis=0)
        cbar_label = "Centered log2 expression"
    else:
        heatmap_plot = heatmap_data
        cbar_label = "log2 expression"

    # -------------------------
    # Plot
    # -------------------------
    plt.figure(figsize=(max(10, len(heatmap_plot.columns) * 0.4), 3))
    cmap = DIV_CMAP
    try:
        cmap.set_bad("black")
    except Exception:
        pass
    masked = np.ma.masked_invalid(heatmap_plot.values)
    if center_by_gene:
        max_abs = np.nanmax(np.abs(heatmap_plot.values)) if heatmap_plot.size else None
        if max_abs is None:
            im = plt.imshow(masked, aspect="auto", cmap=cmap)
        else:
            im = plt.imshow(masked, aspect="auto", cmap=cmap, vmin=-max_abs, vmax=max_abs)
    else:
        im = plt.imshow(masked, aspect="auto", cmap=cmap)
    plt.colorbar(im, label=cbar_label)

    plt.xticks(range(len(heatmap_plot.columns)), heatmap_plot.columns, rotation=90)
    plt.yticks(range(len(heatmap_plot.index)), heatmap_plot.index)

    plt.xlabel("Gene")
    plt.ylabel("Condition")
    plt.title(f"{name1} vs {name2} (Model-selected genes)")

    plt.tight_layout()

    # default to heatmaps subfolder
    if outpath is None:
        outp = Path("analysis_outputs") / "heatmaps" / f"simple_heatmap_{name1}_vs_{name2}.png"
    else:
        outp = Path(outpath)
    outp.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outp, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()
