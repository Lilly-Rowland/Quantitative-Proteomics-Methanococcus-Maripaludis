import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Iterable, Union
import seaborn as sns

# plotting theme
sns.set(style="whitegrid")
PATHWAY_POS = "#E64B35"
PATHWAY_NEG = "#4A90E2"
PATHWAY_NEUTRAL = "#9EA7AA"


# PROBLEM HOW TO GET THE METHANOGENSI MM GENE NAMES???
METHANOGENESIS_GENES = [
    "mcrA", "mcrB", "mcrG",
    "fwdA", "fwdB",
    "mtd", "mer"
]

# METHANOGENESIS_GENES = [
#     "MMP1559", "MMP1555", "MMP1096",
#     "fwdA", "fwdB",
#     "mtd", "mer"
# ]


def summarize_pathway_across_datasets(df1_bio: pd.DataFrame, df2_bio: pd.DataFrame, genes: Iterable[str], name1: str, name2: str, outdir: Union[str, Path] = "analysis_outputs") -> pd.DataFrame:
    """Create a small summary table for the given genes comparing two datasets.

    Returns a DataFrame with columns: gene, mean_<name1>, mean_<name2>, log2FC (<name1> - <name2>), n_<name1>, n_<name2>
    and saves CSV to outdir/pathway_summary_<name1>_vs_<name2>.csv
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for g in genes:
        x = df1_bio.loc[df1_bio[df1_bio.columns[0]] == g, "log2_ratio"] if False else df1_bio.loc[df1_bio[df1_bio.columns[0]] == g, "log2_ratio"]
        # the gene column name is expected to be the first column in these averaged bio tables (GENE_COL)
        # but to be robust, try common names
        if x.empty and "Gene names (ordered locus )" in df1_bio.columns:
            x = df1_bio.loc[df1_bio["Gene names (ordered locus )"] == g, "log2_ratio"]
        if x.empty and "gene" in df1_bio.columns:
            x = df1_bio.loc[df1_bio["gene"] == g, "log2_ratio"]

        y = df2_bio.loc[df2_bio[df2_bio.columns[0]] == g, "log2_ratio"] if False else df2_bio.loc[df2_bio[df2_bio.columns[0]] == g, "log2_ratio"]
        if y.empty and "Gene names (ordered locus )" in df2_bio.columns:
            y = df2_bio.loc[df2_bio["Gene names (ordered locus )"] == g, "log2_ratio"]
        if y.empty and "gene" in df2_bio.columns:
            y = df2_bio.loc[df2_bio["gene"] == g, "log2_ratio"]

        rows.append({
            "gene": g,
            f"mean_{name1}": x.mean() if len(x) > 0 else np.nan,
            f"mean_{name2}": y.mean() if len(y) > 0 else np.nan,
            f"n_{name1}": int(len(x)),
            f"n_{name2}": int(len(y)),
            "log2FC": (x.mean() - y.mean()) if (len(x) > 0 and len(y) > 0) else np.nan,
        })

    summary = pd.DataFrame(rows)
    summary = summary.sort_values("gene")

    out_csv = outdir / f"pathway_summary_{name1}_vs_{name2}.csv"
    summary.to_csv(out_csv, index=False)

    return summary


def plot_pathway_log2fc(summary_df: pd.DataFrame, outpath=None, show=False):
    """Barplot of log2FC for pathway genes (summary_df must contain 'gene' and 'log2FC')."""
    if summary_df.empty:
        print("No pathway summary to plot.")
        return

    df = summary_df.dropna(subset=["log2FC"]).copy()
    if df.empty:
        print("No log2FC values available for pathway plot.")
        return

    plt.figure(figsize=(8, max(2, len(df) * 0.5)))
    y = np.arange(len(df))
    colors = [PATHWAY_POS if v > 0 else PATHWAY_NEG if v < 0 else PATHWAY_NEUTRAL for v in df["log2FC"]]
    plt.barh(y, df["log2FC"], color=colors, edgecolor="#222222", linewidth=0.3)
    plt.yticks(y, df["gene"])
    plt.xlabel("log2FC (dataset1 - dataset2)")
    plt.title("Pathway gene log2 fold changes")
    plt.axvline(0, color=PATHWAY_NEUTRAL, linewidth=0.8)
    plt.tight_layout()

    if outpath is not None:
        plt.savefig(outpath, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_pathway_growth_trends(df1_bio: pd.DataFrame, df2_bio: pd.DataFrame, genes: Iterable[str], name1: str, name2: str, outpath=None, show=False):
    """Create small-figure panels: per-gene mean log2_ratio across growth rates for both datasets.

    Saves to outpath if provided.
    """
    # helper: compute mean per growth_rate per gene
    def mean_by_growth(df):
        # determine gene col
        gene_col = None
        for candidate in ["Gene names (ordered locus )", "gene"]:
            if candidate in df.columns:
                gene_col = candidate
                break
        if gene_col is None:
            gene_col = df.columns[0]

        return (
            df[df[gene_col].isin(genes)]
            .groupby([gene_col, "growth_rate"], as_index=False)["log2_ratio"].mean()
            .rename(columns={gene_col: "gene"})
        )

    m1 = mean_by_growth(df1_bio)
    m2 = mean_by_growth(df2_bio)

    # build x axis (sorted growth rates common to both for consistency)
    grs = sorted(set(m1["growth_rate"]).union(set(m2["growth_rate"])), key=lambda x: float(x) if str(x).replace('.', '', 1).isdigit() else x)

    genes_present = [g for g in genes if ((g in m1["gene"].values) or (g in m2["gene"].values))]
    if not genes_present:
        print("No pathway genes present in datasets for growth-trend plotting.")
        return

    n = len(genes_present)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.5 * nrows), squeeze=False)

    for i, gene in enumerate(genes_present):
        r = i // ncols
        c = i % ncols
        ax = axes[r][c]
        d1 = m1[m1["gene"] == gene].set_index("growth_rate")["log2_ratio"].reindex(grs)
        d2 = m2[m2["gene"] == gene].set_index("growth_rate")["log2_ratio"].reindex(grs)

    ax.plot(grs, d1.values, marker="o", label=name1, color=PATHWAY_POS)
    ax.plot(grs, d2.values, marker="s", label=name2, color=PATHWAY_NEG)
    ax.set_title(gene)
    ax.set_xlabel("growth_rate")
    ax.set_ylabel("mean log2_ratio")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize="small")

    # hide empty subplots
    for j in range(i + 1, nrows * ncols):
        r = j // ncols
        c = j % ncols
        axes[r][c].axis("off")

    plt.tight_layout()
    if outpath is not None:
        plt.savefig(outpath, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()
