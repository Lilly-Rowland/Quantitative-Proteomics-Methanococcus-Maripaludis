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



METHANOGENESIS_GENES = {
    "MMP1559": "mcrA", "MMP1555": "mcrB", "MMP1558": "mcrG",
    "MMP1248": "fwdA", "MMP1691": "fwdB", "MMP0372": "mtd",
    "MMP0058": "mer", "MMP0825": "hdrA", "MMP1297": "fdhB", 
    "MMP1298": "fdhA", "MMP1301": "fdhC", "MMP0138": "fdhA",
    "MMP0139": "fdhB", "MMP0508": "fmdE", "MMP0509": "fmdA",
    "MMP0510": "fmdC", "MMP0511": "fmdB", "MMP0512": "fmdB",
    "MMP0200": "fmdE"
}


def summarize_pathway_across_datasets(df1_bio: pd.DataFrame, df2_bio: pd.DataFrame, gene_dict=None, name1: str = "dataset1", name2: str = "dataset2", outdir: Union[str, Path] = "analysis_outputs") -> pd.DataFrame:
    """Create a small summary table for the given genes comparing two datasets.

    gene_dict: optional dict mapping locus -> symbol; if None the module-global
    `METHANOGENESIS_GENES` will be used when available.
    """
    if gene_dict is None and isinstance(METHANOGENESIS_GENES, dict):
        genes = list(METHANOGENESIS_GENES.keys())
    elif gene_dict is None:
        genes = list(METHANOGENESIS_GENES)
    elif isinstance(gene_dict, dict):
        genes = list(gene_dict.keys())
    else:
        # assume iterable of gene ids
        genes = list(gene_dict)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for g in genes:
        x = df1_bio.loc[df1_bio[df1_bio.columns[0]] == g, "log2_ratio"] if False else df1_bio.loc[df1_bio[df1_bio.columns[0]] == g, "log2_ratio"]
        # the gene column name is expected to be the first column in these averaged bio tables (GENE_COL)
        # but to be robust, try common names
        if x.empty and "Gene names (ordered locus )" in df1_bio.columns:
            x = df1_bio.loc[df1_bio["Gene names (ordered locus )"] == g, "log2_ratio"]
            print(x)
        if x.empty and "gene" in df1_bio.columns:
            x = df1_bio.loc[df1_bio["gene"] == g, "log2_ratio"]

        y = df2_bio.loc[df2_bio[df2_bio.columns[0]] == g, "log2_ratio"] if False else df2_bio.loc[df2_bio[df2_bio.columns[0]] == g, "log2_ratio"]
        print(y)
        if y.empty and "Gene names (ordered locus )" in df2_bio.columns:
            y = df2_bio.loc[df2_bio["Gene names (ordered locus )"] == g, "log2_ratio"]
        if y.empty and "gene" in df2_bio.columns:
            y = df2_bio.loc[df2_bio["gene"] == g, "log2_ratio"]
        print(y)
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
    # optionally use a mapping dict available in the module globals
    # name_map = None
    # if isinstance(METHANOGENESIS_GENES, dict):
    #     name_map = METHANOGENESIS_GENES
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
    # use module-level mapping dict to show 'LOCUS (symbol)' labels — mapping is assumed available
    labels = [f"{g} ({METHANOGENESIS_GENES[g]})" for g in df["gene"]]
    plt.yticks(y, labels)
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