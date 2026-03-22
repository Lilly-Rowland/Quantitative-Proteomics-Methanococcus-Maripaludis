import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
from pathlib import Path


# =========================
# Helpers
# =========================

GENE_COL = "Gene names (ordered locus )"


def load_and_prepare(csv_path, dataset_name):
    """
    Load a cleaned CSV and prepare it for downstream analysis.
    Assumes columns include:
      - Gene names (ordered locus )
      - growth_rate
      - bio_rep
      - tech_rep
      - value
    """
    df = pd.read_csv(csv_path).copy()
    df["dataset"] = dataset_name

    # keep only positive numeric values for log2
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[df["value"].notna()].copy()
    df = df[df["value"] > 0].copy()

    df["growth_rate"] = df["growth_rate"].astype(str).str.lower().str.strip()
    df["log2_ratio"] = np.log2(df["value"])

    return df


def average_tech_reps(df):
    """
    Collapse technical replicates so each row becomes one biological replicate.
    If tech_rep is all missing, this still works fine.
    """
    group_cols = [GENE_COL, "growth_rate", "bio_rep"]
    df_bio = (
        df.groupby(group_cols, as_index=False)["log2_ratio"]
        .mean()
    )
    return df_bio


def volcano_plot(results, title, outpath=None):
    """
    Create a volcano plot from a results dataframe containing:
      - log2FC
      - p_value
    """
    plot_df = results.copy()
    plot_df["neglog10_p"] = -np.log10(plot_df["p_value"])

    plt.figure(figsize=(7, 5))
    plt.scatter(
        plot_df["log2FC"],
        plot_df["neglog10_p"],
        s=12,
        alpha=0.7
    )
    plt.axvline(0, linewidth=1)
    plt.xlabel("log2 Fold Change")
    plt.ylabel("-log10(p-value)")
    plt.title(title)
    plt.tight_layout()

    if outpath is not None:
        plt.savefig(outpath, dpi=300)
    plt.show()


# =========================
# 1) Within-dataset analysis
# =========================

def compare_growth_rates(df_bio, group1, group2):
    """
    Differential expression within a single dataset:
    compares growth_rate group1 vs group2 for each gene.
    """
    a = df_bio[df_bio["growth_rate"] == group1].copy()
    b = df_bio[df_bio["growth_rate"] == group2].copy()

    genes = sorted(set(a[GENE_COL]).intersection(set(b[GENE_COL])))

    rows = []
    for gene in genes:
        a_vals = a.loc[a[GENE_COL] == gene, "log2_ratio"].dropna()
        b_vals = b.loc[b[GENE_COL] == gene, "log2_ratio"].dropna()

        if len(a_vals) == 0 or len(b_vals) == 0:
            continue

        log2fc = a_vals.mean() - b_vals.mean()

        if len(a_vals) > 1 and len(b_vals) > 1:
            p_value = ttest_ind(a_vals, b_vals, equal_var=False).pvalue
        else:
            p_value = np.nan

        rows.append({
            GENE_COL: gene,
            "group1": group1,
            "group2": group2,
            "mean_group1": a_vals.mean(),
            "mean_group2": b_vals.mean(),
            "log2FC": log2fc,
            "n_group1": len(a_vals),
            "n_group2": len(b_vals),
            "p_value": p_value,
        })

    results = pd.DataFrame(rows)
    return results.sort_values("p_value", na_position="last")


# =========================
# 2) Dataset-vs-dataset comparison (full)
# =========================

def compare_datasets_full(df1_bio, df2_bio, name1="dataset1", name2="dataset2"):
    """
    Compare dataset 1 vs dataset 2 for each gene, ignoring growth rate.
    This answers:
      'Are the overall expression distributions different between datasets?'
    """
    genes = sorted(set(df1_bio[GENE_COL]).intersection(set(df2_bio[GENE_COL])))

    rows = []
    for gene in genes:
        x = df1_bio.loc[df1_bio[GENE_COL] == gene, "log2_ratio"].dropna()
        y = df2_bio.loc[df2_bio[GENE_COL] == gene, "log2_ratio"].dropna()

        if len(x) == 0 or len(y) == 0:
            continue

        log2fc = x.mean() - y.mean()

        if len(x) > 1 and len(y) > 1:
            p_value = ttest_ind(x, y, equal_var=False).pvalue
        else:
            p_value = np.nan

        rows.append({
            GENE_COL: gene,
            "comparison": f"{name1}_vs_{name2}",
            "mean_" + name1: x.mean(),
            "mean_" + name2: y.mean(),
            "log2FC": log2fc,
            "n_" + name1: len(x),
            "n_" + name2: len(y),
            "p_value": p_value,
        })

    results = pd.DataFrame(rows)
    return results.sort_values("p_value", na_position="last")


# =========================
# 3) Dataset-vs-dataset comparison (growth-rate controlled)
# =========================

def compare_datasets_growth_controlled(df1_bio, df2_bio, name1="dataset1", name2="dataset2"):
    """
    Compare dataset 1 vs dataset 2 while controlling for growth rate.

    Strategy:
      - only use growth rates present in both datasets
      - for each gene and each matched growth rate:
            compare dataset1 vs dataset2 within that growth rate
      - summarize across growth rates by averaging the within-growth-rate effects

    Output:
      - one row per gene with an average matched log2FC
      - optional pooled p-value summary using mean of per-rate p-values is NOT statistically ideal,
        so we keep per-rate details too
    """
    common_growth_rates = sorted(
        set(df1_bio["growth_rate"]).intersection(set(df2_bio["growth_rate"]))
    )
    common_genes = sorted(
        set(df1_bio[GENE_COL]).intersection(set(df2_bio[GENE_COL]))
    )

    per_rate_rows = []

    for gr in common_growth_rates:
        d1_gr = df1_bio[df1_bio["growth_rate"] == gr]
        d2_gr = df2_bio[df2_bio["growth_rate"] == gr]

        genes_here = sorted(set(d1_gr[GENE_COL]).intersection(set(d2_gr[GENE_COL])))

        for gene in genes_here:
            x = d1_gr.loc[d1_gr[GENE_COL] == gene, "log2_ratio"].dropna()
            y = d2_gr.loc[d2_gr[GENE_COL] == gene, "log2_ratio"].dropna()

            if len(x) == 0 or len(y) == 0:
                continue

            log2fc = x.mean() - y.mean()

            if len(x) > 1 and len(y) > 1:
                p_value = ttest_ind(x, y, equal_var=False).pvalue
            else:
                p_value = np.nan

            per_rate_rows.append({
                GENE_COL: gene,
                "growth_rate": gr,
                "mean_" + name1: x.mean(),
                "mean_" + name2: y.mean(),
                "log2FC_within_growth_rate": log2fc,
                "n_" + name1: len(x),
                "n_" + name2: len(y),
                "p_value_within_growth_rate": p_value,
            })

    per_rate_df = pd.DataFrame(per_rate_rows)

    if per_rate_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    summary = (
        per_rate_df.groupby(GENE_COL, as_index=False)
        .agg(
            matched_growth_rates=("growth_rate", "nunique"),
            avg_log2FC=("log2FC_within_growth_rate", "mean"),
            median_log2FC=("log2FC_within_growth_rate", "median"),
            mean_p_value=("p_value_within_growth_rate", "mean"),
        )
        .sort_values("mean_p_value", na_position="last")
    )

    return summary, per_rate_df


# =========================
# 4) Main analysis
# =========================

def main():
    # -------- input files --------
    file1 = "data/formate_dataset_cleaned.csv"
    file2 = "data/phosphate_dataset_cleaned.csv"

    name1 = "formate"
    name2 = "newdata"

    outdir = Path("analysis_outputs")
    outdir.mkdir(exist_ok=True)

    # -------- load --------
    df1 = load_and_prepare(file1, name1)
    df2 = load_and_prepare(file2, name2)

    # -------- collapse tech reps --------
    df1_bio = average_tech_reps(df1)
    df2_bio = average_tech_reps(df2)

    # save collapsed versions
    df1_bio.to_csv(outdir / f"{name1}_bio_level.csv", index=False)
    df2_bio.to_csv(outdir / f"{name2}_bio_level.csv", index=False)

    # -----------------------------------
    # A) Individual dataset analyses
    # -----------------------------------
    # Choose whatever growth-rate comparisons make sense for each dataset.
    # Example comparisons:
    comparisons_1 = [("xs", "s")]
    comparisons_2 = [("xs", "i")]   # change this if needed

    for g1, g2 in comparisons_1:
        res = compare_growth_rates(df1_bio, g1, g2)
        res.to_csv(outdir / f"{name1}_{g1}_vs_{g2}.csv", index=False)
        print(f"\n{name1}: {g1} vs {g2}")
        print(res.head())

        if not res.empty:
            volcano_plot(
                res.dropna(subset=["p_value"]),
                title=f"Volcano Plot – {name1} {g1.upper()} vs {g2.upper()}",
                outpath=outdir / f"{name1}_{g1}_vs_{g2}_volcano.png"
            )

    for g1, g2 in comparisons_2:
        res = compare_growth_rates(df2_bio, g1, g2)
        res.to_csv(outdir / f"{name2}_{g1}_vs_{g2}.csv", index=False)
        print(f"\n{name2}: {g1} vs {g2}")
        print(res.head())

        if not res.empty:
            volcano_plot(
                res.dropna(subset=["p_value"]),
                title=f"Volcano Plot – {name2} {g1.upper()} vs {g2.upper()}",
                outpath=outdir / f"{name2}_{g1}_vs_{g2}_volcano.png"
            )

    # -----------------------------------
    # B) Dataset-to-dataset full comparison
    # -----------------------------------
    full_compare = compare_datasets_full(df1_bio, df2_bio, name1=name1, name2=name2)
    full_compare.to_csv(outdir / f"{name1}_vs_{name2}_full.csv", index=False)

    print(f"\nFull dataset comparison: {name1} vs {name2}")
    print(full_compare.head())

    if not full_compare.empty:
        volcano_plot(
            full_compare.dropna(subset=["p_value"]),
            title=f"Volcano Plot – {name1} vs {name2} (full)",
            outpath=outdir / f"{name1}_vs_{name2}_full_volcano.png"
        )

    # -----------------------------------
    # C) Dataset-to-dataset growth-controlled comparison
    # -----------------------------------
    controlled_summary, controlled_per_rate = compare_datasets_growth_controlled(
        df1_bio, df2_bio, name1=name1, name2=name2
    )

    controlled_summary.to_csv(
        outdir / f"{name1}_vs_{name2}_growth_controlled_summary.csv",
        index=False
    )
    controlled_per_rate.to_csv(
        outdir / f"{name1}_vs_{name2}_growth_controlled_per_rate.csv",
        index=False
    )

    print(f"\nGrowth-controlled dataset comparison: {name1} vs {name2}")
    print(controlled_summary.head())

    # Plot growth-controlled summary using avg_log2FC vs mean_p_value
    if not controlled_summary.empty:
        plot_df = controlled_summary.dropna(subset=["mean_p_value"]).copy()
        plot_df["neglog10_p"] = -np.log10(plot_df["mean_p_value"])

        plt.figure(figsize=(7, 5))
        plt.scatter(plot_df["avg_log2FC"], plot_df["neglog10_p"], s=12, alpha=0.7)
        plt.axvline(0, linewidth=1)
        plt.xlabel("Average matched log2 Fold Change")
        plt.ylabel("-log10(mean within-growth-rate p-value)")
        plt.title(f"Volcano Plot – {name1} vs {name2} (growth-rate controlled)")
        plt.tight_layout()
        plt.savefig(outdir / f"{name1}_vs_{name2}_growth_controlled_volcano.png", dpi=300)
        plt.show()


if __name__ == "__main__":
    main()