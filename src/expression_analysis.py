import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf


GENE_COL = "Gene names (ordered locus )"


# =========================================================
# 1. Loading / preprocessing
# =========================================================

def load_and_prepare(csv_path, dataset_name):
    """
    Load cleaned csv and set up columns used later.
    """
    df = pd.read_csv(csv_path).copy()
    df["dataset"] = dataset_name

    # force values numeric and remove anything that cannot be log-transformed
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[df["value"].notna()].copy()
    df = df[df["value"] > 0].copy()

    # standardize growth rate labels and compute log2 ratio
    df["growth_rate"] = df["growth_rate"].astype(str).str.strip().str.lower()
    df["bio_rep"] = pd.to_numeric(df["bio_rep"], errors="coerce")
    df["log2_ratio"] = np.log2(df["value"])

    return df


def average_tech_reps(df):
    """
    Average technical replicates so each row is one biological replicate.
    For the formate dataset this collapses technical replicates;
    for phosphate it should just leave one value per bio replicate if no tech reps exist.
    """
    return (
        df.groupby([GENE_COL, "growth_rate", "bio_rep"], as_index=False)["log2_ratio"]
        .mean()
    )


def add_fdr(df, p_col="p_value", out_col="fdr"):
    """
    Add Benjamini-Hochberg adjusted p-values.
    """
    df = df.copy()
    df[out_col] = np.nan

    mask = df[p_col].notna()
    if mask.sum() > 0:
        df.loc[mask, out_col] = multipletests(df.loc[mask, p_col], method="fdr_bh")[1]

    return df


def safe_neglog10(series):
    """
    Compute -log10 for plotting and avoid issues with exact zeros.
    """
    s = series.copy()
    s = s.replace(0, np.nextafter(0, 1))
    return -np.log10(s)


def volcano_plot(
    results,
    x_col,
    p_col,
    title,
    outpath=None,
    fc_thresh=1.0,
    sig_thresh=0.05,
    label_top_n=0,
    gene_col=GENE_COL
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
    plt.scatter(plot_df.loc[~sig_mask, x_col], plot_df.loc[~sig_mask, "neglog10_p"], s=12, alpha=0.6)
    plt.scatter(plot_df.loc[sig_mask, x_col], plot_df.loc[sig_mask, "neglog10_p"], s=14, alpha=0.8)

    # reference lines
    plt.axvline(0, linewidth=1)
    plt.axvline(fc_thresh, linestyle="--", linewidth=1)
    plt.axvline(-fc_thresh, linestyle="--", linewidth=1)
    plt.axhline(-np.log10(sig_thresh), linestyle="--", linewidth=1)

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

    plt.show()


# =========================================================
# 2. Within-dataset DE: growth-rate comparison
# =========================================================

def compare_growth_rates(df_bio, group1, group2):
    """
    Compare two growth rates within one dataset.
    """
    a = df_bio[df_bio["growth_rate"] == group1].copy()
    b = df_bio[df_bio["growth_rate"] == group2].copy()

    # only test genes observed in both groups
    genes = sorted(set(a[GENE_COL]).intersection(set(b[GENE_COL])))

    rows = []
    for gene in genes:
        a_vals = a.loc[a[GENE_COL] == gene, "log2_ratio"].dropna()
        b_vals = b.loc[b[GENE_COL] == gene, "log2_ratio"].dropna()

        if len(a_vals) == 0 or len(b_vals) == 0:
            continue

        # log2FC is mean(group1) - mean(group2)
        log2fc = a_vals.mean() - b_vals.mean()

        # Welch t-test if both groups have >1 observation
        p_value = np.nan
        if len(a_vals) > 1 and len(b_vals) > 1:
            p_value = ttest_ind(a_vals, b_vals, equal_var=False).pvalue

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
    if results.empty:
        return results

    results = add_fdr(results, p_col="p_value", out_col="fdr")
    return results.sort_values(["fdr", "p_value"], na_position="last")


# =========================================================
# 3. Dataset-vs-dataset full comparison
# =========================================================

def compare_datasets_full(df1_bio, df2_bio, name1="dataset1", name2="dataset2"):
    """
    Compare datasets directly, ignoring growth rate.
    """
    genes = sorted(set(df1_bio[GENE_COL]).intersection(set(df2_bio[GENE_COL])))

    rows = []
    for gene in genes:
        x = df1_bio.loc[df1_bio[GENE_COL] == gene, "log2_ratio"].dropna()
        y = df2_bio.loc[df2_bio[GENE_COL] == gene, "log2_ratio"].dropna()

        if len(x) == 0 or len(y) == 0:
            continue

        # overall mean difference between datasets
        log2fc = x.mean() - y.mean()

        p_value = np.nan
        if len(x) > 1 and len(y) > 1:
            p_value = ttest_ind(x, y, equal_var=False).pvalue

        rows.append({
            GENE_COL: gene,
            "comparison": f"{name1}_vs_{name2}",
            f"mean_{name1}": x.mean(),
            f"mean_{name2}": y.mean(),
            "log2FC": log2fc,
            f"n_{name1}": len(x),
            f"n_{name2}": len(y),
            "p_value": p_value,
        })

    results = pd.DataFrame(rows)
    if results.empty:
        return results

    results = add_fdr(results, p_col="p_value", out_col="fdr")
    return results.sort_values(["fdr", "p_value"], na_position="last")


# =========================================================
# 4. Dataset-vs-dataset matched by growth rate!!!
# =========================================================

def compare_datasets_growth_controlled_matched(df1_bio, df2_bio, name1="dataset1", name2="dataset2"):
    """
    Compare datasets while matching within shared growth-rate groups.

    Returns:
      - summary table across matched growth rates
      - per-growth-rate table
    """
    # only growth rates present in both datasets are used
    common_growth_rates = sorted(set(df1_bio["growth_rate"]).intersection(set(df2_bio["growth_rate"])))
    per_rate_rows = []

    for gr in common_growth_rates:
        d1_gr = df1_bio[df1_bio["growth_rate"] == gr]
        d2_gr = df2_bio[df2_bio["growth_rate"] == gr]

        # compare only genes seen in both datasets for this growth rate
        genes_here = sorted(set(d1_gr[GENE_COL]).intersection(set(d2_gr[GENE_COL])))

        for gene in genes_here:
            x = d1_gr.loc[d1_gr[GENE_COL] == gene, "log2_ratio"].dropna()
            y = d2_gr.loc[d2_gr[GENE_COL] == gene, "log2_ratio"].dropna()

            if len(x) == 0 or len(y) == 0:
                continue

            # dataset difference within one growth-rate stratum
            log2fc = x.mean() - y.mean()

            p_value = np.nan
            if len(x) > 1 and len(y) > 1:
                p_value = ttest_ind(x, y, equal_var=False).pvalue

            per_rate_rows.append({
                GENE_COL: gene,
                "growth_rate": gr,
                f"mean_{name1}": x.mean(),
                f"mean_{name2}": y.mean(),
                "log2FC_within_growth_rate": log2fc,
                f"n_{name1}": len(x),
                f"n_{name2}": len(y),
                "p_value_within_growth_rate": p_value,
            })

    per_rate_df = pd.DataFrame(per_rate_rows)
    if per_rate_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    per_rate_df = add_fdr(per_rate_df, p_col="p_value_within_growth_rate", out_col="fdr_within_growth_rate")

    # summarize matched results across growth rates for each gene
    summary = (
        per_rate_df.groupby(GENE_COL, as_index=False)
        .agg(
            matched_growth_rates=("growth_rate", "nunique"),
            avg_log2FC=("log2FC_within_growth_rate", "mean"),
            median_log2FC=("log2FC_within_growth_rate", "median"),
            mean_p_value=("p_value_within_growth_rate", "mean"),
            min_p_value=("p_value_within_growth_rate", "min"),
            mean_fdr=("fdr_within_growth_rate", "mean"),
        )
    )

    summary = add_fdr(summary, p_col="mean_p_value", out_col="summary_fdr")
    summary = summary.sort_values(["summary_fdr", "mean_p_value"], na_position="last")

    return summary, per_rate_df


# =========================================================
# 5. Model-based growth-controlled comparison
# =========================================================

def compare_datasets_growth_controlled_model(df1_bio, df2_bio, name1="dataset1", name2="dataset2"):
    """
    Growth-controlled comparison using a per-gene linear model:

        log2_ratio ~ C(dataset) + C(growth_rate)

    The dataset coefficient estimates the difference between datasets
    after accounting for growth rate.
    """
    x1 = df1_bio.copy()
    x2 = df2_bio.copy()
    x1["dataset"] = name1
    x2["dataset"] = name2

    combined = pd.concat([x1, x2], ignore_index=True)

    # restrict to growth rates shared by both datasets
    common_growth_rates = sorted(
        set(x1["growth_rate"]).intersection(set(x2["growth_rate"]))
    )
    combined = combined[combined["growth_rate"].isin(common_growth_rates)].copy()

    genes = sorted(combined[GENE_COL].unique())

    rows = []
    for gene in genes:
        sub = combined[combined[GENE_COL] == gene].copy()

        # need both datasets, at least two growth rates, and enough rows to fit model
        if sub["dataset"].nunique() < 2 or sub["growth_rate"].nunique() < 2 or len(sub) < 4:
            continue

        try:
            model = smf.ols(
                'log2_ratio ~ C(dataset, Treatment(reference="%s")) + C(growth_rate)' % name2,
                data=sub
            ).fit()

            # this is the coefficient corresponding to dataset effect
            dataset_term = [t for t in model.params.index if t.startswith("C(dataset")]
            if len(dataset_term) != 1:
                continue

            term = dataset_term[0]

            rows.append({
                GENE_COL: gene,
                "dataset_effect_log2FC": model.params.get(term, np.nan),
                "p_value": model.pvalues.get(term, np.nan),
                "n_obs": len(sub),
                "n_growth_rates": sub["growth_rate"].nunique(),
                "mean_" + name1: sub.loc[sub["dataset"] == name1, "log2_ratio"].mean(),
                "mean_" + name2: sub.loc[sub["dataset"] == name2, "log2_ratio"].mean(),
            })

        except Exception:
            # skip genes where model fit fails
            continue

    results = pd.DataFrame(rows)
    if results.empty:
        return results

    results = add_fdr(results, p_col="p_value", out_col="fdr")
    return results.sort_values(["fdr", "p_value"], na_position="last")


# =========================================================
# 6. Shared significant genes
# =========================================================

def shared_significant_hits(df_a, df_b, label_a, label_b, fc_col="log2FC", fdr_col="fdr",
                            fdr_thresh=0.05, fc_thresh=1.0):
    """
    Find genes significant in both result tables.
    """
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

    # check whether the direction of change matches between result tables
    merged["same_direction"] = (
        np.sign(merged[f"{fc_col}_{label_a}"]) == np.sign(merged[f"{fc_col}_{label_b}"])
    )

    return merged.sort_values([f"{fdr_col}_{label_a}", f"{fdr_col}_{label_b}"])


# =========================================================
# 7. Top-hit tables
# =========================================================

def save_top_hits(results, outpath, effect_col, p_col="fdr", n=50):
    """
    Save top genes ranked by adjusted p-value and effect size.
    """
    if results.empty:
        return

    cols = [c for c in results.columns if c in [GENE_COL, effect_col, p_col, "p_value", "group1", "group2"]]
    extra_cols = [c for c in results.columns if c not in cols]
    ordered_cols = cols + extra_cols

    top = results.sort_values([p_col, effect_col], ascending=[True, False], na_position="last").head(n)
    top[ordered_cols].to_csv(outpath, index=False)


# =========================================================
# 8. Main
# =========================================================

def main():
    file1 = "data/formate_dataset_cleaned.csv"
    file2 = "data/phosphate_dataset_cleaned.csv"

    name1 = "formate"
    name2 = "phosphate"

    outdir = Path("analysis_outputs")
    outdir.mkdir(exist_ok=True)

    # -------------------------
    # Load + preprocess
    # -------------------------
    df1 = load_and_prepare(file1, name1)
    df2 = load_and_prepare(file2, name2)

    # collapse to one value per gene / growth rate / biological replicate
    df1_bio = average_tech_reps(df1)
    df2_bio = average_tech_reps(df2)

    df1_bio.to_csv(outdir / f"{name1}_bio_level.csv", index=False)
    df2_bio.to_csv(outdir / f"{name2}_bio_level.csv", index=False)

    # -------------------------
    # Inspect available growth rates
    # -------------------------
    rates1 = sorted(df1_bio["growth_rate"].dropna().unique())
    rates2 = sorted(df2_bio["growth_rate"].dropna().unique())
    common_rates = sorted(set(rates1).intersection(set(rates2)))

    print(f"{name1} growth rates: {rates1}")
    print(f"{name2} growth rates: {rates2}")
    print(f"Shared growth rates: {common_rates}")

    # -------------------------
    # Individual dataset analyses
    # -------------------------
    # set these to the comparisons you want within each dataset
    comparisons_1 = [("xs", "s")]
    comparisons_2 = [("xs", "i")]

    individual_results = {}

    for g1, g2 in comparisons_1:
        res = compare_growth_rates(df1_bio, g1, g2)
        individual_results[f"{name1}_{g1}_vs_{g2}"] = res

        out_csv = outdir / f"{name1}_{g1}_vs_{g2}.csv"
        res.to_csv(out_csv, index=False)
        save_top_hits(res, outdir / f"{name1}_{g1}_vs_{g2}_top_hits.csv", effect_col="log2FC", p_col="fdr", n=50)

        print(f"\n{name1}: {g1} vs {g2}")
        print(res.head())

        if not res.empty:
            volcano_plot(
                res,
                x_col="log2FC",
                p_col="fdr",
                title=f"Volcano Plot – {name1} {g1.upper()} vs {g2.upper()}",
                outpath=outdir / f"{name1}_{g1}_vs_{g2}_volcano.png",
                fc_thresh=1.0,
                sig_thresh=0.05,
                label_top_n=10
            )

    for g1, g2 in comparisons_2:
        res = compare_growth_rates(df2_bio, g1, g2)
        individual_results[f"{name2}_{g1}_vs_{g2}"] = res

        out_csv = outdir / f"{name2}_{g1}_vs_{g2}.csv"
        res.to_csv(out_csv, index=False)
        save_top_hits(res, outdir / f"{name2}_{g1}_vs_{g2}_top_hits.csv", effect_col="log2FC", p_col="fdr", n=50)

        print(f"\n{name2}: {g1} vs {g2}")
        print(res.head())

        if not res.empty:
            volcano_plot(
                res,
                x_col="log2FC",
                p_col="fdr",
                title=f"Volcano Plot – {name2} {g1.upper()} vs {g2.upper()}",
                outpath=outdir / f"{name2}_{g1}_vs_{g2}_volcano.png",
                fc_thresh=1.0,
                sig_thresh=0.05,
                label_top_n=10
            )

    # -------------------------
    # Compare significant hits between the two individual analyses
    # -------------------------
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
            fc_thresh=1.0
        )
        overlap.to_csv(outdir / "shared_significant_genes_between_individual_analyses.csv", index=False)
        print("\nShared significant genes between individual analyses:")
        print(overlap.head())

    # -------------------------
    # Full dataset comparison
    # -------------------------
    full_compare = compare_datasets_full(df1_bio, df2_bio, name1=name1, name2=name2)
    full_compare.to_csv(outdir / f"{name1}_vs_{name2}_full.csv", index=False)
    save_top_hits(full_compare, outdir / f"{name1}_vs_{name2}_full_top_hits.csv", effect_col="log2FC", p_col="fdr", n=50)

    print(f"\nFull comparison: {name1} vs {name2}")
    print(full_compare.head())

    if not full_compare.empty:
        volcano_plot(
            full_compare,
            x_col="log2FC",
            p_col="fdr",
            title=f"Volcano Plot – {name1} vs {name2} (full)",
            outpath=outdir / f"{name1}_vs_{name2}_full_volcano.png",
            fc_thresh=1.0,
            sig_thresh=0.05,
            label_top_n=10
        )

    # -------------------------
    # Matched growth-controlled comparison
    # -------------------------
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
        save_top_hits(
            controlled_summary.rename(columns={"avg_log2FC": "log2FC", "summary_fdr": "fdr"}),
            outdir / f"{name1}_vs_{name2}_growth_controlled_matched_top_hits.csv",
            effect_col="log2FC",
            p_col="fdr",
            n=50
        )

    print(f"\nGrowth-controlled matched comparison: {name1} vs {name2}")
    print(controlled_summary.head())

    if not controlled_summary.empty:
        volcano_plot(
            controlled_summary.rename(columns={"avg_log2FC": "log2FC", "summary_fdr": "fdr"}),
            x_col="log2FC",
            p_col="fdr",
            title=f"Volcano Plot – {name1} vs {name2} (matched growth-rate controlled)",
            outpath=outdir / f"{name1}_vs_{name2}_growth_controlled_matched_volcano.png",
            fc_thresh=1.0,
            sig_thresh=0.05,
            label_top_n=10
        )

    # -------------------------
    # Model-based growth-controlled comparison
    # -------------------------
    model_compare = compare_datasets_growth_controlled_model(
        df1_bio, df2_bio, name1=name1, name2=name2
    )
    model_compare.to_csv(
        outdir / f"{name1}_vs_{name2}_growth_controlled_model.csv",
        index=False
    )
    save_top_hits(
        model_compare.rename(columns={"dataset_effect_log2FC": "log2FC"}),
        outdir / f"{name1}_vs_{name2}_growth_controlled_model_top_hits.csv",
        effect_col="log2FC",
        p_col="fdr",
        n=50
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
            fc_thresh=1.0,
            sig_thresh=0.05,
            label_top_n=10
        )

    print(f"\nDone. Outputs saved in: {outdir.resolve()}")


if __name__ == "__main__":
    main()