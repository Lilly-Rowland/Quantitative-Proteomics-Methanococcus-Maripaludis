import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels import api as sm
import statsmodels.formula.api as smf
from utils import GENE_COL, add_fdr


def compare_growth_rates(df_bio, group1, group2):
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


def compare_datasets_full(df1_bio, df2_bio, name1="dataset1", name2="dataset2"):
    genes = sorted(set(df1_bio[GENE_COL]).intersection(set(df2_bio[GENE_COL])))

    rows = []
    for gene in genes:
        x = df1_bio.loc[df1_bio[GENE_COL] == gene, "log2_ratio"].dropna()
        y = df2_bio.loc[df2_bio[GENE_COL] == gene, "log2_ratio"].dropna()

        if len(x) == 0 or len(y) == 0:
            continue

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


def compare_datasets_growth_controlled_matched(df1_bio, df2_bio, name1="dataset1", name2="dataset2"):
    common_growth_rates = sorted(set(df1_bio["growth_rate"]).intersection(set(df2_bio["growth_rate"])))
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


def compare_datasets_growth_controlled_model(df1_bio, df2_bio, name1="dataset1", name2="dataset2"):
    x1 = df1_bio.copy()
    x2 = df2_bio.copy()
    x1["dataset"] = name1
    x2["dataset"] = name2

    combined = pd.concat([x1, x2], ignore_index=True)

    common_growth_rates = sorted(
        set(x1["growth_rate"]).intersection(set(x2["growth_rate"]))
    )
    combined = combined[combined["growth_rate"].isin(common_growth_rates)].copy()

    genes = sorted(combined[GENE_COL].unique())

    rows = []
    for gene in genes:
        sub = combined[combined[GENE_COL] == gene].copy()
        if sub["dataset"].nunique() < 2 or sub["growth_rate"].nunique() < 2 or len(sub) < 4:
            continue

        try:
            model = smf.ols(
                'log2_ratio ~ C(dataset, Treatment(reference="%s")) + C(growth_rate)' % name2,
                data=sub
            ).fit()

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
            continue

    results = pd.DataFrame(rows)
    if results.empty:
        return results

    results = add_fdr(results, p_col="p_value", out_col="fdr")
    return results.sort_values(["fdr", "p_value"], na_position="last")
