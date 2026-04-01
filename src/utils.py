import numpy as np
import pandas as pd

GENE_COL = "Gene names (ordered locus )"


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
    from statsmodels.stats.multitest import multipletests

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
