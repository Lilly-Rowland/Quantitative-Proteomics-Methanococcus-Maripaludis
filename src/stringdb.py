"""Simple wrapper for calling the STRING-db enrichment API.

"""
from typing import Iterable, Optional
import io
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import textwrap
from pathlib import Path

try:
    import requests
except Exception:  # pragma: no cover - runtime dependency
    requests = None

def run_directional_string_enrichment(
    model_compare: pd.DataFrame,
    gene_col: str,
    effect_col: str = "dataset_effect_log2FC",
    fdr_col: str = "fdr",
    effect_threshold: float = 0.0,
    fdr_threshold: Optional[float] = None,
    top_n: Optional[int] = 100,
    species: Optional[int] = 267377,
    outdir: Optional[str] = None,
    prefix: str = "string_enrichment",
    caller_identity: str = "quantitative-proteomics"
):
    """
    Run STRING enrichment separately for genes up in formate and up in phosphate.

    Positive effect_col => up in formate
    Negative effect_col => up in phosphate
    """
    df = model_compare.copy()

    req = {gene_col, effect_col}
    if not req.issubset(df.columns):
        raise ValueError(f"model_compare must contain columns: {req}")

    df = df.dropna(subset=[gene_col, effect_col]).copy()

    if fdr_threshold is not None and fdr_col in df.columns:
        df = df[df[fdr_col] <= fdr_threshold].copy()

    up_formate = df[df[effect_col] > effect_threshold].copy()
    up_phosphate = df[df[effect_col] < -effect_threshold].copy()

    # optional ranking before enrichment
    up_formate = up_formate.sort_values(effect_col, ascending=False)
    up_phosphate = up_phosphate.sort_values(effect_col, ascending=True)

    if top_n is not None:
        up_formate = up_formate.head(top_n)
        up_phosphate = up_phosphate.head(top_n)

    formate_genes = up_formate[gene_col].dropna().astype(str).tolist()
    phosphate_genes = up_phosphate[gene_col].dropna().astype(str).tolist()

    outdir = Path(outdir or "analysis_outputs")
    outdir.mkdir(parents=True, exist_ok=True)

    outputs = {}

    if formate_genes:
        formate_tsv = outdir / f"{prefix}_up_in_formate.tsv"
        formate_df = run_string_enrichment(
            formate_genes,
            species=species,
            limit=500,
            output_path=str(formate_tsv),
            caller_identity=caller_identity
        )
        formate_csv = outdir / f"{prefix}_up_in_formate.csv"
        formate_df.to_csv(formate_csv, index=False)
        outputs["formate"] = {
            "genes": formate_genes,
            "tsv": str(formate_tsv),
            "csv": str(formate_csv),
            "df": formate_df,
        }

    if phosphate_genes:
        phosphate_tsv = outdir / f"{prefix}_up_in_phosphate.tsv"
        phosphate_df = run_string_enrichment(
            phosphate_genes,
            species=species,
            limit=500,
            output_path=str(phosphate_tsv),
            caller_identity=caller_identity
        )
        phosphate_csv = outdir / f"{prefix}_up_in_phosphate.csv"
        phosphate_df.to_csv(phosphate_csv, index=False)
        outputs["phosphate"] = {
            "genes": phosphate_genes,
            "tsv": str(phosphate_tsv),
            "csv": str(phosphate_csv),
            "df": phosphate_df,
        }

    return outputs

def run_string_enrichment(
    identifiers: Iterable[str],
    species: Optional[int] = 267377,
    limit: int = 100,
    output_path: Optional[str] = None,
    caller_identity: str = "quantitative-proteomics"
) -> pd.DataFrame:
    """
    Submit a list of gene identifiers to STRING's enrichment API and return a
    results DataFrame (TSV parsed).

    Parameters
    - identifiers: iterable of gene identifiers (one per gene). STRING will try
      to map them to its internal identifiers; use locus IDs or gene names as
      available.
    - species: NCBI taxonomy id (int). Default set to 2187 (please adjust if
      you know a different taxon id for your organism).
    - limit: maximum number of returned terms (passed to API where supported).
    - output_path: if provided, save the raw TSV to this path.
    - caller_identity: a short string identifying the caller (optional).

    Returns a pandas.DataFrame parsed from the TSV response. Raises
    RuntimeError on failure or if `requests` is not available.
    """
    if requests is None:
        raise RuntimeError("The 'requests' package is required for STRING API calls. Install it first.")

    ids_text = "\n".join(map(str, identifiers))

    # Build request data. Use the TSV enrichment endpoint.
    url = "https://string-db.org/api/tsv/enrichment"
    data = {
        "identifiers": ids_text,
        "species": species,
        "caller_identity": caller_identity,
        "limit": limit,
    }

    resp = requests.post(url, data=data, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"STRING API request failed (status={resp.status_code}): {resp.text[:200]}")

    text = resp.text
    if output_path is not None:
        try:
            with open(output_path, "w") as fh:
                fh.write(text)
        except Exception:
            # don't fail on save, still return parsed df
            pass

    # Parse TSV into DataFrame; API returns headered TSV
    try:
        df = pd.read_csv(io.StringIO(text), sep="\t")
    except Exception as e:
        raise RuntimeError(f"Failed to parse STRING TSV response: {e}")

    return df


def _read_df(df_or_path):
    """Accept either a pandas.DataFrame or a path to a TSV/CSV and return a DataFrame."""
    if isinstance(df_or_path, pd.DataFrame):
        return df_or_path.copy()
    p = str(df_or_path)
    # try TSV first, fallback to CSV
    try:
        return pd.read_csv(p, sep="\t")
    except Exception:
        return pd.read_csv(p)
def plot_top_terms_bar(df_or_path, outpath=None, top_n=20, category=None, show=False):
    """Horizontal barplot of top terms ordered by significance (-log10 p-value)."""
    df = _read_df(df_or_path)
    if category is not None:
        df = df[df["category"].astype(str) == str(category)].copy()

    if df.empty:
        print("No enrichment terms to plot (after optional filtering).")
        return

    for col in ("fdr", "p_value", "number_of_genes"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["p_value"]).copy()
    if df.empty:
        print("No enrichment terms with p_value available to plot.")
        return

    df["neglog10_pvalue"] = -np.log10(df["p_value"].clip(lower=1e-300))

    # build label column explicitly
    desc = df["description"].fillna(df.get("term", pd.Series(df.index, index=df.index))).astype(str)
    if "category" in df.columns:
        source = df["category"].fillna("unknown").astype(str)
        df["label"] = desc + " [" + source + "]"
        df["label"] = df["label"].apply(lambda s: "\n".join(textwrap.wrap(str(s), width=40)))
    else:
        df["label"] = desc

    plot_df = (
        df.sort_values("neglog10_pvalue", ascending=False)
          .drop_duplicates(subset=["label"], keep="first")
          .head(top_n)
          .copy()
    )

    values = plot_df["neglog10_pvalue"].values

    sns.set(style="whitegrid")
    BAR_COLOR = "#4A90E2"
    plt.figure(figsize=(9, max(4, len(values) * 0.38)))
    y = np.arange(len(values))
    plt.barh(y, values, color=BAR_COLOR, alpha=0.95)
    plt.yticks(y, plot_df["label"], fontsize=9)
    plt.xlabel("Normalized -log10(p-value)")
    plt.title(f"Top {len(values)} enriched terms")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    if outpath is not None:
        plt.savefig(outpath, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

def plot_terms_dotplot(df_or_path, outpath=None, top_n=30, category=None, show=False):
    """Dotplot: x = -log10(p_value), y = term [source], size = number_of_genes."""
    df = _read_df(df_or_path)
    if category is not None:
        df = df[df["category"].astype(str) == str(category)].copy()

    if df.empty:
        print("No enrichment terms to plot (after optional filtering).")
        return

    for col in ("fdr", "p_value", "number_of_genes"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["p_value"]).copy()
    if df.empty:
        print("No enrichment terms with p_value available to plot.")
        return

    df["neglog10_pvalue"] = -np.log10(df["p_value"].clip(lower=1e-300))

    # build label column explicitly before any deduping
    desc = df["description"].fillna(df.get("term", pd.Series(df.index, index=df.index))).astype(str)
    if "category" in df.columns:
        source = df["category"].fillna("unknown").astype(str)
        df["label"] = desc + " [" + source + "]"
        df["label"] = df["label"].apply(lambda s: s[:100] + "..." if len(s) > 100 else s)
        df["label"] = df["label"].apply(lambda s: "\n".join(textwrap.wrap(str(s), width=60)))
    else:
        df["label"] = desc

    plot_df = (
        df.sort_values("neglog10_pvalue", ascending=False)
          .drop_duplicates(subset=["label"], keep="first")
          .head(top_n)
          .copy()
    )

    x_raw = plot_df["neglog10_pvalue"].values
    x = (x_raw - np.min(x_raw)) / (np.max(x_raw) - np.min(x_raw) + 1e-6)
    x = x_raw
    y = np.arange(len(plot_df))[::-1]

    sizes = plot_df.get(
        "number_of_genes",
        pd.Series(1, index=plot_df.index)
    ).fillna(1).astype(float).values
    sizes = (sizes - sizes.min() + 1) / max(1.0, (sizes.max() - sizes.min() + 1e-6)) * 200

    sns.set(style="whitegrid")
    cmap = sns.color_palette("plasma", as_cmap=True)

    plt.figure(figsize=(9, max(4, len(plot_df) * 0.38)))
    plt.scatter(
        x,
        y,
        s=sizes,
        c=x,
        cmap=cmap,
        alpha=0.9,
        edgecolors="#222222",
        linewidths=0.3
    )

    plt.yticks(y, plot_df["label"], fontsize=9)
    plt.xlabel("Normalized -log10(p-value)")
    plt.title(f"STRING enrichment (top {len(plot_df)} terms)")
    plt.colorbar(label="-log10(p-value)")
    plt.tight_layout()

    if outpath is not None:
        plt.savefig(outpath, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt



def plot_pathway_venn_from_enrichment(
    formate_enrichment_tsv,
    phosphate_enrichment_tsv,
    outpath=None,
    pval_threshold=0.05,
    use_fdr=False,
    use_description_only=False,
    show=False,
):
    """
    Clean Venn diagram of significantly enriched pathways in formate vs phosphate.

    Parameters
    ----------
    pval_threshold : float
        Significance threshold applied to p_value or fdr.
    use_fdr : bool
        If True, filter on fdr instead of p_value.
    """
    from matplotlib_venn import venn2

    def _read_enrichment(path):
        df = pd.read_csv(path, sep="\t")

        for col in ("p_value", "fdr"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        stat_col = "fdr" if use_fdr else "p_value"
        df = df.dropna(subset=[stat_col]).copy()
        df = df[df[stat_col] < pval_threshold].copy()

        desc = df["description"].fillna(df.get("term", pd.Series(df.index, index=df.index))).astype(str)

        if use_description_only:
            df["label"] = desc
        else:
            if "category" in df.columns:
                src = df["category"].fillna("unknown").astype(str)
                df["label"] = desc + " [" + src + "]"
            else:
                df["label"] = desc

        df = (
            df.sort_values(stat_col, ascending=True)
              .drop_duplicates(subset=["label"], keep="first")
              .copy()
        )

        return df

    df_formate = _read_enrichment(formate_enrichment_tsv)
    df_phosphate = _read_enrichment(phosphate_enrichment_tsv)

    set_formate = set(df_formate["label"])
    set_phosphate = set(df_phosphate["label"])

    plt.figure(figsize=(7, 7))
    venn2(
        [set_formate, set_phosphate],
        set_labels=(
            f"Up in formate\n({len(set_formate)} significant)",
            f"Up in phosphate\n({len(set_phosphate)} significant)"
        )
    )
    stat_name = "FDR" if use_fdr else "p-value"
    plt.title(f"Overlap of enriched pathways ({stat_name} < {pval_threshold})")

    if outpath is not None:
        outpath = Path(outpath)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outpath, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def save_pathway_overlap_table_from_enrichment(
    formate_enrichment_tsv,
    phosphate_enrichment_tsv,
    outpath=None,
    pval_threshold=0.05,
    use_fdr=False,
    use_description_only=False,
):
    """
    Save a table listing significantly enriched pathways unique to formate,
    unique to phosphate, and overlapping.
    """
    def _read_enrichment(path, direction_name):
        df = pd.read_csv(path, sep="\t")

        for col in ("p_value", "fdr"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        stat_col = "fdr" if use_fdr else "p_value"
        df = df.dropna(subset=[stat_col]).copy()
        df = df[df[stat_col] < pval_threshold].copy()

        desc = df["description"].fillna(df.get("term", pd.Series(df.index, index=df.index))).astype(str)

        if use_description_only:
            df["label"] = desc
        else:
            if "category" in df.columns:
                src = df["category"].fillna("unknown").astype(str)
                df["label"] = desc + " [" + src + "]"
            else:
                df["label"] = desc

        df = (
            df.sort_values(stat_col, ascending=True)
              .drop_duplicates(subset=["label"], keep="first")
              .copy()
        )

        df["direction"] = direction_name
        return df

    df_formate = _read_enrichment(formate_enrichment_tsv, "formate")
    df_phosphate = _read_enrichment(phosphate_enrichment_tsv, "phosphate")

    set_formate = set(df_formate["label"])
    set_phosphate = set(df_phosphate["label"])

    overlap_labels = set_formate & set_phosphate
    formate_only_labels = set_formate - set_phosphate
    phosphate_only_labels = set_phosphate - set_formate

    formate_only = df_formate[df_formate["label"].isin(formate_only_labels)].copy()
    phosphate_only = df_phosphate[df_phosphate["label"].isin(phosphate_only_labels)].copy()
    overlap_formate = df_formate[df_formate["label"].isin(overlap_labels)].copy()
    overlap_phosphate = df_phosphate[df_phosphate["label"].isin(overlap_labels)].copy()

    formate_only["venn_region"] = "formate_only"
    phosphate_only["venn_region"] = "phosphate_only"
    overlap_formate["venn_region"] = "overlap"
    overlap_phosphate["venn_region"] = "overlap"

    overlap_merged = overlap_formate[["label", "p_value", "fdr", "description", "category"]].merge(
        overlap_phosphate[["label", "p_value", "fdr"]],
        on="label",
        how="outer",
        suffixes=("_formate", "_phosphate")
    )
    overlap_merged["venn_region"] = "overlap"

    formate_only_out = formate_only[["label", "p_value", "fdr", "description", "category", "venn_region"]].copy()
    phosphate_only_out = phosphate_only[["label", "p_value", "fdr", "description", "category", "venn_region"]].copy()

    formate_only_out = formate_only_out.rename(columns={"p_value": "p_value_formate", "fdr": "fdr_formate"})
    phosphate_only_out = phosphate_only_out.rename(columns={"p_value": "p_value_phosphate", "fdr": "fdr_phosphate"})

    table = pd.concat([
        formate_only_out,
        phosphate_only_out,
        overlap_merged
    ], ignore_index=True, sort=False)

    region_order = {"formate_only": 0, "overlap": 1, "phosphate_only": 2}
    table["_region_order"] = table["venn_region"].map(region_order)

    sort_col = "p_value_formate" if "p_value_formate" in table.columns else None
    if sort_col is not None:
        table = table.sort_values(["_region_order", sort_col], na_position="last")
    else:
        table = table.sort_values("_region_order", na_position="last")

    table = table.drop(columns="_region_order")

    if outpath is not None:
        outpath = Path(outpath)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(outpath, index=False)

    return table



def make_enrichment_plots(tsv_path, outdir=None, prefix="string_enrichment", top_n_bar=20, top_n_dot=20, show=False, category=None):
    """Convenience function: read TSV and create two plots saved to outdir.

    Returns list of saved file paths.
    """
    df = _read_df(tsv_path)
    outdir = outdir or Path("analysis_outputs")
    try:
        outdir = Path(outdir)
    except Exception:
        from pathlib import Path as _P
        outdir = _P(str(outdir))
    outdir.mkdir(parents=True, exist_ok=True)

    bar_path = outdir / f"{prefix}_top_terms_bar.png"
    dot_path = outdir / f"{prefix}_dotplot.png"

    plot_top_terms_bar(df, outpath=str(bar_path), top_n=top_n_bar, category=category, show=show)
    plot_terms_dotplot(df, outpath=str(dot_path), top_n=top_n_dot, category=category, show=show)

    return [str(bar_path), str(dot_path)]
