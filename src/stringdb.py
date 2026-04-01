"""Simple wrapper for calling the STRING-db enrichment API.

"""
from typing import Iterable, Optional
import io
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

try:
    import requests
except Exception:  # pragma: no cover - runtime dependency
    requests = None


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
    """Horizontal barplot of top terms ordered by significance (-log10 FDR).

    Parameters
    - df_or_path: DataFrame or path to STRING enrichment TSV
    - outpath: path to save PNG file; if None will not save
    - top_n: number of terms to show
    - category: if provided, filter to this category (e.g., 'Process', 'Component', 'COMPARTMENTS')
    - show: whether to call plt.show()
    """
    df = _read_df(df_or_path)
    if category is not None:
        df = df[df["category"].astype(str) == str(category)].copy()

    if df.empty:
        print("No enrichment terms to plot (after optional filtering).")
        return

    # ensure numeric
    for col in ("fdr", "p_value", "number_of_genes"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["fdr"]).copy()
    df["neglog10_fdr"] = -np.log10(df["fdr"].clip(lower=1e-300))

    plot_df = df.sort_values("neglog10_fdr", ascending=False).head(top_n)

    labels = plot_df["description"].fillna(plot_df.get("term", plot_df.index)).astype(str)
    values = plot_df["neglog10_fdr"].values

    plt.figure(figsize=(8, max(4, len(values) * 0.35)))
    y = np.arange(len(values))
    plt.barh(y, values, color="tab:blue", alpha=0.85)
    plt.yticks(y, labels)
    plt.xlabel("-log10(FDR)")
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
    """Dotplot: x = -log10(fdr), y = term, size = number_of_genes.

    Useful for quickly seeing both significance and gene counts.
    """
    df = _read_df(df_or_path)
    if category is not None:
        df = df[df["category"].astype(str) == str(category)].copy()

    if df.empty:
        print("No enrichment terms to plot (after optional filtering).")
        return

    for col in ("fdr", "p_value", "number_of_genes"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["fdr"]).copy()
    df["neglog10_fdr"] = -np.log10(df["fdr"].clip(lower=1e-300))

    plot_df = df.sort_values("neglog10_fdr", ascending=False).head(top_n).copy()
    plot_df["label"] = plot_df["description"].fillna(plot_df.get("term", plot_df.index)).astype(str)

    x = plot_df["neglog10_fdr"].values
    y = np.arange(len(plot_df))[::-1]
    sizes = plot_df.get("number_of_genes", pd.Series(1, index=plot_df.index)).fillna(1).astype(float).values
    sizes = (sizes - sizes.min() + 1) / max(1.0, (sizes.max() - sizes.min() + 1e-6)) * 200

    plt.figure(figsize=(8, max(4, len(plot_df) * 0.35)))
    plt.scatter(x, y, s=sizes, c=x, cmap="viridis", alpha=0.8)
    plt.yticks(y, plot_df["label"])
    plt.xlabel("-log10(FDR)")
    plt.title(f"STRING enrichment (top {len(plot_df)} terms)")
    plt.colorbar(label="-log10(FDR)")
    plt.tight_layout()

    if outpath is not None:
        plt.savefig(outpath, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def make_enrichment_plots(tsv_path, outdir=None, prefix="string_enrichment", top_n_bar=20, top_n_dot=30, show=False, category=None):
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
