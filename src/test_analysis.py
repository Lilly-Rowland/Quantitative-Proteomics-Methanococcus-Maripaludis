import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

df = pd.read_csv("formate_dataset_cleaned.csv")

df["log2_ratio"] = np.log2(df["value"])



# Get average technical replicates --> Now each row = one biological replicate measurement.
df_bio = (
    df.groupby(
        ["Gene names (ordered locus )", "growth_rate", "bio_rep"],
        as_index=False
    )["log2_ratio"]
    .mean()
)

#comparing two growth rates...
xs = df_bio[df_bio["growth_rate"] == "xs"]
s  = df_bio[df_bio["growth_rate"] == "s"]


# get log fold change
xs_mean = xs.groupby("Gene names (ordered locus )")["log2_ratio"].mean()
s_mean  = s.groupby("Gene names (ordered locus )")["log2_ratio"].mean()

log2fc = xs_mean - s_mean


# perform a welch's t-test

def pval(gene):

    xs_vals = xs.loc[
        xs["Gene names (ordered locus )"] == gene,
        "log2_ratio"
    ]

    s_vals = s.loc[
        s["Gene names (ordered locus )"] == gene,
        "log2_ratio"
    ]

    if len(xs_vals) > 1 and len(s_vals) > 1:
        return ttest_ind(xs_vals, s_vals, equal_var=False).pvalue
    else:
        return np.nan
    
results = pd.DataFrame({
    "log2FC": log2fc
})

results["p_value"] = results.index.map(pval)

results = results.dropna(subset=["log2FC"]).copy()

print(results.head())

# Make a plot
results["neglog10_p"] = -np.log10(results["p_value"])


plt.figure()

plt.scatter(
    results["log2FC"],
    results["neglog10_p"],
    s=10
)

plt.xlabel("log2 Fold Change (XS vs S)")
plt.ylabel("-log10(p-value)")
plt.title("Volcano Plot – Formate XS vs S")

plt.axvline(0)
plt.show()