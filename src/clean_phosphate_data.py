import pandas as pd
import numpy as np
import re

file = "data/phospate_dataset.xlsx"

# Read with 2 header rows
df_wide = pd.read_excel(file, header=[0, 1])

# Forward-fill the top header row because merged Excel cells become NaN after reading
lvl0 = pd.Series(df_wide.columns.get_level_values(0)).ffill()
lvl1 = pd.Series(df_wide.columns.get_level_values(1))
df_wide.columns = pd.MultiIndex.from_arrays([lvl0, lvl1])

# First 6 columns are metadata
df_meta = df_wide.iloc[:, :6].copy()
df_meas = df_wide.iloc[:, 6:].copy()

# Flatten metadata column names using the second header row when possible
def flatten_meta_name(col):
    if isinstance(col, tuple):
        for lvl in [1, 0]:
            s = str(col[lvl]).strip()
            if s and s != "nan" and not s.startswith("Unnamed"):
                return s
        return str(col)
    return str(col)

df_meta.columns = [flatten_meta_name(c) for c in df_meta.columns]

# Melt measurement columns into long format
df_long = df_meas.stack([0, 1]).reset_index()

# Rename generated columns
df_long = df_long.rename(columns={
    df_long.columns[-3]: "block",
    df_long.columns[-2]: "sample",
    df_long.columns[-1]: "value",
})

# Merge metadata back using original row index
row_id_col = df_long.columns[0]
df_long = df_long.merge(
    df_meta.reset_index().rename(columns={"index": row_id_col}),
    on=row_id_col,
    how="left"
).drop(columns=[row_id_col])

# Parse sample names like:
# P-XS1 -> growth_rate=xs, bio_rep=1
# P-I2  -> growth_rate=i,  bio_rep=2
# P-F3  -> growth_rate=f,  bio_rep=3
# E1    -> growth_rate=exp, bio_rep=1
# there arent any technical replicates
def parse_sample(sample):
    s = str(sample).strip().upper()

    # E1, E2, E3
    m_exp = re.match(r"^E(\d+)$", s)
    if m_exp:
        return pd.Series({
            "growth_rate": "exp",
            "bio_rep": int(m_exp.group(1))
        })

    # P-XS1, P-I2, P-F3, etc.
    m = re.match(r"^P-([A-Z]+)(\d+)$", s)
    if m:
        return pd.Series({
            "growth_rate": m.group(1).lower(),
            "bio_rep": int(m.group(2))
        })

    return pd.Series({
        "growth_rate": np.nan,
        "bio_rep": np.nan
    })

df_long = pd.concat([df_long, df_long["sample"].apply(parse_sample)], axis=1)

# No separate tech replicate information exists in this file
df_long["tech_rep"] = pd.Series([pd.NA] * len(df_long), dtype="Int64")

# Identify measurement type
df_long["measure_type"] = np.where(
    df_long["block"].astype(str).str.contains("Relative quantification", case=False, na=False),
    "ratio_14N15N",
    np.where(
        df_long["block"].astype(str).str.contains("Mass fraction", case=False, na=False),
        "mass_fraction_spectral_count",
        "other"
    )
)

# Keep only ratio data, matching your old pipeline
df_ratio_only = df_long[df_long["measure_type"] == "ratio_14N15N"].copy()

# Reorder columns to match your old final format
df = df_ratio_only.reindex(columns=[
    'Gene names (ordered locus )',
    'growth_rate',
    'bio_rep',
    'tech_rep',
    'value',
    'Anabolism/Catabolism',
    'Assigned functional subsystem',
    'Annotation (Uniprot)',
    'Automatic classification (RAST)',
    'Present in dataset',
    'sample',
    'measure_type',
    'block'
])

# Drop helper columns to match prior output
df = df.drop(columns=['sample', 'measure_type', 'block'])

print(df.head())
df.to_csv("data/phosphate_dataset_cleaned.csv", index=False)