import pandas as pd
import numpy as np
import re


file = "data/formate_dataset.xlsx"

# Read with 3 header rows
df_wide = pd.read_excel(file, header=[0, 1, 2])

# ---- Split by position: A–F = meta, G+ = measurements
df_meta = df_wide.iloc[:, :6].copy() # columns A-F
df_meas = df_wide.iloc[:, 6:].copy() # columns G onward (MultiIndex with three levels)

# Identify meta columns and use 2nd header (sample name) if it exists (is a measurement col)
def flatten_meta_name(col):
    # if measurement data with multilevel col name
    if isinstance(col, tuple):
        # look at each of the levels if it is multilevel
        for lvl in [1, 0, 2]:
            s = str(col[lvl]).strip()
            if s and s != "nan" and not s.startswith("Unnamed"):
                return s
        return str(col)
    return str(col)

# flatten the name for each item in there
df_meta.columns = [flatten_meta_name(c) for c in df_meta.columns]

# ---- Melt measurements to make df flat
df_long = df_meas.stack([0, 1, 2]).reset_index()
# multilevel column stack creates columns: original row index levels + the 3 column levels + value
# rename:
df_long = df_long.rename(columns={
    df_long.columns[-4]: "block",
    df_long.columns[-3]: "sample",
    df_long.columns[-2]: "tech_rep",
    df_long.columns[-1]: "value",
})

# Attach metadata by row index
row_id_col = df_long.columns[0]
df_long = df_long.merge(
    df_meta.reset_index().rename(columns={"index": row_id_col}),
    on=row_id_col,
    how="left"
).drop(columns=[row_id_col])

# ---- Parse sample to get growth_Rate + bio_rep
def parse_sample(sample):
    s = str(sample).strip()
    if s.upper() == "EXP": # these didn't have biorep
        return pd.Series({"growth_rate": "exp", "bio_rep": np.nan})
    m = re.match(r"^(XS|S|M|F)(\d+)$", s, flags=re.I)
    if not m:
        return pd.Series({"growth_rate": np.nan, "bio_rep": np.nan})
    return pd.Series({"growth_rate": m.group(1).lower(), "bio_rep": int(m.group(2))})

df_long = pd.concat([df_long, df_long["sample"].apply(parse_sample)], axis=1)
df_long["tech_rep"] = pd.to_numeric(df_long["tech_rep"], errors="coerce").astype("Int64")

# ---- Identify measure type, then keep only ratios (dont wanto to keep spectral counting)
df_long["measure_type"] = np.where(
    df_long["block"].astype(str).str.contains("Relative quantification", case=False, na=False),
    "ratio_14N15N",
    np.where(
        df_long["block"].astype(str).str.contains("Mass fraction", case=False, na=False),
        "mass_fraction_spectral_count",
        "other"
    )
)

df_ratio_only = df_long[df_long["measure_type"] == "ratio_14N15N"].copy()


# reindex to be more readable
df = df_ratio_only.reindex(columns=['Gene names (ordered locus )', 'growth_rate', 'bio_rep', 'tech_rep', 'value', 'Anabolism/Catabolism', 'Assigned functional subsystem', 'Annotation (Uniprot)', 'Automatic classification (RAST)', 'Present in dataset', 'sample', 'measure_type', 'block'])
df = df.drop(columns=['sample', 'measure_type', 'block']) # remove these cols because they dont tell anytbing

print(df.head())
df.to_csv("data/formate_dataset_cleaned.csv", index=False)