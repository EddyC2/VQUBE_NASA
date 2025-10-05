#Clustering algorithm for data analysis

import re
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

# ---- Load & clean ----
CSV_PATH = "CDA_ind.csv"               # change path if needed
df = pd.read_csv(CSV_PATH)

year_cols = [c for c in df.columns if re.match(r"^CDA\.ind\.\d{4}$", c)]
years = np.array([int(c.split('.')[-1]) for c in year_cols])

df[year_cols] = df[year_cols].replace(-8888, np.nan)
df = df.loc[df[year_cols].notna().sum(axis=1) >= 10].copy()
df[year_cols] = df[year_cols].interpolate(axis=1, limit_direction="both")

# ---- Features: linear trend per country ----
def fit_trend(row_vals, years_array):
    mask = ~np.isnan(row_vals)
    X = years_array[mask].reshape(-1, 1)
    y = row_vals[mask]
    if len(y) < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    lr = LinearRegression().fit(X, y)
    slope = lr.coef_[0]
    y_hat = lr.predict(X)
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    start_val = row_vals[0]
    end_val = row_vals[-1]
    delta = end_val - start_val
    mean_level = np.nanmean(row_vals)
    pct_slope_per_year = (slope / mean_level) if (mean_level and not np.isnan(mean_level)) else np.nan
    return slope, r2, start_val, end_val, delta, pct_slope_per_year

features = df[year_cols].apply(
    lambda r: pd.Series(
        fit_trend(r.values, years),
        index=["slope","r2","start","end","delta","pct_slope_per_year"]
    ),
    axis=1
)

feat_df = pd.concat([df[["code","iso","country"]], features], axis=1).replace([np.inf, -np.inf], np.nan)
feat_df_clean = feat_df.dropna(subset=["slope", "pct_slope_per_year", "r2", "delta"]).copy()

# ---- Clustering (K=3) ----
X = feat_df_clean[["slope", "pct_slope_per_year", "r2", "delta"]].values
X_mean, X_std = X.mean(axis=0), X.std(axis=0); X_std[X_std==0] = 1.0
X_scaled = (X - X_mean) / X_std

kmeans = KMeans(n_clusters=3, n_init=20, random_state=42)
feat_df_clean["cluster_id"] = kmeans.fit_predict(X_scaled)

# ---- Label clusters by centroid slope ----
centroids = kmeans.cluster_centers_
centroids_unscaled = centroids * X_std + X_mean
centroids_df = pd.DataFrame(centroids_unscaled, columns=["slope","pct_slope_per_year","r2","delta"])
centroids_df["cluster_id"] = range(3)

median_abs_slope = feat_df_clean["slope"].abs().median()
steady_threshold = 0.5 * median_abs_slope if median_abs_slope > 0 else 0.01

def to_label(s):
    if abs(s) <= steady_threshold:
        return "Steady"
    return "Increasing" if s > 0 else "Decreasing"

centroids_df["trend_label"] = centroids_df["slope"].apply(to_label)
id_to_label = dict(zip(centroids_df["cluster_id"], centroids_df["trend_label"]))
feat_df_clean["trend_label"] = feat_df_clean["cluster_id"].map(id_to_label)

# ---- Final table (THIS is 'result') ----
out_cols = ["code","iso","country","slope","pct_slope_per_year","r2","start","end","delta","cluster_id","trend_label"]
result = feat_df_clean[out_cols].sort_values(["trend_label","country"]).reset_index(drop=True)

# ---- CSV (optional) ----
result.to_csv("co2_trend_clusters.csv", index=False)

# ---- Excel export with multiple sheets ----
'''
from openpyxl import Workbook
with pd.ExcelWriter("co2_trend_clusters.xlsx", engine="openpyxl") as writer:
    # All countries
    result.to_excel(writer, sheet_name="All_Clusters", index=False)
    # Per-cluster sheets
    for label in ["Increasing","Steady","Decreasing"]:
        result[result["trend_label"] == label].sort_values("country").to_excel(writer, sheet_name=label, index=False)
    # Summary sheet: counts + centroid diagnostics
    summary_counts = (
        result["trend_label"].value_counts().rename_axis("trend_label").reset_index(name="num_countries")
        .sort_values("trend_label")
    )
    summary_counts.to_excel(writer, sheet_name="Summary", index=False, startrow=0)
    centroids_df[["cluster_id","slope","pct_slope_per_year","r2","delta","trend_label"]].to_excel(
        writer, sheet_name="Summary", index=False, startrow=len(summary_counts)+2
    )
    '''

print("✔ Wrote: co2_trend_clusters.xlsx")
