import re
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import plotly.express as px
CSV_PATH = "CDA_ind.csv"
df = pd.read_csv(CSV_PATH)
year_cols = [c for c in df.columns if re.match(r"^CDA\.ind\.\d{4}$", c)]
years = np.array([int(c.split('.')[-1]) for c in year_cols])
df[year_cols] = df[year_cols].replace(-8888, np.nan)
min_valid_points = 10
valid_mask = df[year_cols].notna().sum(axis=1) >= min_valid_points
df = df.loc[valid_mask].copy()
df[year_cols] = df[year_cols].interpolate(axis=1, limit_direction="both")
def fit_trend(row_vals, years_array):
    """
    Fit a simple linear regression CO2 = a + b*year on available points.
    Returns slope b (units per year), R^2, start_value, end_value, delta (end-start), pct_slope_per_year.
    """
    mask = ~np.isnan(row_vals)
    X = years_array[mask].reshape(-1, 1)
    y = row_vals[mask]
    if len(y) < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    lr = LinearRegression()
    lr.fit(X, y)
    slope = lr.coef_[0]  # units of emission per year
    y_hat = lr.predict(X)
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    start_val = row_vals[0]
    end_val = row_vals[-1]
    delta = end_val - start_val

    # Percent slope per year = slope divided by mean level (to be scale-free)
    mean_level = np.nanmean(row_vals)
    pct_slope_per_year = (slope / mean_level) if (mean_level and not np.isnan(mean_level)) else np.nan

    return slope, r2, start_val, end_val, delta, pct_slope_per_year

features = df[year_cols].apply(lambda r: pd.Series(fit_trend(r.values, years),
                                                   index=["slope","r2","start","end","delta","pct_slope_per_year"]),
                               axis=1)

feat_df = pd.concat([df[["code","iso","country"]], features], axis=1)

# Replace infs if any
feat_df = feat_df.replace([np.inf, -np.inf], np.nan)

# Drop rows with NaNs in the key fields used by clustering
feat_df_clean = feat_df.dropna(subset=["slope", "pct_slope_per_year", "r2", "delta"]).copy()

# ----------------------------
# 3) Clustering (KMeans k=3)
# ----------------------------
X = feat_df_clean[["slope", "pct_slope_per_year", "r2", "delta"]].values

# KMeans is sensitive to scale; robust-scaling is optional but helpful
# We'll do a simple standardization
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_std[X_std == 0] = 1.0
X_scaled = (X - X_mean) / X_std

kmeans = KMeans(n_clusters=3, n_init=20, random_state=42)
labels = kmeans.fit_predict(X_scaled)

feat_df_clean["cluster_id"] = labels

# ----------------------------
# 4) Label clusters by centroid slope sign
# ----------------------------
centroids = kmeans.cluster_centers_  # in scaled space
# Convert centroid back to original scale for interpretability
centroids_unscaled = centroids * X_std + X_mean
centroids_df = pd.DataFrame(centroids_unscaled, columns=["slope","pct_slope_per_year","r2","delta"])
centroids_df["cluster_id"] = range(3)

# Decide which centroid corresponds to Increasing / Steady / Decreasing based on "slope"
# We’ll treat "steady" as small absolute slope. Threshold can be tuned.
# A reasonable heuristic: absolute slope < 0.5 * median(|slope|) across all countries → steady
median_abs_slope = feat_df_clean["slope"].abs().median()
steady_threshold = 0.5 * median_abs_slope if median_abs_slope > 0 else 0.01  # fallback small value

def cluster_label_from_slope(s):
    if abs(s) <= steady_threshold:
        return "Steady"
    return "Increasing" if s > 0 else "Decreasing"

centroids_df["trend_label"] = centroids_df["slope"].apply(cluster_label_from_slope)

# Map numeric cluster ids to labels
id_to_label = dict(zip(centroids_df["cluster_id"], centroids_df["trend_label"]))
feat_df_clean["trend_label"] = feat_df_clean["cluster_id"].map(id_to_label)

# ----------------------------
# 5) Save results
# ----------------------------
out_cols = ["code","iso","country","slope","pct_slope_per_year","r2","start","end","delta","cluster_id","trend_label"]
result = feat_df_clean[out_cols].sort_values(["trend_label","country"]).reset_index(drop=True)
result.to_csv("co2_trend_clusters.csv", index=False)
print("Saved: co2_trend_clusters.csv")
print(result["trend_label"].value_counts())

# ----------------------------
# 6) Quick sanity check: show top examples from each cluster
# ----------------------------

fig = px.scatter(
    result,
    x="slope",
    y="delta",
    color="trend_label",
    hover_name="country",  # shows country name when you hover
    title="CO₂ Emission Trends by Country (Interactive)",
    color_discrete_map={"Increasing": "red", "Steady": "blue", "Decreasing": "green"},
    width=900,
    height=600

)

fig.update_traces(marker=dict(size=10, line=dict(width=0.5, color='DarkSlateGrey')))
fig.update_layout(
    legend=dict(
        title="CO₂ Emission Trend",     # custom title
        orientation="v",                # vertical legend
        x=1.02, y=1,                    # position on right
        bgcolor="rgba(255,255,255,0.6)",# translucent background
        bordercolor="black",
        borderwidth=1
    )
)
fig.show()

fig_json = fig.to_json()
with open("co2_chart.json", "w", encoding="utf-8") as f:
    f.write(fig_json)


'''
import re
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

CSV_PATH = "CDA_ind.csv"
df = pd.read_csv(CSV_PATH)
year_cols = [c for c in df.columns if re.match(r"^CDA\.ind\.\d{4}$", c)]
years = np.array([int(c.split('.')[-1]) for c in year_cols])
df[year_cols] = df[year_cols].replace(-8888, np.nan)
min_valid_points = 10
valid_mask = df[year_cols].notna().sum(axis=1) >= min_valid_points
df = df.loc[valid_mask].copy()
df[year_cols] = df[year_cols].interpolate(axis=1, limit_direction="both")

def fit_trend(row_vals, years_array):
    """
    Fit a simple linear regression CO2 = a + b*year on available points.
    Returns slope b (units per year), R^2, start_value, end_value, delta (end-start), pct_slope_per_year.
    """
    mask = ~np.isnan(row_vals)
    X = years_array[mask].reshape(-1, 1)
    y = row_vals[mask]
    if len(y) < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    lr = LinearRegression()
    lr.fit(X, y)
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

features = df[year_cols].apply(lambda r: pd.Series(fit_trend(r.values, years),
                                                   index=["slope","r2","start","end","delta","pct_slope_per_year"]),
                               axis=1)

feat_df = pd.concat([df[["code","iso","country"]], features], axis=1)
feat_df = feat_df.replace([np.inf, -np.inf], np.nan)
feat_df_clean = feat_df.dropna(subset=["slope", "pct_slope_per_year", "r2", "delta"]).copy()

# Clustering
X = feat_df_clean[["slope", "pct_slope_per_year", "r2", "delta"]].values
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_std[X_std == 0] = 1.0
X_scaled = (X - X_mean) / X_std

kmeans = KMeans(n_clusters=3, n_init=20, random_state=42)
labels = kmeans.fit_predict(X_scaled)
feat_df_clean["cluster_id"] = labels

# Label clusters
centroids = kmeans.cluster_centers_
centroids_unscaled = centroids * X_std + X_mean
centroids_df = pd.DataFrame(centroids_unscaled, columns=["slope","pct_slope_per_year","r2","delta"])
centroids_df["cluster_id"] = range(3)

median_abs_slope = feat_df_clean["slope"].abs().median()
steady_threshold = 0.5 * median_abs_slope if median_abs_slope > 0 else 0.01

def cluster_label_from_slope(s):
    if abs(s) <= steady_threshold:
        return "Steady"
    return "Increasing" if s > 0 else "Decreasing"

centroids_df["trend_label"] = centroids_df["slope"].apply(cluster_label_from_slope)
id_to_label = dict(zip(centroids_df["cluster_id"], centroids_df["trend_label"]))
feat_df_clean["trend_label"] = feat_df_clean["cluster_id"].map(id_to_label)

# Save results
out_cols = ["code","iso","country","slope","pct_slope_per_year","r2","start","end","delta","cluster_id","trend_label"]
result = feat_df_clean[out_cols].sort_values(["trend_label","country"]).reset_index(drop=True)
result.to_csv("co2_trend_clusters.csv", index=False)
print("Saved: co2_trend_clusters.csv")
print(result["trend_label"].value_counts())

# ----------------------------
# Matplotlib Scatter Plot
# ----------------------------

# Define colors for each trend
color_map = {"Increasing": "red", "Steady": "blue", "Decreasing": "green"}

fig, ax = plt.subplots(figsize=(12, 8))

# Plot each trend category
for trend in ["Increasing", "Steady", "Decreasing"]:
    subset = result[result["trend_label"] == trend]
    ax.scatter(
        subset["slope"], 
        subset["delta"], 
        c=color_map[trend], 
        label=trend,
        s=100,  # marker size
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5
    )

# Add labels and title
ax.set_xlabel('Slope (CO₂ change per year)', fontsize=12, fontweight='bold')
ax.set_ylabel('Delta (Total change)', fontsize=12, fontweight='bold')
ax.set_title('CO₂ Emission Trends by Country', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(title='CO₂ Emission Trend', loc='best', frameon=True, shadow=True)

# Add zero lines for reference
ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

plt.tight_layout()
plt.savefig('co2_trends_matplotlib.png', dpi=300, bbox_inches='tight')
print("Saved: co2_trends_matplotlib.png")
plt.show()

# ----------------------------
# Optional: Annotate specific countries
# ----------------------------

fig, ax = plt.subplots(figsize=(14, 10))

for trend in ["Increasing", "Steady", "Decreasing"]:
    subset = result[result["trend_label"] == trend]
    ax.scatter(
        subset["slope"], 
        subset["delta"], 
        c=color_map[trend], 
        label=trend,
        s=100,
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5
    )

# Annotate some notable countries (top/bottom by slope)
top_increasing = result[result["trend_label"] == "Increasing"].nlargest(3, "slope")
top_decreasing = result[result["trend_label"] == "Decreasing"].nsmallest(3, "slope")

for _, row in pd.concat([top_increasing, top_decreasing]).iterrows():
    ax.annotate(
        row["country"],
        xy=(row["slope"], row["delta"]),
        xytext=(10, 10),
        textcoords='offset points',
        fontsize=8,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='black', lw=0.5)
    )

ax.set_xlabel('Slope (CO₂ change per year)', fontsize=12, fontweight='bold')
ax.set_ylabel('Delta (Total change)', fontsize=12, fontweight='bold')
ax.set_title('CO₂ Emission Trends by Country (with Notable Countries)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(title='CO₂ Emission Trend', loc='best', frameon=True, shadow=True)
ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

plt.tight_layout()
plt.savefig('co2_trends_annotated.png', dpi=300, bbox_inches='tight')
print("Saved: co2_trends_annotated.png")
plt.show()
'''