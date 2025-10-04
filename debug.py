import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
print("Loading data...")
df = pd.read_csv("CDA_ind.csv")
ghp = pd.read_csv("GHP_ind.csv")
par = pd.read_csv("PAR_ind.csv")

# Melt to long format
df_long = df.melt(
    id_vars=["code", "iso", "country"], 
    var_name="year", 
    value_name="value"
)
ghp_long = ghp.melt(
    id_vars=["code", "iso", "country"], 
    var_name="year", 
    value_name="value"
)
par_long = par.melt(
    id_vars=["code", "iso", "country"], 
    var_name="year", 
    value_name="value"
)

# Extract years
df_long["year"] = df_long["year"].str.extract(r'(\d{4})').astype(int)
ghp_long["year"] = ghp_long["year"].str.extract(r'(\d{4})').astype(int)
par_long["year"] = par_long["year"].str.extract(r'(\d{4})').astype(int)

# Basic data info
print("\n=== DATA OVERVIEW ===")
print(f"CDA - Years: {df_long['year'].min()} to {df_long['year'].max()}")
print(f"CDA - Countries: {df_long['country'].nunique()}")
print(f"CDA - Value range: {df_long['value'].min():.2f} to {df_long['value'].max():.2f}")

print(f"\nGHP - Years: {ghp_long['year'].min()} to {ghp_long['year'].max()}")
print(f"GHP - Countries: {ghp_long['country'].nunique()}")
print(f"GHP - Value range: {ghp_long['value'].min():.2f} to {ghp_long['value'].max():.2f}")

print(f"\nPAR - Years: {par_long['year'].min()} to {par_long['year'].max()}")
print(f"PAR - Countries: {par_long['country'].nunique()}")
print(f"PAR - Value range: {par_long['value'].min():.2f} to {par_long['value'].max():.2f}")

# Check for missing data
print("\n=== MISSING DATA ===")
print(f"CDA - Missing values: {df_long['value'].isna().sum()} ({df_long['value'].isna().sum()/len(df_long)*100:.1f}%)")
print(f"GHP - Missing values: {ghp_long['value'].isna().sum()} ({ghp_long['value'].isna().sum()/len(ghp_long)*100:.1f}%)")
print(f"PAR - Missing values: {par_long['value'].isna().sum()} ({par_long['value'].isna().sum()/len(par_long)*100:.1f}%)")


# Plot sample countries from CDA dataset
print("\n=== VISUALIZING CDA DATA PATTERNS ===")
sample_countries = ['United States', 'China', 'India', 'Germany', 'Brazil', 'Japan']

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, country in enumerate(sample_countries):
    data = df_long[df_long['country'] == country].sort_values('year')
    
    if not data.empty:
        axes[idx].plot(data['year'], data['value'], 'o-', linewidth=2, markersize=6)
        axes[idx].set_title(f"{country}", fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Year')
        axes[idx].set_ylabel('Value (0-100)')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_ylim(-5, 105)
        
        # Add trend line
        valid = data.dropna(subset=['value'])
        if len(valid) > 1:
            z = np.polyfit(valid['year'], valid['value'], 1)
            p = np.poly1d(z)
            axes[idx].plot(valid['year'], p(valid['year']), "--", alpha=0.5, color='red', label='Linear trend')
            axes[idx].legend()

plt.suptitle('CDA Data: Sample Country Patterns (1995-2020)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('debug_cda_patterns.png', dpi=150)
print("Saved: debug_cda_patterns.png")
plt.close()


# Analyze trend strength across all countries
print("\n=== ANALYZING TREND STRENGTH (CDA) ===")
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

trend_analysis = []

for country in df_long['country'].unique():
    data = df_long[df_long['country'] == country].sort_values('year')
    valid = data.dropna(subset=['value'])
    
    if len(valid) >= 10:  # Need at least 10 data points
        X = valid['year'].values.reshape(-1, 1)
        y = valid['value'].values
        
        model = LinearRegression()
        model.fit(X, y)
        r2 = r2_score(y, model.predict(X))
        slope = model.coef_[0]
        
        trend_analysis.append({
            'country': country,
            'r2': r2,
            'slope': slope,
            'n_points': len(valid),
            'std_dev': y.std(),
            'mean_value': y.mean()
        })

trend_df = pd.DataFrame(trend_analysis)

print(f"\nCountries with good linear fit (R² > 0.7): {(trend_df['r2'] > 0.7).sum()} / {len(trend_df)}")
print(f"Countries with poor linear fit (R² < 0.3): {(trend_df['r2'] < 0.3).sum()} / {len(trend_df)}")
print(f"Average R²: {trend_df['r2'].mean():.3f}")
print(f"Average slope: {trend_df['slope'].mean():.3f} (positive = increasing)")

print("\n=== COUNTRIES WITH BEST LINEAR FIT ===")
print(trend_df.nlargest(10, 'r2')[['country', 'r2', 'slope', 'mean_value']])

print("\n=== COUNTRIES WITH WORST LINEAR FIT ===")
print(trend_df.nsmallest(10, 'r2')[['country', 'r2', 'slope', 'std_dev']])


# Distribution of R² values
plt.figure(figsize=(10, 6))
plt.hist(trend_df['r2'], bins=30, edgecolor='black')
plt.xlabel('R² Score (fit to linear trend)')
plt.ylabel('Number of Countries')
plt.title('Distribution of Linear Fit Quality Across Countries (CDA)')
plt.axvline(trend_df['r2'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {trend_df["r2"].mean():.3f}')
plt.axvline(0.7, color='green', linestyle='--', linewidth=2, label='Good fit threshold (0.7)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('debug_r2_distribution.png', dpi=150)
print("\nSaved: debug_r2_distribution.png")
plt.close()


# Show worst-fit countries visually
print("\n=== VISUALIZING WORST-FIT COUNTRIES ===")
worst_countries = trend_df.nsmallest(6, 'r2')['country'].values

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, country in enumerate(worst_countries):
    data = df_long[df_long['country'] == country].sort_values('year')
    valid = data.dropna(subset=['value'])
    
    if not valid.empty:
        axes[idx].plot(valid['year'], valid['value'], 'o-', linewidth=2, markersize=6)
        
        # Add trend line
        X = valid['year'].values.reshape(-1, 1)
        y = valid['value'].values
        model = LinearRegression()
        model.fit(X, y)
        axes[idx].plot(valid['year'], model.predict(X), "--", alpha=0.5, color='red', linewidth=2)
        
        r2 = trend_df[trend_df['country'] == country]['r2'].values[0]
        axes[idx].set_title(f"{country}\nR² = {r2:.3f}", fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Year')
        axes[idx].set_ylabel('Value (0-100)')
        axes[idx].grid(True, alpha=0.3)

plt.suptitle('Worst Linear Fits: Data Too Volatile for Linear Regression', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('debug_worst_fits.png', dpi=150)
print("Saved: debug_worst_fits.png")
plt.close()


print("\n=== SUMMARY ===")
print(f"Total countries analyzed: {len(trend_df)}")
print(f"Good fits (R² > 0.7): {(trend_df['r2'] > 0.7).sum()} ({(trend_df['r2'] > 0.7).sum()/len(trend_df)*100:.1f}%)")
print(f"Moderate fits (0.3 < R² < 0.7): {((trend_df['r2'] > 0.3) & (trend_df['r2'] < 0.7)).sum()} ({((trend_df['r2'] > 0.3) & (trend_df['r2'] < 0.7)).sum()/len(trend_df)*100:.1f}%)")
print(f"Poor fits (R² < 0.3): {(trend_df['r2'] < 0.3).sum()} ({(trend_df['r2'] < 0.3).sum()/len(trend_df)*100:.1f}%)")

print("\n=== RECOMMENDATION ===")
if trend_df['r2'].mean() > 0.5:
    print("✅ Linear regression is appropriate for most countries")
    print(f"   Average MAE of 13.22 is reasonable given the data variability")
else:
    print("⚠️  Linear regression struggles with this data")
    print("   Consider alternative methods:")
    print("   - Moving average (last 3-5 years)")
    print("   - Exponential smoothing")
    print("   - Use last known value")

print("\nCheck the generated plots to see the actual data patterns!")