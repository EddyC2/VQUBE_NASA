
import pandas as pd 
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import os 

sentinels = [-8888, -9999]
MIN_YEARS = 6
future_years = np.arange(2021, 2040)
EPS = 1e-6 

# Loading the data
df = pd.read_csv("CDA_ind.csv")
ghp = pd.read_csv("GHP_ind.csv")
par = pd.read_csv("PAR_ind.csv")

# Melt all datasets
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

print("\n==============")
print("CDA range:", df_long['value'].min(), "to", df_long['value'].max())
print("GHP range:", ghp_long['value'].min(), "to", ghp_long['value'].max())
print("PAR range:", par_long['value'].min(), "to", par_long['value'].max())

def clean_and_impute(data, dataset_name):
    """Clean data and impute missing values"""
    print(f"\n=== Processing {dataset_name} ===")
    
    # Replace sentinel values
    data.loc[data['value'].isin(sentinels), 'value'] = np.nan
    
    # Check for suspicious values
    suspicious = data[(data['value'] > 100) | (data['value'] < 0)]
    print(f"Suspicious rows (value >100 or <0): {suspicious.shape}")
    if not suspicious.empty:
        print(suspicious.head(10))
    
    # Remove out-of-range values
    data.loc[data['value'] > 100, 'value'] = np.nan
    data.loc[data['value'] < 0, 'value'] = np.nan
    
    # Impute within each country
    data["value_imputed"] = (
        data.groupby("country")["value"]
        .transform(lambda s: s.interpolate(method="linear", limit_direction="both"))
    )
    
    # Fill remaining NaNs with global mean by year
    global_by_year = data.groupby('year')['value_imputed'].transform('mean')
    data['value_imputed'] = data['value_imputed'].fillna(global_by_year)
    
    return data


def make_predictions(data, dataset_name):
    """Generate predictions for all countries"""
    predictions_list = []
    
    for country, grp in data.groupby('country'):
        grp = grp.sort_values('year')
        valid = grp.dropna(subset=['value_imputed'])
        years = valid['year'].values
        vals = valid['value_imputed'].values

        if len(valid) == 0:
            preds = np.array([np.nan]*len(future_years))
        elif len(valid) < MIN_YEARS:
            last_val = vals[-1]
            preds = np.array([last_val]*len(future_years))
        else:
            X = years.reshape(-1,1)
            y = vals
            model = LinearRegression()
            model.fit(X, y)
            preds = model.predict(future_years.reshape(-1,1))
            preds = np.clip(preds, 0.0, 100.0)

        predictions_list.append(pd.DataFrame({
            'country': country,
            'year': future_years,
            'predicted_value': preds
        }))

    predictions_df = pd.concat(predictions_list, ignore_index=True)
    
    # Check for bad predictions
    bad = predictions_df[(predictions_df['predicted_value'] < 0) | 
                         (predictions_df['predicted_value'] > 100)]
    print(f"Any bad predictions after clipping: {bad.shape}")
    if not bad.empty:
        print(bad.head())
    
    return predictions_df


def plot_predictions(data, predictions, dataset_name, countries=['Albania', 'United States', 'China']):
    """Plot historical data and predictions"""
    plt.figure(figsize=(10, 6))
    
    for c in countries:
        hist = data[data['country']==c].sort_values('year')
        pred = predictions[predictions['country']==c]
        if not hist.empty:
            plt.plot(hist['year'], hist['value_imputed'], label=f"{c} hist")
        if not pred.empty:
            plt.plot(pred['year'], pred['predicted_value'], linestyle='--', label=f"{c} pred")
    
    plt.xlabel("Year")
    plt.ylabel("Value (0-100)")
    plt.title(f"{dataset_name} - Historical Data & Predictions")
    plt.legend()
    plt.savefig(f"{dataset_name}_predictions_plot.png")
    plt.close()
    print(f"Plot saved: {dataset_name}_predictions_plot.png")


# Process each dataset
datasets = [
    (df_long, "CDA"),
    (ghp_long, "GHP"),
    (par_long, "PAR")
]

for data, name in datasets:
    # Clean and impute
    data_cleaned = clean_and_impute(data, name)
    
    # Make predictions
    predictions = make_predictions(data_cleaned, name)
    
    # Save predictions
    output_file = f"{name}_predictions_2030.csv"
    predictions.to_csv(output_file, index=False)
    print(f"Predictions saved: {output_file}")
    
    # Create plot
    plot_predictions(data_cleaned, predictions, name)

print("\n=== All processing complete! ===")


