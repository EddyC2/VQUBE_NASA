import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import os

# Create output folder
output_folder = "accuracy_evaluation"
os.makedirs(output_folder, exist_ok=True)
print(f"Output folder created: {output_folder}/\n")

sentinels = [-8888, -9999]
MIN_YEARS = 6
EPS = 1e-6 

# Loading the data
df = pd.read_csv("CDA_ind.csv")

df_long = df.melt(
    id_vars=["code", "iso", "country"], 
    var_name="year", 
    value_name="value"
)

df_long["year"] = df_long["year"].str.extract(r'(\d{4})').astype(int)

# Clean data
df_long.loc[df_long['value'].isin(sentinels), 'value'] = np.nan
df_long.loc[df_long['value'] > 100, 'value'] = np.nan
df_long.loc[df_long['value'] < 0, 'value'] = np.nan

# Impute missing values
df_long["value_imputed"] = (
    df_long.groupby("country")["value"]
    .transform(lambda s: s.interpolate(method="linear", limit_direction="both"))
)

global_by_year = df_long.groupby('year')['value_imputed'].transform('mean')
df_long['value_imputed'] = df_long['value_imputed'].fillna(global_by_year)


# ==== ACCURACY EVALUATION ====

# Define train/test split year
SPLIT_YEAR = 2015  # Use data before 2015 to train, 2015+ to test

all_metrics = []
test_predictions = []

for country, grp in df_long.groupby('country'):
    grp = grp.sort_values('year')
    
    # Split into train and test
    train = grp[grp['year'] < SPLIT_YEAR].dropna(subset=['value_imputed'])
    test = grp[grp['year'] >= SPLIT_YEAR].dropna(subset=['value_imputed'])
    
    # Need enough training data and at least some test data
    if len(train) < MIN_YEARS or len(test) == 0:
        continue
    
    # Train model
    X_train = train['year'].values.reshape(-1, 1)
    y_train = train['value_imputed'].values
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict on test set
    X_test = test['year'].values.reshape(-1, 1)
    y_test = test['value_imputed'].values
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0.0, 100.0)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    all_metrics.append({
        'country': country,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'n_train': len(train),
        'n_test': len(test)
    })
    
    # Store predictions for visualization
    for year, actual, pred in zip(test['year'].values, y_test, y_pred):
        test_predictions.append({
            'country': country,
            'year': year,
            'actual': actual,
            'predicted': pred,
            'error': abs(actual - pred)
        })

# Create metrics DataFrame
metrics_df = pd.DataFrame(all_metrics)
test_pred_df = pd.DataFrame(test_predictions)

print("=== OVERALL ACCURACY METRICS ===")
print(f"Average MAE (Mean Absolute Error): {metrics_df['MAE'].mean():.2f}")
print(f"Average RMSE (Root Mean Squared Error): {metrics_df['RMSE'].mean():.2f}")
print(f"Average R² Score: {metrics_df['R2'].mean():.3f}")
print(f"\nNumber of countries evaluated: {len(metrics_df)}")

print("\n=== BEST PERFORMING COUNTRIES (Lowest MAE) ===")
print(metrics_df.nsmallest(10, 'MAE')[['country', 'MAE', 'RMSE', 'R2']])

print("\n=== WORST PERFORMING COUNTRIES (Highest MAE) ===")
print(metrics_df.nlargest(10, 'MAE')[['country', 'MAE', 'RMSE', 'R2']])

# Save metrics
metrics_path = os.path.join(output_folder, "prediction_accuracy_metrics.csv")
metrics_df.to_csv(metrics_path, index=False)
print(f"\nMetrics saved to: {metrics_path}")


# ==== VISUALIZATIONS ====

# 1. Distribution of errors
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(metrics_df['MAE'], bins=30, edgecolor='black')
plt.xlabel('Mean Absolute Error')
plt.ylabel('Number of Countries')
plt.title('Distribution of MAE Across Countries')
plt.axvline(metrics_df['MAE'].mean(), color='red', linestyle='--', label=f'Mean: {metrics_df["MAE"].mean():.2f}')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(metrics_df['R2'], bins=30, edgecolor='black')
plt.xlabel('R² Score')
plt.ylabel('Number of Countries')
plt.title('Distribution of R² Across Countries')
plt.axvline(metrics_df['R2'].mean(), color='red', linestyle='--', label=f'Mean: {metrics_df["R2"].mean():.3f}')
plt.legend()

plt.tight_layout()
plot_path = os.path.join(output_folder, "accuracy_distributions.png")
plt.savefig(plot_path)
plt.close()
print(f"Plot saved: {plot_path}")


# 2. Actual vs Predicted scatter plot
plt.figure(figsize=(8, 8))
plt.scatter(test_pred_df['actual'], test_pred_df['predicted'], alpha=0.3, s=10)
plt.plot([0, 100], [0, 100], 'r--', label='Perfect prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values (All Countries)')
plt.legend()
plt.grid(True, alpha=0.3)
plot_path = os.path.join(output_folder, "actual_vs_predicted.png")
plt.savefig(plot_path)
plt.close()
print(f"Plot saved: {plot_path}")


# 3. Example countries with predictions
countries_to_plot = ['United States', 'China', 'India', 'Germany']
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, country in enumerate(countries_to_plot):
    if idx >= len(axes):
        break
    
    country_data = df_long[df_long['country'] == country].sort_values('year')
    country_test = test_pred_df[test_pred_df['country'] == country]
    
    if not country_data.empty:
        ax = axes[idx]
        
        # Plot all historical data
        train_data = country_data[country_data['year'] < SPLIT_YEAR]
        test_data = country_data[country_data['year'] >= SPLIT_YEAR]
        
        ax.plot(train_data['year'], train_data['value_imputed'], 'o-', label='Training data', color='blue')
        ax.plot(test_data['year'], test_data['value_imputed'], 'o-', label='Actual (test)', color='green')
        
        if not country_test.empty:
            ax.plot(country_test['year'], country_test['predicted'], 's--', label='Predicted', color='red')
            mae = metrics_df[metrics_df['country'] == country]['MAE'].values[0]
            ax.set_title(f"{country}\nMAE: {mae:.2f}")
        else:
            ax.set_title(country)
        
        ax.set_xlabel('Year')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(output_folder, "example_predictions.png")
plt.savefig(plot_path)
plt.close()
print(f"Plot saved: {plot_path}")


print("\n=== ALL FILES SAVED TO FOLDER ===")
print(f"Check the '{output_folder}/' folder for:")
print("  - prediction_accuracy_metrics.csv")
print("  - accuracy_distributions.png")
print("  - actual_vs_predicted.png")
print("  - example_predictions.png")

print("\n=== INTERPRETATION GUIDE ===")
print("MAE (Mean Absolute Error): Average difference between actual and predicted")
print("  - Lower is better (0 = perfect)")
print(f"  - Your average: {metrics_df['MAE'].mean():.2f} percentage points")
print("\nRMSE (Root Mean Squared Error): Like MAE but penalizes large errors more")
print("  - Lower is better")
print(f"  - Your average: {metrics_df['RMSE'].mean():.2f}")
print("\nR² Score: How well the model fits the data")
print("  - Range: -∞ to 1.0")
print("  - 1.0 = perfect fit, 0 = no better than average, negative = worse than average")
print(f"  - Your average: {metrics_df['R2'].mean():.3f}")

