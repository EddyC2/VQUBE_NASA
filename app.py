import pandas as pd 
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import os 


predictions_list = []
sentinels = [-8888, -9999]    # add any other sentinel values you know
MIN_YEARS = 6                 # require at least this many valid years to fit a country model
future_years = np.arange(2021, 2031)   # adjust as needed
EPS = 1e-6 


#loading the data using

df = pd.read_csv("CDA_ind.csv")
ghp = pd.read_csv("GHP_ind.csv")
par= pd.read_csv("PAR_ind.csv")

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
#getting the year number from the column names
df_long["year"] = df_long["year"].str.extract(r'(\d{4})').astype(int)

ghp_long["year"] = ghp_long["year"].str.extract(r'(\d{4})').astype(int)

par_long["year"] = par_long["year"].str.extract(r'(\d{4})').astype(int)
#clean data
df_long.loc[df_long['value'].isin(sentinels), 'value'] = np.nan

#impute missing data using linear interpolation
df_long["value_imputed"] = (
    df_long.groupby("country")["value"]
    .transform(lambda s: s.interpolate(method="linear", limit_direction="both"))
)


suspicious = df_long[(df_long['value'] > 100) | (df_long['value'] < 0)]
print("Suspicious rows (value >100 or <0):", suspicious.shape)

print(suspicious.head(10))

#clean data using sentinel values

df_long.loc[df_long['value'] > 100, 'value'] = np.nan
df_long.loc[df_long['value'] < 0, 'value'] = np.nan


#Missing values are filled with global average for that year
global_by_year = df_long.groupby('year')['value_imputed'].transform('mean')
df_long['value_imputed'] = df_long['value_imputed'].fillna(global_by_year)


def to_logit_percent(p_array):
    p = np.clip(p_array / 100.0, EPS, 1 - EPS)   # map to (0,1)
    return np.log(p / (1 - p))

def from_logit_percent(logit_vals):
    p = 1 / (1 + np.exp(-logit_vals))
    return np.clip(p * 100.0, 0.0, 100.0)

#for each country: Not enough data, uses last known value for future years
#Enough data: fits a linear regression model
#Clipping ensures predictions are between 0-100%


for country, grp in df_long.groupby('country'):
    grp = grp.sort_values('year')
    valid = grp.dropna(subset=['value_imputed'])
    years = valid['year'].values
    vals = valid['value_imputed'].values

    if len(valid) == 0:
        # no data -> predict NaN (or fallback to global mean by year)
        preds = np.array([np.nan]*len(future_years))
    elif len(valid) < MIN_YEARS:
        # not enough points -> fallback: use last observed value (or global mean)
        last_val = vals[-1]
        preds = np.array([last_val]*len(future_years))
    else:
        # Option A: simple linear regression on raw percent (easy)
        X = years.reshape(-1,1)
        y = vals
        model = LinearRegression()
        model.fit(X, y)
        preds = model.predict(future_years.reshape(-1,1))
        preds = np.clip(preds, 0.0, 100.0)

        # Option B (recommended for bounded percent): logit-transform, fit, inverse
        # Uncomment these lines to use the logit approach instead of Option A.
        # y_logit = to_logit_percent(vals)
        # model = LinearRegression()
        # model.fit(X, y_logit)
        # preds_logit = model.predict(future_years.reshape(-1,1))
        # preds = from_logit_percent(preds_logit)

    predictions_list.append(pd.DataFrame({
        'country': country,
        'year': future_years,
        'predicted_value': preds
    }))

predictions_df = pd.concat(predictions_list, ignore_index=True)


#checks if any prediction escpaed 0-100
bad = predictions_df[(predictions_df['predicted_value'] < 0) | (predictions_df['predicted_value'] > 100)]
print("Any bad predictions after clipping:", bad.shape)
print(bad.head())


for c in ['Albania', 'United States', 'China']:
    hist = df_long[df_long['country']==c].sort_values('year')
    pred = predictions_df[predictions_df['country']==c]
    if not hist.empty:
        plt.plot(hist['year'], hist['value_imputed'], label=f"{c} hist")
    if not pred.empty:
        plt.plot(pred['year'], pred['predicted_value'], linestyle='--', label=f"{c} pred")
plt.xlabel("Year"); plt.ylabel("Value (0-100)"); plt.legend(); plt.show()

predictions_df.to_csv("world_predictions_2030.csv", index=False)

print(f"Predictions saved successfully to!")

#print(df_long.head())
#print(ghp_long.head())``

#print(par_long.head())

#print(predictions_df[predictions_df["year"] == 2030].sort_values(by="predicted_value", ascending=False))
'''
countries_to_plot = ["United States", "China", "India"]
for c in countries_to_plot:
    subset = predictions_df[predictions_df["country"] == c]
    plt.plot(subset["year"], subset["predicted_value"], label=c)

plt.title("Predicted Environmental Values by Country")
plt.xlabel("Year")
plt.ylabel("Predicted Value")
plt.legend()
plt.show()
'''

