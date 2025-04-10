# train_srp_model.py
"""
Train a linear regression model to predict Suggested Retail Price (SRP) per kg of mangoes

Required files:
  - final_dataset.csv         # Combined dataset with columns: year, month (e.g. 'March'), location, mango_type, grade (A/B/C/Rejected), base_price, inflation_rate, srp
  - grades.json               # JSON mapping grades to numeric multipliers, e.g. {"A":1.0, "B":0.8, "C":0.6, "Rejected":0.2}
Optional files for extended analysis:
  - production_costs.csv      # Columns: location, estimated_cost_per_kg

Dependencies:
  pandas, scikit-learn, joblib
Install via: pip install pandas scikit-learn joblib
"""
import pandas as pd
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --- Step 1: Load data ---
# Combined dataset (should contain target 'srp')
df = pd.read_csv("final_dataset.csv")
# Grade multipliers
with open("grades.json") as f:
    grade_map = json.load(f)

# Optional: load production costs
try:
    prod_costs = pd.read_csv("production_costs.csv")
    df = df.merge(prod_costs, on="location", how="left")
    # now 'estimated_cost_per_kg' is available
    include_cost = True
except FileNotFoundError:
    include_cost = False

# --- Step 2: Feature engineering ---
# Convert month name to numeric (1-12)
df['month'] = pd.to_datetime(df['month'], format='%B').dt.month
# Map grade to numeric multiplier
df['grade'] = df['grade'].map(grade_map)

# Optional: compute margin over cost
if include_cost:
    df['margin_per_kg'] = df['base_price'] - df['estimated_cost_per_kg']
    feature_cols = ['year','month','grade','base_price','inflation_rate','estimated_cost_per_kg']
else:
    feature_cols = ['year','month','grade','base_price','inflation_rate']

target_col = 'srp'

# Drop rows with missing values
df = df.dropna(subset=feature_cols + [target_col])

X = df[feature_cols]
y = df[target_col]

# --- Step 3: Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 4: Train model ---
model = LinearRegression()
model.fit(X_train, y_train)

# --- Step 5: Evaluate ---
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Model performance:")
print(f"  Mean Squared Error: {mse:.2f}")
print(f"  R^2 Score: {r2:.3f}")
print(f"  Coefficients: {dict(zip(feature_cols, model.coef_))}")
print(f"  Intercept: {model.intercept_:.2f}")

# --- Step 6: Save model ---
joblib.dump(model, 'srp_model.pkl')
print("Trained model saved to srp_model.pkl")
