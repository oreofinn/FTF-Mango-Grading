# regenerate_inflation_files.py

import pandas as pd

# 1. Point this at your downloaded PSA file
INPUT_XLSX = "Statistical Tables on March 2025 CPI for All Income Households (2018=100)_s4y2t.xlsx"

# 2. Read the sheet with historical inflation (Table 14)
xls = pd.ExcelFile(INPUT_XLSX)
df14 = xls.parse("table 14", header=None)

# 3. Extract �Year� (col 0) and �Ave� (last column, col 25)
inflation = df14.iloc[7:, [0, 25]].copy()
inflation.columns = ["Year", "Inflation_Rate"]

# 4. Clean up and cast types
inflation = inflation[pd.to_numeric(inflation["Year"], errors="coerce").notnull()]
inflation["Year"] = inflation["Year"].astype(int)
inflation = inflation[pd.to_numeric(inflation["Inflation_Rate"], errors="coerce").notnull()]
inflation["Inflation_Rate"] = inflation["Inflation_Rate"].astype(float)

# 5. Export
inflation.to_csv("inflation_rates_1994_2025.csv", index=False)
inflation.to_json("inflation_rates_1994_2025.json", orient="records", indent=2)

print("Files generated:")
print(" � inflation_rates_1994_2025.csv")
print(" � inflation_rates_1994_2025.json")
