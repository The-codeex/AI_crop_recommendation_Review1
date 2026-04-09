# last working code using old dataset 


import pandas as pd

# Load datasets
yield_df = pd.read_csv("data/clean_yield_data.csv")
rain_df = pd.read_csv("data/rainfall_data.csv")
temp_df = pd.read_csv("data/temperature_data.csv")

print("Yield dataset:", yield_df.shape)
print("Rainfall dataset:", rain_df.shape)
print("Temperature dataset:", temp_df.shape)

# Rename columns
yield_df = yield_df.rename(columns={
    "State_Name": "State",
    "Crop_Year": "Year"
})

rain_df = rain_df.rename(columns={"YEAR": "Year"})
temp_df = temp_df.rename(columns={"YEAR": "Year", "ANNUAL": "Temperature"})

# Select useful columns
rain_df = rain_df[["Year","ANNUAL"]]
rain_df = rain_df.rename(columns={"ANNUAL":"Rainfall"})

temp_df = temp_df[["Year","Temperature"]]
# Force numeric types  ← fixes your error
rain_df["Rainfall"]    = pd.to_numeric(rain_df["Rainfall"],    errors="coerce")
temp_df["Temperature"] = pd.to_numeric(temp_df["Temperature"], errors="coerce")

# Merge datasets
df = yield_df.merge(rain_df, on="Year", how="left")
df = df.merge(temp_df, on="Year", how="left")

# Handle missing values
df = df.dropna()
df["Rainfall"] = df["Rainfall"].fillna(df["Rainfall"].mean())
df["Temperature"] = df["Temperature"].fillna(df["Temperature"].mean())
df = df[df["Rainfall"] >= 0]
df = df[df["Temperature"] > 0]

# Create Yield column
# Area: The total land area (in hectares) under cultivation for the specific crop.
# Production: The quantity of crop production (in metric tons).
df["Yield"] = df["Production"] / df["Area"]
# metric tons per hectare for Yield

# -------------------------------
# Outlier Removal on Yield (IQR)
# -------------------------------
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

df = remove_outliers(df, "Yield")


# Check for remaining missing values
print(df.isnull().sum())
print("Cleaned shape:", df.shape)
# Save dataset
df.to_csv("data/final_dataset.csv", index=False)

print("Final dataset created successfully")






















