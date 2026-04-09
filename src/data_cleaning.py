

import pandas as pd

df = pd.read_csv("data/yield_data.csv")

# -------------------------------
# Handle Missing Values
# -------------------------------
df = df.dropna()
df = df[df["Area"] > 0]
df = df[df["Production"] > 0]

df["Area"].fillna(df["Area"].median(), inplace=True)
df["Production"].fillna(df["Production"].median(), inplace=True)

# -------------------------------
# Outlier Removal (IQR)
# -------------------------------

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    return df[(df[column] >= lower) & (df[column] <= upper)]

df = remove_outliers(df, "Area")
df = remove_outliers(df, "Production")

# -------------------------------
# Save cleaned data
# -------------------------------

df.to_csv("data/clean_yield_data.csv", index=False)

print("Data cleaned successfully")




















































