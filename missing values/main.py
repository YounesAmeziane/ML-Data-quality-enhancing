import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load the dataset
df = pd.read_csv("test.csv")

# Separate numerical and categorical columns
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

# Encode categorical features
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))  # Convert NaN to string to avoid errors
    label_encoders[col] = le  # Store for inverse transform if needed

# Function to check for multicollinearity (with NaN & Inf handling)
def check_multicollinearity(data, threshold=10):
    # Drop rows with NaN or Inf values before VIF calculation
    data_clean = data.replace([np.inf, -np.inf], np.nan).dropna()

    if data_clean.shape[0] == 0:  # If all rows are dropped, return
        print("⚠️ Not enough data after removing NaNs/Infs for VIF calculation.")
        return pd.DataFrame()

    vif_data = pd.DataFrame()
    vif_data["Feature"] = data_clean.columns
    vif_data["VIF"] = [variance_inflation_factor(data_clean.values, i) for i in range(data_clean.shape[1])]

    high_vif = vif_data[vif_data["VIF"] > threshold]
    if not high_vif.empty:
        print("\n⚠️ Warning: High multicollinearity detected!")
        print(high_vif)
    
    return high_vif


# Function to perform Random Forest Imputation
def random_forest_imputation(data, num_cols, cat_cols):
    df_imputed = data.copy()

    for col in df_imputed.columns:
        if df_imputed[col].isnull().sum() > 0:
            X_train = df_imputed.loc[df_imputed[col].notnull()].drop(col, axis=1)
            y_train = df_imputed.loc[df_imputed[col].notnull(), col]
            X_test = df_imputed.loc[df_imputed[col].isnull()].drop(col, axis=1)

            if X_train.shape[0] == 0:  # Skip if no training data
                print(f"⚠️ Skipping column '{col}' (all values are missing)")
                continue

            # Check if column is categorical or numerical
            if col in num_cols:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                model = RandomForestClassifier(n_estimators=100, random_state=42)

            model.fit(X_train, y_train)

            # Predict and fill missing values
            df_imputed.loc[df_imputed[col].isnull(), col] = model.predict(X_test)

    return df_imputed

# Perform multicollinearity check before imputation
check_multicollinearity(df[num_cols])


# Apply Random Forest Imputation
df_cleaned = random_forest_imputation(df, num_cols, cat_cols)

# Preserve original data types
df_cleaned[num_cols] = df_cleaned[num_cols].astype(df[num_cols].dtypes.to_dict())

# Decode categorical variables back (Optional)
for col in cat_cols:
    df_cleaned[col] = label_encoders[col].inverse_transform(df_cleaned[col].astype(int))

# Save cleaned dataset
df_cleaned.to_csv("synthetic_dataset_imputed.csv", index=False)

# Display the cleaned dataset
print(df_cleaned)
