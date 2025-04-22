import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

def missing_values(data):
    try:
        df = pd.DataFrame(data)
    except FileNotFoundError:
        print(f"Error: the file {data} not found. Please make sure the file exists.")
        exit()

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Encode categorical features
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Function to check for multicollinearity
    def check_multicollinearity(data, threshold=10):
        data_clean = data.replace([np.inf, -np.inf], np.nan).dropna()
        if data_clean.shape[0] == 0:
            print("⚠️ Not enough data after removing NaNs/Infs for VIF calculation.")
            return pd.DataFrame()
        vif_data = pd.DataFrame()
        vif_data["Feature"] = data_clean.columns
        vif_data["VIF"] = [variance_inflation_factor(data_clean.values, i) for i in range(data_clean.shape[1])]
        high_vif = vif_data[vif_data["VIF"] > threshold]
        if not high_vif.empty:
            print("\n⚠️ Warning: High multicollinearity detected in numerical features!")
            print(high_vif)
        return high_vif

    # Function for Random Forest Imputation
    def random_forest_imputation(data, num_cols, cat_cols, evaluate=False, original_data=None):
        df_imputed = data.copy()
        imputation_scores = {}
        print("\nStarting Random Forest Imputation...")
        for col in df_imputed.columns:
            if df_imputed[col].isnull().sum() > 0:
                print(f"Imputing missing values for column: {col}")
                X_train = df_imputed.loc[df_imputed[col].notnull()].drop(col, axis=1)
                y_train = df_imputed.loc[df_imputed[col].notnull(), col]
                X_test = df_imputed.loc[df_imputed[col].isnull()].drop(col, axis=1)
                missing_indices = df_imputed.loc[df_imputed[col].isnull()].index

                if X_train.shape[0] == 0:
                    print(f"⚠️ Skipping column '{col}' (all values are missing)")
                    continue

                if col in num_cols:
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                else:
                    model = RandomForestClassifier(n_estimators=100, random_state=42)

                model.fit(X_train, y_train)
                predicted_values = model.predict(X_test)
                df_imputed.loc[missing_indices, col] = predicted_values

                if evaluate and original_data is not None and col in original_data.columns:
                    original_missing_values = original_data.loc[missing_indices, col]
                    # Checking if there are any NaNs in the original missing values before calculating MSE
                    if original_missing_values.isnull().any():
                        print(f"⚠️ Warning: Original data for column '{col}' at missing indices contains NaN. Skipping MSE calculation for this column.")
                    elif col in num_cols:
                        mse = mean_squared_error(original_missing_values, predicted_values)
                        imputation_scores[col] = f"MSE: {mse:.4f}"
                    else:
                        predicted_original = label_encoders.get(col, LabelEncoder()).inverse_transform(predicted_values.astype(int))
                        original_missing_original = original_data.loc[missing_indices, col].astype(str).values
                        accuracy = accuracy_score(original_missing_original, predicted_original)
                        imputation_scores[col] = f"Accuracy: {accuracy:.4f}"

        print("Random Forest Imputation complete.")
        return df_imputed, imputation_scores

    # Check for multicollinearity in numerical features before imputation
    print("Checking for multicollinearity in original numerical features...")
    high_vif_features_original = check_multicollinearity(df[num_cols].copy())

    df_original = df.copy()
    if 'numerical_column_with_no_nans' in df.columns:
        mask = np.random.choice([True, False], size=df.shape[0], p=[0.1, 0.9])
        df.loc[mask, 'numerical_column_with_no_nans'] = np.nan
    if 'categorical_column_with_no_nans' in df.columns and 'categorical_column_with_no_nans' in label_encoders:
        mask_cat = np.random.choice([True, False], size=df.shape[0], p=[0.1, 0.9])
        df.loc[mask_cat, 'categorical_column_with_no_nans'] = np.nan

    # Perform Random Forest Imputation
    df_cleaned, imputation_scores = random_forest_imputation(df.copy(), num_cols, cat_cols, evaluate=True, original_data=df_original)

    original_num_dtypes = df_original[num_cols].dtypes.to_dict()
    for col in num_cols:
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].astype(original_num_dtypes.get(col, 'float64'))

    print("\nInverse transforming categorical features...")
    for col in cat_cols:
        if col in df_cleaned.columns:
            df_cleaned[col] = label_encoders[col].inverse_transform(df_cleaned[col].astype(int))
    print("Categorical feature inverse transformation complete.")

    # Check for multicollinearity after imputation
    print("\nChecking for multicollinearity in numerical features after imputation...")
    high_vif_features_imputed = check_multicollinearity(df_cleaned[num_cols].copy())

    print("\nGenerating visualizations to compare distributions before and after imputation...")
    for col in df_original.columns:
        if df_original[col].isnull().sum() > 0:
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            sns.histplot(df_original[col], kde=True)
            plt.title(f'Original Distribution of {col} (with NaNs)')
            plt.subplot(1, 2, 2)
            sns.histplot(df_cleaned[col], kde=True)
            plt.title(f'Distribution of {col} After Imputation')
            plt.tight_layout()
            plt.show()

    # Print imputation scores
    if imputation_scores:
        print("\n--- Imputation Evaluation Scores ---")
        for col, score in imputation_scores.items():
            print(f"Column: {col}, Score: {score}")

    print("\nSaving the imputed dataset to 'synthetic_dataset_imputed.csv'")
    df_cleaned.to_csv("synthetic_dataset_imputed.csv", index=False)

    print("\n✅ --- Complete ---✅")
    print("📂Data preprocessing and imputation finished. The imputed dataset is saved to 'synthetic_dataset_imputed.csv'.")
    if not high_vif_features_original.empty:
        print("\n⚠️ High multicollinearity was detected in the original numerical features. Consider addressing this before further modeling.")
    if not high_vif_features_imputed.empty:
        print("\n⚠️ High multicollinearity was detected in the numerical features after imputation. The imputation process might have influenced multicollinearity.")
