import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
from tqdm import tqdm
import warnings

def missing_values(df, output_path="synthetic_dataset_imputed.csv", vif_threshold=10):
    """
    Handle missing values in a DataFrame using Random Forest imputation,
    with multicollinearity checking and categorical encoding preservation.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame with missing values
        output_path (str): Path to save the cleaned dataset
        vif_threshold (float): Threshold for multicollinearity detection
        
    Returns:
        pd.DataFrame: DataFrame with imputed values
    """
    # Suppress sklearn warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # Input validation
    if df.empty:
        print("⚠️ Empty DataFrame provided!")
        return df
        
    print("\n🔍 Initial Data Overview:")
    print(f"• Rows: {len(df)}")
    print(f"• Columns: {len(df.columns)}")
    print(f"• Total missing values: {df.isnull().sum().sum()}\n")
    
    # Let user select columns to process
    print("Available columns:")
    for i, col in enumerate(df.columns):
        print(f"{i+1}. {col} (Missing: {df[col].isnull().sum()})")
    
    selected = input("\nEnter column numbers to process (comma-separated, or 'all'): ")
    if selected.lower() != 'all':
        try:
            cols = [df.columns[int(i)-1] for i in selected.split(',')]
            df = df[cols]
        except (ValueError, IndexError):
            print("⚠️ Invalid selection. Processing all columns.")
    
    # Prepare data structures
    original_dtypes = df.dtypes.to_dict()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    label_encoders = {}

    # Step 1: Encode Categorical Variables
    print("\nEncoding categorical columns...")
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            # Add special category for missing values before encoding
            df[col] = df[col].astype(str).replace('nan', 'MISSING')
        
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    print("✅ Encoding complete.")

    # Step 2: Random Forest Imputation
    print("\nStarting Random Forest Imputation...")
    df_imputed = df.copy()
    
    for col in tqdm(df_imputed.columns, desc="Imputing"):
        if df_imputed[col].isnull().sum() > 0:
            try:
                # Prepare training data
                X_train = df_imputed[df_imputed[col].notnull()].drop(col, axis=1)
                y_train = df_imputed.loc[df_imputed[col].notnull(), col]
                X_test = df_imputed[df_imputed[col].isnull()].drop(col, axis=1)

                if len(X_train) == 0:
                    print(f"⚠️ Column '{col}' skipped - no valid training data")
                    continue

                # Select appropriate model
                model = (RandomForestRegressor(n_estimators=100, random_state=42) 
                         if col in num_cols 
                         else RandomForestClassifier(n_estimators=100, random_state=42))

                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                df_imputed.loc[df_imputed[col].isnull(), col] = preds
                
            except Exception as e:
                print(f"⚠️ Failed to impute '{col}': {str(e)}")
                continue

    print("\n🎯 Imputation complete.")
    print(f"• Remaining missing values: {df_imputed.isnull().sum().sum()}")

    # Step 3: Restore Original Data Types and Labels
    print("\nRestoring original formats...")
    for col, dtype in original_dtypes.items():
        if col in df_imputed.columns:
            if col in cat_cols:
                # Decode categorical columns
                le = label_encoders.get(col)
                if le:
                    df_imputed[col] = le.inverse_transform(df_imputed[col].astype(int))
                    # Convert back to original dtype if possible
                    try:
                        df_imputed[col] = df_imputed[col].astype(dtype)
                    except (ValueError, TypeError):
                        pass
            else:
                # Convert numerical columns
                df_imputed[col] = df_imputed[col].astype(dtype)

    # Step 4: Multicollinearity Check (VIF)
    print("\nChecking multicollinearity...")
    numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) >= 2:
        data_clean = df_imputed[numeric_cols].replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(data_clean) >= 2:
            try:
                vif_data = pd.DataFrame()
                vif_data["Feature"] = data_clean.columns
                
                # Calculate VIF for each feature
                vif_list = []
                for i in range(min(data_clean.shape[1], data_clean.shape[0]-1)):
                    try:
                        vif = variance_inflation_factor(data_clean.values, i)
                        vif_list.append(vif)
                    except:
                        vif_list.append(np.nan)
                
                vif_data["VIF"] = vif_list
                high_vif = vif_data[vif_data["VIF"] > vif_threshold]
                
                if not high_vif.empty:
                    print("⚠️ High multicollinearity detected:")
                    print(high_vif.sort_values("VIF", ascending=False))
                else:
                    print("✅ No significant multicollinearity found.")
            except Exception as e:
                print(f"⚠️ VIF calculation failed: {str(e)}")
        else:
            print("⚠️ Not enough clean data for VIF calculation")
    else:
        print("⚠️ Not enough numeric columns for VIF analysis")

    # Step 5: Save Results
    try:
        df_imputed.to_csv(output_path, index=False)
        print(f"\nCleaned data saved to: {output_path}")
    except Exception as e:
        print(f"⚠️ Failed to save results: {str(e)}")

    print("\n Missing value processing complete!")
    return df_imputed