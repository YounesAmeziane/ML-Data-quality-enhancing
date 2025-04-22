# Missing Value Imputation & Multicollinearity Detection using Random Forest
Younes Ameziane Feb 2025

This project handles missing values in a dataset using Random Forest imputation and detects multicollinearity using Variance Inflation Factor (VIF). The script ensures data consistency, warns about potential issues, and preserves categorical and numerical features properly.

## Features
- ✅ Handles Missing Data: Uses Random Forest Regression for numerical values and Random Forest Classification for categorical values.
- ✅ Detects Multicollinearity: Uses Variance Inflation Factor (VIF) to warn about highly correlated features.
- ✅ Encodes and Decodes Categorical Data: Uses Label Encoding for categorical columns before imputation and reverses encoding afterward.
- ✅ Preserves Data Types: Ensures original numerical and categorical types remain intact after processing.
- ✅ Includes Warnings & Recommendations: Notifies the user about highly correlated features and missing columns.

---

## 📂 Project Structure
```
📂 missing_values_project
│── 📄 dataset.csv              # Input dataset (must be provided)
│── 📄 main.py                  # Python script for imputation & VIF check
│── 📄 synthetic_dataset_imputed.csv  # Output dataset after processing
│── 📄 README.md                # Project documentation
```

---

## 🛠️ Installation & Setup
### 1️⃣ Install Required Dependencies
Make sure you have Python 3.7+ installed, then run:
```
pip install pandas numpy scikit-learn statsmodels
```
2️⃣ Place Your Dataset

Place your dataset in the same folder as `main.py` and name it `dataset.csv`.
3️⃣ Run the Script
```
python main.py
```

---

## 🚀 How It Works
**Step 1️⃣: Load & Preprocess Data**
- Reads `dataset.csv`
- Identifies numerical and categorical columns
- Applies Label Encoding to categorical variables

**Step 2️⃣: Check for Multicollinearity (VIF)**
- Computes Variance Inflation Factor (VIF)
- If VIF > 10, prints a warning
- Suggests which features to remove

**Step 3️⃣: Perform Random Forest Imputation**
- Numerical columns → Uses Random Forest Regressor
- Categorical columns → Uses Random Forest Classifier
- Fills in missing values based on learned patterns

**Step 4️⃣: Decode Categorical Data & Save Output**
- Restores original categorical values
- Saves cleaned dataset as `synthetic_dataset_imputed.csv`

---

## 📖 Understanding Key Concepts
**1️⃣ What is Variance Inflation Factor (VIF)?**

VIF detects if features are too similar to each other (highly correlated).
- VIF > 10 → High multicollinearity (problematic, redundant data)
- VIF 5-10 → Moderate correlation (consider reviewing)
- VIF < 5 → Low correlation (good!)

**2️⃣ What is Random Forest Imputation?**
- Uses Random Forest models to predict and fill missing values
- Numerical values → Predicted using RandomForestRegressor
- Categorical values → Predicted using RandomForestClassifier

**3️⃣ Warnings You May See**
| Warning | Meaning | Solution |
|---------|---------|----------|
| ⚠️ **High Multicollinearity Detected!** | Some features are too correlated (VIF > 10) | Remove one of the correlated features |
| ⚠️ **Skipping column `{col}` (all values are missing)** | A column is completely empty | Drop the column or fill it manually |
| ⚠️ **Not enough data for VIF calculation** | Data contains too many NaN/Inf values | Remove missing values before running |


## 📊 Example Output
After running `main.py`, you'll get a cleaned dataset:
```
Name, Age, Salary, City
Alice, 25, 50000, New York
Bob, 30, 60000, San Francisco
Charlie, 28, 55000, Chicago
```
Original missing values have been imputed using Random Forest models.

---

## 🛠️ Customization
### Adjust the VIF Threshold

If you want to change the VIF threshold (default is 10), modify this line in `main.py`:
```
check_multicollinearity(df[num_cols], threshold=5)  # Stricter check
```
### Adjust Random Forest Parameters

Modify the number of trees (`n_estimators`) in `main.py`:
```
model = RandomForestRegressor(n_estimators=200, random_state=42)  # More trees for better accuracy
```

## 📜 License
This project is **open-source** under the **MIT** License.
Feel free to modify and use it as needed! 🚀
