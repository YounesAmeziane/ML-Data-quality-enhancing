# Anomaly Detection with Isolation Forest
This Python script performs anomaly detection using the Isolation Forest algorithm from `scikit-learn`. It provides an interactive interface to let users drop irrelevant columns, preprocesses the data, detects anomalies, and visualizes results using **Seaborn** and **Matplotlib**.

---

## Features

- **Flexible Data Input:** Accepts a structured dataset in dictionary or DataFrame format.
- **Interactive Column Selection:** Users can choose which columns to drop before modeling.
- **Standardized Preprocessing:** Applies `StandardScaler` for normalization.
- **Isolation Forest Algorithm:** Detects outliers with customizable contamination.
- **Anomaly Scoring:** Ranks anomalies by severity.
- **Visualizations:**
    - Pair plots for anomaly comparison.
    - Histogram of anomaly scores.


## Requirements
*See requirememnts.txt*
- Python 3.6+
- pandas
- scikit-learn
- seaborn
- matplotlib


## Output

- **DataFrame** output showing:
    - Anomaly score for each row
    - Whether the row is labeled as `Normal` or `Anomaly`
- **Pair plot** colored by anomaly label
- **Histogram** showing the distribution of anomaly scores


## How It Works
- Isolation Forest isolates observations by randomly selecting a feature and splitting it randomly between the max and min values of that feature.
- Anomalies are isolated faster and hence get a lower anomaly score.
- A decision function outputs an anomaly score, which is used to label and rank the data.


## Notes
- The default contamination rate is `0.05` (5% of data assumed anomalous), the higher the percentege the stricter the decision will be
- Modify the script if you want to automate or suppress the interactive input process.










