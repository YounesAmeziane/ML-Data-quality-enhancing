import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def anomaly_detection(data):
    df = pd.DataFrame(data)

    print('Choose which columns are not relevant to drop them. When you are done, enter "done".')

    while True:
        print("\nCurrent columns:")
        for i in df.columns:
            print(f"- {i}")
        
        drop_col = input("\nSelect a column to drop (or type 'done' to finish): ").strip()
        
        if drop_col.lower() == 'done':
            break
        elif drop_col in df.columns:
            df = df.drop(columns=[drop_col])
            print(f"Column '{drop_col}' dropped.\n")
        else:
            print(f"Column '{drop_col}' not found. Please check the name.\n")

    print("\nFinal columns:")
    for i in df.columns:
        print(f"- {i}")

    non_feature_columns = ['anomaly_score', 'anomaly_label']
    features = [col for col in df.columns if col not in non_feature_columns]

    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    anomaly_scores = model.fit_predict(X_scaled)

    df['anomaly_score'] = model.decision_function(X_scaled)

    df['anomaly_label'] = ['Anomaly' if score == -1 else 'Normal' for score in anomaly_scores]

    print('Results with Anomaly Scores:')
    print(df[['anomaly_label', 'anomaly_score'] + features].head())

    ranked_anomalies = df[df['anomaly_label'] == 'Anomaly'].sort_values(by='anomaly_score')
    print('\nRanked Anomalies (most anomalous first):')
    print(ranked_anomalies[['anomaly_label', 'anomaly_score'] + features])

    anomaly_count = (df['anomaly_label'] == 'Anomaly').sum()
    print(f'Total number of anomalies: {anomaly_count}')

    sns.pairplot(
        df,
        hue='anomaly_label',
        vars=features,
        palette={'Normal': 'blue', 'Anomaly': 'red'},
    )
    plt.suptitle('Anomaly Detection with Isolation Forest and Scores', y=1.02)
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.histplot(df['anomaly_score'], kde=True)
    plt.title('Distribution of Isolation Forest Anomaly Scores')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.show()
