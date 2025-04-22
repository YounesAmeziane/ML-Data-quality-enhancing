import pandas as pd
import os
from anomaly.anomaly import anomaly_detection
from duplicate.duplication import duplication
from missingValues.missing_values import missing_values

print('Hello and welcome to the Data Quality Project (Beta)')

while True:
    data = input('\nPlease enter the dataset name (must be existing in the "data" directory): \n:::')
    file_path = os.path.join('data', data)

    if os.path.isfile(file_path):
        print(f"File '{data}' found! Loading...")
        data = pd.read_csv(file_path)
        print('Data loaded succesfully.\n')
        print('Here is the head of the file: ')
        print(data.head())
        break
    else:
        print('File does not exist. Please enter an existing file name.')

# Select the feature to be used on the dataset

features = ['Anomaly detection', 'Duplication check', 'Missing values detection and impotation']

if 'data' in locals():
    print('Services available:\n')
    for index, service in enumerate(features):
        print(f'{index + 1}. {service}')
    try:
        feature_number_input = input("Enter the number of the service you want: ")
        feature_index = int(feature_number_input) - 1
        feature_selected = features[feature_index]
        print(f"You selected: {feature_selected}")
    except ValueError:
        print("Invalid input. Please enter a number.")
    except IndexError:
        print("Invalid selection. Please enter a number from the list.")

if feature_selected == features[0]:
    anomaly_detection(data)
elif feature_selected == features[1]:
    duplication(data)
elif feature_selected == features[2]:
    missing_values(data)
else:
    print('Invalid feature selection.')