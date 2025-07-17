import pandas as pd  # type: ignore
import os
from anomaly.anomaly import anomaly_detection
from duplicate.duplication import duplication
from missingValues.missing_values import missing_values
from sqlalchemy import create_engine
from dotenv import load_dotenv


load_dotenv()

print('Hello and welcome to the Data Quality Project (Beta)')

server_name = os.getenv('SERVER_NAME')
database_name = 'hrdm_dev'
engine = create_engine(
    f"mssql+pyodbc://{server_name}/{database_name}?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes&MARS_Connection=yes"
)
sql = input("Enter SQL query: ")
query = f"""{sql}"""
data = pd.read_sql(query, engine)
print(data.head())
engine.dispose()

def get_data(df):
    columns = df.shape[1]
    rows = df.shape[0]
    total_missing = df.isnull().sum().sum()
    return {
        'columns': columns,
        'rows': rows,
        'total_missing': total_missing
    }

print('Data summary:')
summary = get_data(data)
print(f"Columns: {summary['columns']}, Rows: {summary['rows']}, Total Missing Values: {summary['total_missing']}")

# Select the feature to be used on the dataset
features = ['Anomaly detection', 'Duplication check', 'Missing values detection and imputation']

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
    exit()
except IndexError:
    print("Invalid selection. Please enter a number from the list.")
    exit()

if feature_selected == features[0]:
    anomaly_detection(data)
elif feature_selected == features[1]:
    duplication(data, None)  # file_path not needed
elif feature_selected == features[2]:
    missing_values(data)
else:
    print('Invalid feature selection.')