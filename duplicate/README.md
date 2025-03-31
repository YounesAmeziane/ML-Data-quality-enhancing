# CSV Column Analysis Script

Developed by: **Younes Ameziane**  
Date: **March 2025**


---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Error Handling](#error-handling)
- [Example Output](#example-output)

---

## Overview
This Python script allows users to analyze a CSV file by selecting a specific column. It provides a count of each unique value in the selected column and allows users to filter out all but one occurrence of values that appear more than once.)

---

## Features
-Reads a CSV file specified by the user.
-Lists all available columns for selection.
-Displays the count of each unique value (excluding those that appear only once).
-Provides an option to retain only one occurrence of values with duplicates.
-Saves the filtered dataset as a new CSV file.

---

## Requirements
-Python 3.x
-pandas
-os

---

## Installation
Ensure you have Python installed, then install pandas using:
```bash
pip install pandas
```

---

## Usage
1. Run the script
```bash
python main.py
```
2. python script.py
3. Enter the path to the CSV file when prompted.
4. Choose the column you want to analyze by entering its corresponding number.
5. View the count of each value (only values appearing more than once are displayed).
6. Decide whether to keep only one occurrence of the duplicated values.
7. If chosen, a filtered CSV file will be saved with the prefix `filtered_`

---

## Error Handling
-If the file is not found, the script will notify the user.
-If the file is empty, the script will display an appropriate message.
-If an unexpected error occurs, the script will print the error message.

---

## Example Output
```bash
Enter the CSV file path: data.csv

Available columns:
1. Name
2. Age
3. City

Enter the number of the column you want to analyze: 2

Value counts for column 'Age' (count > 1):
25: 3
30: 2
45: 4

Do you want to keep only one occurrence of these values? (yes/no): yes

Updated dataset saved as filtered_data.csv
```





