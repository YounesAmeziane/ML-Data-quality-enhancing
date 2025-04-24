# ML-Data-quality-enhancing
A Python-based toolkit designed to assess and improve the quality of datasets used in machine learning (ML) pipelines. By identifying and addressing common data issues, this tool aims to enhance model performance and reliability.

Developed by: **Younes Ameziane**  
Date: **November 2024**

---

## Table of Contents
- [Features](#features)
- [Project structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Testing](#testing)
- [Contributing](#contributing)

---

## Features
- **Data Quality Assessment:** Automatically detects issues such as missing values, duplicates, and inconsistent data types.
- **Data Cleaning:** Provides functionalities to clean and preprocess data, including handling missing values and removing duplicates.
- **Visualization:** Generates visual reports to help understand data quality issues and the impact of cleaning operations.
- **Modular Design:** Structured to allow easy integration of additional data quality checks and cleaning methods.

---


## Project structure
```
ML-Data-quality-enhancing/
├── data/                   # Directory for input datasets
├── anomaly/                # Anomaly directory
│   ├── anomaly.py          # File with the anomaly function
├── duplicate/              # Duplicate directory
│   ├── duplication.py      # File with the anomaly function
├── missingValues/          # Missing values imputation directory
│   ├── missing_values.py   # File with the anomaly function
├── env/                    # Virtual environement
│   ├── include/
│   ├── lib/
│   ├── Scripts/    
│   └── share    
├── main.py                 # File that manages the entire project, calls the functions, etc..
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Installation
Clone this repository to your local machine:
```bash
git clone https://github.com/yourusername/ML-Data-quality-enhancing.git
cd ML-Data-quality-enhancing
```

Create a Virtual Environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install Dependencies:
```
pip install -r requirements.txt
```















