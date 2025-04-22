import pandas as pd
import os

def duplication(data):
    
    try:
        df = data
        
        print("\nAvailable columns:")
        for i, col in enumerate(df.columns):
            print(f"{i + 1}. {col}")
        
        col_index = int(input("\nEnter the number of the column you want to analyze: ")) - 1
        if col_index < 0 or col_index >= len(df.columns):
            print("Invalid selection.")
            return
        
        selected_col = df.columns[col_index]
        
        value_counts = df[selected_col].value_counts().to_dict()
        
        print(f"\nValue counts for column '{selected_col}' (count > 1):")
        filtered_values = {value: count for value, count in value_counts.items() if count > 1}
        for value, count in filtered_values.items():
            print(f"{value}: {count}")
        
        choice = input("\nDo you want to keep only one occurrence of these values? (yes/no): ").strip().lower()
        if choice == 'yes':
            df = df[df[selected_col].isin(filtered_values.keys())]
            df = df.drop_duplicates(subset=[selected_col], keep='first')
            
            base_filename = os.path.basename(data)
            output_file = "filtered_" + base_filename
            
            df.to_csv(output_file, index=False)
            print(f"\nUpdated dataset saved as {output_file}")
    
    except FileNotFoundError:
        print("File not found. Please enter a valid file path.")
    except pd.errors.EmptyDataError:
        print("The file is empty.")
    except Exception as e:
        print(f"An error occurred: {e}")
