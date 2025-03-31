import pandas as pd
import os

def main():
    # Ask user for CSV file path
    file_path = 'your_file'
    
    try:
        # Read CSV file
        df = pd.read_csv(file_path)
        
        # Display available columns
        print("\nAvailable columns:")
        for i, col in enumerate(df.columns):
            print(f"{i + 1}. {col}")
        
        # Ask user to select a column
        col_index = int(input("\nEnter the number of the column you want to analyze: ")) - 1
        if col_index < 0 or col_index >= len(df.columns):
            print("Invalid selection.")
            return
        
        selected_col = df.columns[col_index]
        
        # Count unique values
        value_counts = df[selected_col].value_counts().to_dict()
        
        # Display results (only values with count > 1)
        print(f"\nValue counts for column '{selected_col}' (count > 1):")
        filtered_values = {value: count for value, count in value_counts.items() if count > 1}
        for value, count in filtered_values.items():
            print(f"{value}: {count}")
        
        # Ask user if they want to keep only one occurrence of these values
        choice = input("\nDo you want to keep only one occurrence of these values? (yes/no): ").strip().lower()
        if choice == 'yes':
            df = df[df[selected_col].isin(filtered_values.keys())]
            df = df.drop_duplicates(subset=[selected_col], keep='first')
            
            # Extract filename from path and create new filename
            base_filename = os.path.basename(file_path)
            output_file = "filtered_" + base_filename
            
            # Save modified dataframe
            df.to_csv(output_file, index=False)
            print(f"\nUpdated dataset saved as {output_file}")
    
    except FileNotFoundError:
        print("File not found. Please enter a valid file path.")
    except pd.errors.EmptyDataError:
        print("The file is empty.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
