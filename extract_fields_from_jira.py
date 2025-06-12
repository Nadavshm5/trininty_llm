import pandas as pd
import re
import os

def process_description_column(input_csv, output_csv):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_csv)

    # Define regular expressions to extract the required information
    log_lines_pattern = re.compile(r'Log Lines\|{color:#172b4d}(\d+) - (\d+)')
    log_path_pattern = re.compile(r'Local-Time Log Path\|(.+?)\|')

    # Initialize new columns
    df['start line'] = None
    df['end line'] = None
    df['log path'] = None

    # Process each row in the DataFrame
    for index, row in df.iterrows():
        description = row['Description']

        # Extract start line and end line from "Log Lines"
        log_lines_match = log_lines_pattern.search(description)
        if log_lines_match:
            df.at[index, 'start line'] = int(log_lines_match.group(1))
            df.at[index, 'end line'] = int(log_lines_match.group(2))

        # Extract log path from "Local-Time Log Path"
        log_path_match = log_path_pattern.search(description)
        if log_path_match:
            df.at[index, 'log path'] = log_path_match.group(1)

    # Write the modified DataFrame to a new CSV file
    df.to_csv(output_csv, index=False)



input_csv = input("Insert path here: ")  # Prompt user for the input CSV file path
# Get the directory of the input file and replace the filename with 'processed_tickets.csv'
output_csv = os.path.join(os.path.dirname(input_csv), 'processed_tickets.csv')
process_description_column(input_csv, output_csv)