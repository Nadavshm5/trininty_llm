import os
import pandas as pd
import re
from datetime import datetime
import json

def parse_log_file(log_file_path, start_line, end_line):
    # Define regex patterns for each event with the new timestamp format
    # Load patterns from a configuration file
    config_file_path = os.path.join(os.path.dirname(__file__), 'config.json')
    with open(config_file_path, 'r') as config_file:
        config_dict = json.load(config_file)
        patterns_dict = config_dict['patterns']

    # Convert the loaded patterns into compiled regex patterns
    patterns = {event: re.compile(pattern) for event, pattern in patterns_dict.items()}

    # Regex to extract SSID and MAC address
    ssid_mac_pattern = re.compile(r'(\w+) (\w{2}:\w{2}:\w{2}:\w{2}:\w{2}:\w{2})')

    # List to store parsed log data
    log_data = []
    last_ssid_mac = None

    # Read the log file
    with open(log_file_path, 'r') as file:
        for line_number, line in enumerate(file, start=1):
            if start_line <= line_number <= end_line:
                ssid_mac_match = ssid_mac_pattern.search(line)
                if ssid_mac_match:
                    last_ssid_mac = ssid_mac_match.group(0)

                for event, pattern in patterns.items():
                    match = pattern.search(line)
                    if match:
                        timestamp_str = match.group(1)
                        timestamp = datetime.strptime(timestamp_str, '%m/%d/%Y-%H:%M:%S.%f')
                        log_data.append((timestamp, event, last_ssid_mac, line_number, line.strip()))
                        # Print the identified event and line number
                        # print(f"Identified {event} on line {line_number}: {line.strip()}")

    # Convert to DataFrame
    df = pd.DataFrame(log_data, columns=['Timestamp', 'Event', 'SSID_MAC', 'LineNumber', 'LineContent'])
    return df

def generate_text_representation(df):
    text_representation = []
    for index, row in df.iterrows():
        text_representation.append(f"[{row['Timestamp']}] ({row['LineNumber']}) {row['Event']} - {row['SSID_MAC']}: {row['LineContent']}")
    return "\n".join(text_representation)

def main():
    # log_file_path = r"C:\Work\GenAI\use-cases\Trinity\New folder\log_example_wifi573300.log"  # Path to your log file
    # log_file_path = r"C:\Work\GenAI\use-cases\Trinity\New folder (2)\WifiDriver_99.0.95.5__04-02-2025_20-32-01_273.004.etl.log"
    log_file_path = r"\\apollo.intel.com\apollo-info-eu-west-1\IWFT-EXT\SAHARON-MOBL\WifiDriverLogs-LocalTime\2025-07\WifiDriver_23.140.0.3__12-07-2025_10-39-20_432.001.etl.log"

    # Prompt user for line range
    start_line = 34238
    end_line = 40388

    df = parse_log_file(log_file_path, start_line, end_line)
    print(f"Processed {len(df)} relevant log lines.")

    text_representation = generate_text_representation(df)
    # print("\nGenerated Text Representation:\n")
    # print(text_representation)

    # Save to a file if needed
    txt_path = os.path.join(os.path.dirname(__file__), 'res', 'log_text_representation.txt')
    with open(txt_path, "w") as text_file:
        text_file.write(text_representation)

if __name__ == '__main__':
    main()