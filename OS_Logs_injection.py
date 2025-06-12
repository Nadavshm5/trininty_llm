from datetime import datetime

def convert_timestamp(timestamp):
    # Truncate nanoseconds to microseconds
    timestamp_microseconds = timestamp[:-3]  # Remove the last three digits
    # Convert from '2025/06/04-11:27:31.991626' to '06/04/2025-11:27:31.991'
    dt = datetime.strptime(timestamp_microseconds, '%Y/%m/%d-%H:%M:%S.%f')
    return dt.strftime('%m/%d/%Y-%H:%M:%S.%f')[:-3]

def find_and_insert_patterns(OS_log, Driver_log, patterns):
    # Read the first file and find lines with the patterns
    lines_to_insert = []
    with open(OS_log, 'r') as file1:
        for line in file1:
            for pattern in patterns:
                if pattern in line:
                    # Extract the timestamp and the rest of the line
                    parts = line.split('::')
                    timestamp = parts[1].split(' ')[0]
                    rest_of_line = ' '.join(parts[1].split(' ')[1:])
                    converted_timestamp = convert_timestamp(timestamp)
                    lines_to_insert.append((converted_timestamp, rest_of_line.strip()))
                    break  # Stop checking other patterns once a match is found

    # Read the second file
    with open(Driver_log, 'r') as file2:
        file2_lines = file2.readlines()

    # Prepare to insert lines into the second file at the correct position
    new_file2_lines = []
    insert_index = 0

    for line in file2_lines:
        file2_timestamp = line.split(' ', 1)[0]
        # Insert all lines from file1 that have a timestamp less than the current line in file2
        while insert_index < len(lines_to_insert) and lines_to_insert[insert_index][0] < file2_timestamp:
            new_file2_lines.append(f"{lines_to_insert[insert_index][0]} {lines_to_insert[insert_index][1]}\n")
            insert_index += 1
        # Add the current line from file2
        new_file2_lines.append(line)

    # Append any remaining lines from file1
    while insert_index < len(lines_to_insert):
        new_file2_lines.append(f"{lines_to_insert[insert_index][0]} {lines_to_insert[insert_index][1]}\n")
        insert_index += 1

    # Write the updated content back to the second file
    with open(Driver_log, 'w') as file2:
        file2.writelines(new_file2_lines)

def verify_and_fix_order(Driver_log):
    with open(Driver_log, 'r') as file:
        lines = file.readlines()

    # Extract timestamps and sort lines
    lines_with_timestamps = []
    for line in lines:
        timestamp = line.split(' ', 1)[0]
        lines_with_timestamps.append((timestamp, line))

    # Sort lines based on timestamps
    lines_with_timestamps.sort(key=lambda x: x[0])

    # Write sorted lines back to the file
    with open(Driver_log, 'w') as file:
        for _, line in lines_with_timestamps:
            file.write(line)

    print("Lines have been sorted according to timestamps.")

# Example usage
OS_log = input("please insert your OS log path: ")
Driver_log = input("please insert your Driver log path: ")
patterns = ['Limited Connectivity']
find_and_insert_patterns(OS_log, Driver_log, patterns)

# Verify and fix the order of lines in the Driver_log
verify_and_fix_order(Driver_log)