import csv
from datetime import datetime

# File paths
input_file = './raw data 20052025.csv'
output_file = './raw data 20052025_converted.csv.csv'

# Process the CSV file
with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
    reader = csv.DictReader(infile)
    fieldnames = ['time'] + [col for col in reader.fieldnames if col not in ['Date', 'Time', 'Millisecond']]
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)

    # Write the header
    writer.writeheader()

    for row in reader:
        # Combine Date, Time, and Millisecond into the new time column
        date = row['Date']
        time = row['Time']
        millisecond = row['Millisecond']
        combined_time = f"{date} {time}.{millisecond}"
        formatted_time = datetime.strptime(combined_time, "%Y/%m/%d %H:%M:%S.%f").strftime("%d.%m.%Y %H:%M:%S.%f")[:-3]

        # Create the new row with the time column and remove the old columns
        new_row = {'time': formatted_time}
        for col in fieldnames[1:]:
            new_row[col] = row[col]

        writer.writerow(new_row)

print(f"Updated file saved to {output_file}")