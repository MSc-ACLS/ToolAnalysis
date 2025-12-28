# ...existing code...
import csv
from datetime import datetime

INPUT_FILE = './zhaw_chlamydomonas.csv'
OUTPUT_FILE = './zhaw_chlamydomonas_converted.csv'

def convert_date_format(date_str):
    if not date_str or date_str == 'NA':
        return date_str
    parts = date_str.split('-')
    if len(parts) != 3:
        return date_str
    year, month, day = parts
    return f"{day}.{month}.{year}"

def parse_datetime_flexible(s):
    if not s or s == 'NA':
        return None
    s = s.strip()
    fmts = [
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None

with open(INPUT_FILE, 'r', encoding='utf-8', newline='') as infile, \
     open(OUTPUT_FILE, 'w', encoding='utf-8', newline='') as outfile:
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames or []
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()

    for row in reader:
        # time.string -> "DD.MM.YYYY HH:MM:SS"
        if 'time.string' in row:
            orig = row.get('time.string', '')
            dt = parse_datetime_flexible(orig)
            if dt:
                row['time.string'] = dt.strftime("%d.%m.%Y %H:%M:%S")
            else:
                # keep original if unparseable
                row['time.string'] = orig

        # date -> "DD.MM.YYYY"
        if 'date' in row:
            row['date'] = convert_date_format(row.get('date', ''))

        writer.writerow(row)

print(f"Updated file saved to {OUTPUT_FILE}")
