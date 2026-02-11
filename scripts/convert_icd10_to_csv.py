import csv

input_file = "data/icd10cm_order_2026.txt"
output_file = "data/icd10cm_2026.csv"

with open(input_file, 'r', encoding='utf-8') as infile, \
     open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    
    writer = csv.writer(outfile)
    
    # Write header
    writer.writerow(["sequence", "code", "is_billable", "short_title", "long_title"])
    
    for line in infile:
        # Fixed width slicing (adjust if needed)
        sequence = line[0:5].strip()
        code = line[6:13].strip()
        billable = line[14:15].strip()
        short_title = line[16:76].strip()
        long_title = line[77:].strip()
        
        writer.writerow([sequence, code, billable, short_title, long_title])

print("CSV file created successfully.")
