import pandas as pd

input_file = "data/2015_I9gem.txt"
output_file = "data/2015_I9gem.csv"

rows = []

with open(input_file, "r") as f:
    for line in f:
        parts = line.strip().split()
        
        if len(parts) >= 3:
            icd9 = parts[0]
            icd10 = parts[1]
            flags = parts[2]

            rows.append({
                "icd9_code": icd9,
                "icd10_code": icd10,
                "approximate": flags[0],
                "no_map": flags[1],
                "combination": flags[2],
                "scenario": flags[3],
                "choice_list": flags[4]
            })

df = pd.DataFrame(rows)
df.to_csv(output_file, index=False)

print("GEM CSV regenerated with flag columns.")
