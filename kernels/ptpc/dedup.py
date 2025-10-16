import pandas as pd
import numpy as np

# a8w8_bpreshuffle_tuned_gemm.csv
# a8w8_tuned_gemm.csv
CSV_FILE = 'a8w8_tuned_gemm.csv'

# Read the CSV file
df = pd.read_csv(CSV_FILE)

# Remove entries where 'us' is 'inf'
df = df[df['us'] != 'inf']

# Convert 'us' column to numeric (in case it's stored as string)
df['us'] = pd.to_numeric(df['us'])

# Group by the key columns an
# d find the row with minimum 'us' value
result = df.loc[df.groupby(['cu_num', 'M', 'N', 'K'])['us'].idxmin()]

# Reset index for cleaner output
result = result.reset_index(drop=True)

# Save the result to a new CSV file
result.to_csv(CSV_FILE, index=False)

# Optional: Print the result to see what we got
print(result)