import pandas as pd
import numpy as np
import os

# ================= USER SETTINGS ================= #
BIG_DATASET_PATH = r"D:\sem-4\dietrecommendation\data\recipes.csv"      # <-- CHANGE THIS
SAVE_PATH = r"D:\sem-4\dietrecommendation\data\clean_recipes_10k.csv"
TARGET_ROWS = 10000
CHUNK_SIZE = 50000
# ================================================= #

os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

clean_chunks = []
total_rows = 0

print("🚀 Starting dataset cleaning...")

for chunk in pd.read_csv(BIG_DATASET_PATH, chunksize=CHUNK_SIZE):

    # Remove rows having any missing values
    chunk = chunk.dropna()

    # Remove duplicate rows
    chunk = chunk.drop_duplicates()

    # Keep only meaningful numeric values
    numeric_cols = chunk.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        chunk = chunk[chunk[col] >= 0]

    clean_chunks.append(chunk)
    total_rows += len(chunk)

    print(f"✔ Clean rows collected: {total_rows}")

    if total_rows >= TARGET_ROWS:
        break

# Combine & trim to exact 10k
final_df = pd.concat(clean_chunks).head(TARGET_ROWS)

# Save final dataset
final_df.to_csv(SAVE_PATH, index=False)

print("\n🎯 DONE!")
print("Final clean dataset saved at:")
print(SAVE_PATH)
print("Total rows:", len(final_df))
