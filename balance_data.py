import pandas as pd
from pathlib import Path
from collections import defaultdict
from random import choices

# Paths
ANNOTATIONS_PATH = Path("dataset/annotations.csv")
BALANCED_CSV_PATH = Path("dataset/annotations_balanced.csv")
OUTPUT_DIR = Path("dataset")

# Load annotations
print("Loading annotations.csv...")
df = pd.read_csv(ANNOTATIONS_PATH)

# Balance annotations
print("Balancing dataset...")
class_counts = df["Class ID"].value_counts()
max_count = class_counts.max()
balanced_rows = []
for class_id in class_counts.index:
    class_df = df[df["Class ID"] == class_id]
    repeat = max_count // len(class_df)
    remainder = max_count % len(class_df)
    balanced_rows.extend([*class_df.to_dict("records")] * repeat)
    balanced_rows.extend(choices(class_df.to_dict("records"), k=remainder))

df_balanced = pd.DataFrame(balanced_rows)
df_balanced.to_csv(BALANCED_CSV_PATH, index=False)
print(f"Saved balanced annotations to {BALANCED_CSV_PATH}")

# Helper: write list of unique filenames to file
def write_split(df_split, name):
    output_file = OUTPUT_DIR / f"{name}.txt"
    paths = sorted(set(df_split["Filename"]))
    output_file.write_text("\n".join(paths))
    print(f"{name}.txt -> {len(paths)} entries")

# Write splits
print("Generating split files...")
write_split(df_balanced, "train_balanced")
write_split(df_balanced[df_balanced["Filename"].str.contains("day", case=False)], "train_balanced_day")
write_split(df_balanced[df_balanced["Filename"].str.contains("night", case=False)], "train_balanced_night")
write_split(df_balanced[df_balanced["Filename"].str.contains("test", case=False)], "test_balanced")
write_split(df_balanced[df_balanced["Filename"].str.contains("test") & df_balanced["Filename"].str.contains("day")], "test_balanced_day")
write_split(df_balanced[df_balanced["Filename"].str.contains("test") & df_balanced["Filename"].str.contains("night")], "test_balanced_night")

print("All files generated.")
