import pandas as pd
import json
import os

# Load the CSV (adjust path if needed)
df = pd.read_csv("alignment/dataset/cities.csv")

# Drop rows with missing coords or names
df = df.dropna(subset=["name", "country_name", "latitude", "longitude"])

# Format into desired structure
def format_row(row):
    return {
        "text": f"{row['name'].strip().lower()}, {row['country_name'].strip().lower()}",
        "location": [round(row["latitude"], 6), round(row["longitude"], 6)]
    }

# Apply to all rows
geo_text = [format_row(r) for _, r in df.iterrows()]

# Optional: limit size
geo_text = geo_text[:10000]  # or 2000 for smaller size

# Save
os.makedirs("alignment/dataset", exist_ok=True)
with open("alignment/dataset/world_geo_text.json", "w") as f:
    json.dump(geo_text, f, indent=2)

print(f"✅ Saved {len(geo_text)} entries to alignment/dataset/world_geo_text.json")