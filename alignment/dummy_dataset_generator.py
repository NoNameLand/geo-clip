import random
import json
from sklearn.model_selection import train_test_split

# Define text labels and approximate lat/lon bounds
LABELS = {
    "desert":          [ [24, 36], [-115, -100] ],  # e.g., Nevada, Arizona
    "rainforest":      [ [-5, 5], [-70, -50] ],     # Amazon basin
    "mountain":        [ [45, 50], [5, 15] ],       # Alps
    "urban skyline":   [ [40, 41], [-74.1, -73.9] ],# NYC
    "tundra":          [ [65, 70], [60, 80] ],      # Siberia
    "savanna":         [ [-5, 5], [30, 40] ],       # Central Africa
    "coastal city":    [ [34, 36], [135, 140] ],    # Japan
    "volcano":         [ [15, 20], [-100, -90] ],   # Central America
}

def sample_location(lat_range, lon_range):
    lat = random.uniform(*lat_range)
    lon = random.uniform(*lon_range)
    return [round(lat, 4), round(lon, 4)]

def build_dataset(samples_per_label=100):
    dataset = []
    for label, (lat_range, lon_range) in LABELS.items():
        for _ in range(samples_per_label):
            location = sample_location(lat_range, lon_range)
            dataset.append({"text": label, "location": location})
    return dataset

# Build and split the dataset
full_dataset = build_dataset(samples_per_label=100)  # ~800 examples
train, rest = train_test_split(full_dataset, test_size=0.2, random_state=42)
val, test = train_test_split(rest, test_size=0.5, random_state=42)

# Save to files (optional)
with open("alignment/dataset/train.json", "w") as f:
    json.dump(train, f, indent=2)
with open("alignment/dataset/val.json", "w") as f:
    json.dump(val, f, indent=2)
with open("alignment/dataset/test.json", "w") as f:
    json.dump(test, f, indent=2)

print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
