import csv
import json
import argparse
import os

def csv_to_json(csv_file_path, json_file_path):
    """
    Convert CSV file with columns 'text', 'lat', 'lon' to JSON format
    Expected CSV format:
    text,lat,lon
    "description of location",latitude,longitude
    """
    data = []
    
    with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            # For cities.csv, use 'name', 'latitude', 'longitude' columns
            entry = {
                "text": row.get("text", row.get("name", "")),
                "lat": float(row.get("lat", row.get("latitude", 0.0))),
                "lon": float(row.get("lon", row.get("longitude", 0.0)))
            }
            data.append(entry)
    
    # Save to JSON
    with open(json_file_path, 'w', encoding='utf-8') as jsonfile:
        json.dump(data, jsonfile, indent=2, ensure_ascii=False)
    
    print(f"✅ Converted {len(data)} entries from {csv_file_path} to {json_file_path}")

def main():
    parser = argparse.ArgumentParser(description='Convert CSV to JSON for geo-clip alignment dataset')
    parser.add_argument('csv_file', help='Input CSV file path')
    parser.add_argument('--output', '-o', help='Output JSON file path (optional)')
    
    args = parser.parse_args()
    
    # Determine output path
    if args.output:
        json_file_path = args.output
    else:
        # Create output filename based on input
        base_name = os.path.splitext(os.path.basename(args.csv_file))[0]
        json_file_path = f"alignment/dataset/{base_name}.json"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(json_file_path), exist_ok=True)
    
    # Convert
    csv_to_json(args.csv_file, json_file_path)

if __name__ == "__main__":
    main()
