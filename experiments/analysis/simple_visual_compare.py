#!/usr/bin/env python3
"""
Simple visual comparison script that creates basic grid layouts
"""
import os
import argparse
from PIL import Image
import json
from pathlib import Path

def create_simple_grid(image_dir, output_dir):
    """Create a simple grid comparing different methods"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load metadata
    metadata_path = os.path.join(image_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        print(f"No metadata found at {metadata_path}")
        return
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    image_name = metadata['image']
    methods = ['geo', 'clip', 'text']
    
    print(f"📸 Creating simple comparison for: {image_name}")
    
    # Find available results
    results = []
    for file in os.listdir(image_dir):
        if file.endswith('.png') and file != 'original.png':
            parts = file.replace('.png', '').split('_')
            if len(parts) >= 3:
                method = parts[0]
                strength = float(parts[1][1:])  # Remove 's' prefix
                guidance = float(parts[2][1:])  # Remove 'g' prefix
                results.append({
                    'method': method,
                    'strength': strength,
                    'guidance_scale': guidance,
                    'filename': file
                })
    
    if not results:
        print(f"No valid results found in {image_dir}")
        return
    
    # Group by method for comparison
    methods_data = {method: [] for method in methods}
    for result in results:
        if result['method'] in methods_data:
            methods_data[result['method']].append(result)
    
    # Create method comparison grids for different parameter combinations
    unique_combos = list(set((r['strength'], r['guidance_scale']) for r in results))
    unique_combos.sort()
    
    print(f"Found {len(unique_combos)} parameter combinations")
    
    # Create a grid showing all three methods for several key combinations
    img_size = 256
    margin = 10
    
    # Select a few representative combinations
    key_combos = unique_combos[:min(6, len(unique_combos))]  # Show up to 6 combinations
    
    # Grid: methods as columns, combinations as rows
    grid_width = 3 * img_size + 4 * margin  # 3 methods
    grid_height = len(key_combos) * img_size + (len(key_combos) + 1) * margin
    
    canvas = Image.new('RGB', (grid_width, grid_height), 'white')
    
    # Place images
    for i, (strength, guidance) in enumerate(key_combos):
        y_pos = i * (img_size + margin) + margin
        
        for j, method in enumerate(methods):
            x_pos = j * (img_size + margin) + margin
            
            # Find the image for this combination
            matching_results = [r for r in results 
                              if r['method'] == method 
                              and r['strength'] == strength 
                              and r['guidance_scale'] == guidance]
            
            if matching_results:
                img_path = os.path.join(image_dir, matching_results[0]['filename'])
                if os.path.exists(img_path):
                    try:
                        img = Image.open(img_path).convert('RGB')
                        img = img.resize((img_size, img_size))
                        canvas.paste(img, (x_pos, y_pos))
                        print(f"  ✅ Added {method} s{strength} g{guidance}")
                    except Exception as e:
                        print(f"  ❌ Error with {img_path}: {e}")
    
    # Save the comparison grid
    output_path = os.path.join(output_dir, f"{image_name}_method_comparison.png")
    canvas.save(output_path)
    print(f"💾 Saved method comparison: {output_path}")
    
    # Create individual method grids showing parameter variations
    for method in methods:
        method_results = [r for r in results if r['method'] == method]
        if not method_results:
            continue
            
        # Group by strength and guidance
        strengths = sorted(list(set(r['strength'] for r in method_results)))
        guidances = sorted(list(set(r['guidance_scale'] for r in method_results)))
        
        if len(strengths) <= 1 or len(guidances) <= 1:
            continue  # Need variation to create a meaningful grid
        
        # Create parameter grid for this method
        grid_width = len(guidances) * img_size + (len(guidances) + 1) * margin
        grid_height = len(strengths) * img_size + (len(strengths) + 1) * margin
        
        method_canvas = Image.new('RGB', (grid_width, grid_height), 'white')
        
        for i, strength in enumerate(strengths):
            for j, guidance in enumerate(guidances):
                y_pos = i * (img_size + margin) + margin
                x_pos = j * (img_size + margin) + margin
                
                # Find matching result
                matching = [r for r in method_results 
                           if r['strength'] == strength and r['guidance_scale'] == guidance]
                
                if matching:
                    img_path = os.path.join(image_dir, matching[0]['filename'])
                    if os.path.exists(img_path):
                        try:
                            img = Image.open(img_path).convert('RGB')
                            img = img.resize((img_size, img_size))
                            method_canvas.paste(img, (x_pos, y_pos))
                        except Exception as e:
                            print(f"  ❌ Error with {img_path}: {e}")
        
        # Save method grid
        method_output = os.path.join(output_dir, f"{image_name}_{method}_parameter_grid.png")
        method_canvas.save(method_output)
        print(f"💾 Saved {method} parameter grid: {method_output}")

def main():
    parser = argparse.ArgumentParser(description='Create simple visual comparison grids')
    parser.add_argument('--input_dir', required=True, help='Directory containing test results')
    parser.add_argument('--output_dir', required=True, help='Directory to save comparison grids')
    
    args = parser.parse_args()
    
    print("🎨 Creating simple visual comparison grids...")
    
    if os.path.isdir(args.input_dir):
        # Single image directory
        create_simple_grid(args.input_dir, args.output_dir)
    else:
        print(f"❌ Input directory not found: {args.input_dir}")

if __name__ == "__main__":
    main()
