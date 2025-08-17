#!/usr/bin/env python3
"""
Create a visual comparison grid of the results
"""

import os
import argparse
from PIL import Image, ImageDraw, ImageFont
import json
from pathlib import Path

def get_text_size(draw, text, font):
    """Get text size with compatibility for different PIL versions"""
    try:
        # Try newer PIL method first
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        # Fallback for older PIL versions
        return draw.textsize(text, font=font)
import math

def create_comparison_grid(results_dir="results/comprehensive_test"):
    """Create comparison grids for each image"""
    
    # Find all image directories
    image_dirs = [d for d in os.listdir(results_dir) 
                  if os.path.isdir(os.path.join(results_dir, d)) and d not in ['summary']]
    
    for image_name in image_dirs:
        image_dir = os.path.join(results_dir, image_name)
        metadata_path = os.path.join(image_dir, "metadata.json")
        
        if not os.path.exists(metadata_path):
            continue
            
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"📸 Creating comparison grid for: {image_name}")
        create_single_image_grid(image_dir, metadata, results_dir)

def create_single_image_grid(image_dir, metadata, output_base):
    """Create a comparison grid for a single image"""
    
    image_name = metadata['image']
    location = metadata['location']
    coords = metadata['coordinates']
    results = metadata['results']
    
    # Get unique strengths and guidance scales
    strengths = sorted(set(r['strength'] for r in results))
    guidance_scales = sorted(set(r['guidance_scale'] for r in results))
    
    # Image dimensions
    img_size = 200  # Resize images to this size for grid
    margin = 20
    label_height = 30
    
    # Grid dimensions
    cols = len(guidance_scales) + 1  # +1 for strength labels
    rows = len(strengths) + 1        # +1 for guidance labels
    methods = ['geo', 'clip', 'text']
    
    # Calculate grid size for each method
    grid_width = cols * (img_size + margin) + margin
    grid_height = rows * (img_size + label_height + margin) + margin
    
    # Create grids for each method
    for method in methods:
        print(f"  Creating {method} grid...")
        
        # Create blank canvas
        canvas = Image.new('RGB', (grid_width, grid_height), 'white')
        draw = ImageDraw.Draw(canvas)
        
        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 16)
            small_font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 12)
        except:
            font = ImageFont.load_default()
            small_font = font
        
        # Draw title
        title = f"{image_name.upper()} - {method.upper()} Method"
        subtitle = f"Location: {location} {coords}"
        draw.text((margin, 10), title, fill='black', font=font)
        draw.text((margin, 30), subtitle, fill='gray', font=small_font)
        
        # Draw column headers (guidance scales)
        for j, guidance_scale in enumerate(guidance_scales):
            x = (j + 1) * (img_size + margin) + margin + img_size // 2
            y = 60
            text = f"G: {guidance_scale}"
            text_width, _ = get_text_size(draw, text, small_font)
            draw.text((x - text_width // 2, y), text, fill='blue', font=small_font)
        
        # Draw row headers and images
        for i, strength in enumerate(strengths):
            # Draw strength label
            y = 80 + (i + 1) * (img_size + label_height + margin) + img_size // 2
            text = f"S: {strength}"
            draw.text((margin, y), text, fill='green', font=small_font)
            
            # Draw images for this strength
            for j, guidance_scale in enumerate(guidance_scales):
                # Find the result for this combination
                result = next((r for r in results 
                             if r['strength'] == strength and r['guidance_scale'] == guidance_scale), None)
                
                if result and method in result['outputs']:
                    img_path = result['outputs'][method]
                    
                    if os.path.exists(img_path):
                        # Load and resize image
                        img = Image.open(img_path).convert('RGB')
                        img = img.resize((img_size, img_size), Image.Resampling.LANCZOS)
                        
                        # Calculate position
                        x = (j + 1) * (img_size + margin) + margin
                        y = 80 + (i + 1) * (img_size + label_height + margin)
                        
                        # Paste image
                        canvas.paste(img, (x, y))
                        
                        # Draw border
                        draw.rectangle([x-1, y-1, x+img_size+1, y+img_size+1], outline='black', width=1)
        
        # Save grid
        output_path = os.path.join(output_base, f"{image_name}_{method}_grid.png")
        canvas.save(output_path)
        print(f"  ✅ Saved: {output_path}")

def create_method_comparison(results_dir="results/comprehensive_test"):
    """Create side-by-side comparisons of the three methods"""
    
    image_dirs = [d for d in os.listdir(results_dir) 
                  if os.path.isdir(os.path.join(results_dir, d)) and d not in ['summary']]
    
    for image_name in image_dirs:
        image_dir = os.path.join(results_dir, image_name)
        metadata_path = os.path.join(image_dir, "metadata.json")
        
        if not os.path.exists(metadata_path):
            continue
            
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"📊 Creating method comparison for: {image_name}")
        
        # Pick a representative parameter combination (middle values)
        results = metadata['results']
        strengths = sorted(set(r['strength'] for r in results))
        guidance_scales = sorted(set(r['guidance_scale'] for r in results))
        
        mid_strength = strengths[len(strengths)//2]
        mid_guidance = guidance_scales[len(guidance_scales)//2]
        
        # Find the result for this combination
        result = next((r for r in results 
                     if r['strength'] == mid_strength and r['guidance_scale'] == mid_guidance), None)
        
        if result and all(method in result['outputs'] for method in ['geo', 'clip', 'text']):
            create_method_comparison_image(result, metadata, results_dir)

def create_method_comparison_image(result, metadata, output_base):
    """Create a side-by-side comparison of the three methods"""
    
    methods = ['geo', 'clip', 'text']
    method_names = ['Geo Encoder', 'CLIP Encoder', 'Text Prompt']
    
    img_size = 256
    margin = 20
    label_height = 40
    
    # Canvas dimensions
    canvas_width = len(methods) * (img_size + margin) + margin
    canvas_height = img_size + label_height + margin * 2
    
    canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
    draw = ImageDraw.Draw(canvas)
    
    # Try to load font
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 16)
        small_font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 12)
    except:
        font = ImageFont.load_default()
        small_font = font
    
    # Load and place images
    for i, (method, method_name) in enumerate(zip(methods, method_names)):
        if method in result['outputs']:
            img_path = result['outputs'][method]
            
            if os.path.exists(img_path):
                # Load and resize image
                img = Image.open(img_path).convert('RGB')
                img = img.resize((img_size, img_size), Image.Resampling.LANCZOS)
                
                # Calculate position
                x = i * (img_size + margin) + margin
                y = margin
                
                # Paste image
                canvas.paste(img, (x, y))
                
                # Draw border
                draw.rectangle([x-1, y-1, x+img_size+1, y+img_size+1], outline='black', width=1)
                
                # Draw method label
                label_y = y + img_size + 5
                bbox = draw.textbbox((0, 0), method_name, font=font)
                text_width = bbox[2] - bbox[0]
                draw.text((x + img_size // 2 - text_width // 2, label_y), method_name, fill='black', font=font)
    
    # Draw parameters info
    params_text = f"Strength: {result['strength']}, Guidance: {result['guidance_scale']}"
    draw.text((margin, canvas_height - 20), params_text, fill='gray', font=small_font)
    
    # Save comparison
    image_name = metadata['image']
    output_path = os.path.join(output_base, f"{image_name}_method_comparison.png")
    canvas.save(output_path)
    print(f"  ✅ Saved: {output_path}")

if __name__ == "__main__":
    print("🎨 Creating visual comparison grids...")
    
    # Create individual grids for each method
    create_comparison_grid()
    
    # Create method comparisons
    create_method_comparison()
    
    print("✅ Visual comparisons created!")
    print("📁 Check the results/comprehensive_test/ directory for grid images")
