#!/usr/bin/env python3
"""Monitor training progress"""

import os
import time
import glob

print("🔍 Monitoring training progress...")

model_dir = "/root/Projects/sidehustle/geo-clip/alignment/models/"
os.makedirs(model_dir, exist_ok=True)

while True:
    # Check for checkpoints
    checkpoints = glob.glob(f"{model_dir}mixture_checkpoint_*.pth")
    if checkpoints:
        latest = max(checkpoints, key=os.path.getctime)
        mod_time = time.ctime(os.path.getmtime(latest))
        print(f"📁 Latest checkpoint: {os.path.basename(latest)} (modified: {mod_time})")
    
    # Check for loss plot
    if os.path.exists(f"{model_dir}loss_plot.png"):
        mod_time = time.ctime(os.path.getmtime(f"{model_dir}loss_plot.png"))
        print(f"📊 Loss plot exists (modified: {mod_time})")
    
    # Check for final model
    if os.path.exists(f"{model_dir}geo_seq_model_cities.pt"):
        mod_time = time.ctime(os.path.getmtime(f"{model_dir}geo_seq_model_cities.pt"))
        print(f"🎯 Final model saved! (modified: {mod_time})")
        break
    
    print(f"⏱️  Training in progress... {time.strftime('%H:%M:%S')}")
    time.sleep(30)  # Check every 30 seconds
