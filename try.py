import torch
from geoclip.model import GeoCLIP

# Initialize the GeoCLIP model
model = GeoCLIP()

# Path to your image
image_path = "geoclip/images/Kauai.png"

# Predict the top 5 GPS coordinates
top_pred_gps, top_pred_prob = model.predict(image_path, top_k=5)

# Display the predictions
print("Top 5 GPS Predictions:")
print("======================")
for i in range(5):
    lat, lon = top_pred_gps[i]
    print(f"Prediction {i+1}: Latitude: {lat:.6f}, Longitude: {lon:.6f}")
    print(f"Confidence: {top_pred_prob[i]:.6f}\n")
