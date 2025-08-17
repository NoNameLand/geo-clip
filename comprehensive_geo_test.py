#!/usr/bin/env python3
"""
Comprehensive Geo-Style Transfer Testing Pipeline
Tests multiple images with various strength and guidance scale combinations
"""

import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
import geoclip
import os
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModel
import glob
from tqdm import tqdm
import json
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load SD text encoder (CLIP ViT-L/14) with proper device handling
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🚀 Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    # Enable GPU optimizations for faster training
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

# Initialize text encoder (device assignment will be handled in training loop)
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
text_encoder.eval()  # Freeze
for p in text_encoder.parameters():
    p.requires_grad = False

# Move text encoder to device after model initialization
text_encoder = text_encoder.to(device)

def build_clip_token_anchors(prompts, device="cuda", dtype=torch.float32):
    """Build anchor token embeddings from text prompts using CLIP"""
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder.eval()
    if isinstance(device, str):
        device = torch.device(device)
    text_encoder = text_encoder.to(device)
    
    anchors = []
    with torch.no_grad():
        for prompt in prompts:
            tokens = tokenizer([prompt], padding="max_length", truncation=True, 
                             max_length=77, return_tensors="pt").to(device)
            text_outputs = text_encoder(**tokens)
            anchor = text_outputs.last_hidden_state  # [1, 77, 768]
            anchors.append(anchor.squeeze(0))  # [77, 768]
    
    # Stack to [M, 77, 768]
    anchors = torch.stack(anchors, dim=0).to(dtype)
    return anchors

class GeoEncoder(nn.Module):
    """Geo-conditioned prompt generator using frozen anchor embeddings"""
    def __init__(self, location_encoder, input_feat_dim=512, output_dim=768, 
                 seq_length=77, anchors=None, M=8, r=8):
        super(GeoEncoder, self).__init__()

        # Freeze the location encoder
        self.location_encoder = location_encoder
        for param in self.location_encoder.parameters():
            param.requires_grad = False
        
        self.input_feat_dim = input_feat_dim
        self.output_dim = output_dim
        self.seq_length = seq_length
        self.M = M
        self.r = r
        
        # Store frozen anchor embeddings [M, 77, 768]
        if anchors is not None:
            self.register_buffer('anchors', anchors)  # frozen
        else:
            # Fallback random anchors if not provided
            self.register_buffer('anchors', torch.randn(M, seq_length, output_dim) * 0.02)
        
        # Conditioning network: geo features -> mixing weights
        self.conditioning_net = nn.Sequential(
            nn.Linear(input_feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, M)  # output M mixing weights
        )
        
        # Optional: low-rank adaptation for fine-tuning anchors
        if r > 0:
            self.lora_A = nn.Parameter(torch.randn(M, seq_length, r) * 0.02)
            self.lora_B = nn.Parameter(torch.randn(M, r, output_dim) * 0.02)
        else:
            self.lora_A = None
            self.lora_B = None
        
        # Final layer norm
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(self, coords):
        batch_size = coords.size(0)
        
        # Get geo features (frozen)
        with torch.no_grad():
            geo_features = F.relu(self.location_encoder(coords))  # [B, 512]
        
        # Compute mixing weights
        mixing_weights = self.conditioning_net(geo_features)  # [B, M]
        mixing_weights = F.softmax(mixing_weights, dim=-1)     # [B, M]
        
        # Get base anchors
        base_tokens = self.anchors  # [M, 77, 768]
        
        # Apply LoRA adaptation if enabled
        anchors_tensor = self.get_buffer('anchors')  # Get tensor from buffer
        if self.lora_A is not None and self.lora_B is not None:
            # Low-rank adaptation: anchors + A @ B
            lora_delta = torch.matmul(self.lora_A, self.lora_B)  # [M, 77, 768]
            adapted_tokens = anchors_tensor + lora_delta
        else:
            adapted_tokens = anchors_tensor
        
        # Mix anchors based on geo conditioning
        # [B, M] @ [M, 77, 768] -> [B, 77, 768]
        output_tokens = torch.einsum('bm,mld->bld', mixing_weights, adapted_tokens)
        
        # Final normalization
        output_tokens = self.layer_norm(output_tokens)
        
        return output_tokens

class GeoStyleTransfer:
    def __init__(self, device=None, model_path="alignment/models/mixture_checkpoint_epoch_15.pth"):
        """Initialize the GeoStyleTransfer pipeline"""
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print("🚀 Loading models...")
        
        # Load geo encoder
        # Build anchors from urban prompts
        prompts = [
            "a photo of a city skyline, high detail",
            "wide aerial cityscape, daytime", 
            "aerial night city lights, long exposure",
            "street-level urban scene with pedestrians",
            "historic old town architecture, narrow streets",
            "modern glass skyscrapers downtown",
            "coastal city skyline with water reflection",
            "dense urban core, overcast"
        ]

        # Setup with GPU optimizations
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("🏗️ Building CLIP token anchors...")
        anchors = build_clip_token_anchors(prompts, device=str(device), dtype=torch.float32)  # [M,77,768]
        print(f"✅ Built {len(prompts)} anchor embeddings: {anchors.shape}")

        model = GeoEncoder(
            location_encoder=geoclip.LocationEncoder(),
            input_feat_dim=512,
            output_dim=768,
            seq_length=77,
            anchors=anchors,
            M=len(prompts),
            r=8  # LoRA rank
        ).to(device)
        self.geo_encoder = model.to(self.device)
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    self.geo_encoder.load_state_dict(checkpoint['model_state_dict'], strict=False)
                else:
                    self.geo_encoder.load_state_dict(checkpoint, strict=False)
                print(f"✅ Loaded geo encoder from {model_path}")
            except Exception as e:
                print(f"⚠️ Error loading model: {e}, using random weights")
        else:
            print(f"⚠️ Model checkpoint not found at {model_path}, using random weights")
        self.geo_encoder.eval()
        
        # Load Stable Diffusion pipeline
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
        ).to(self.device)
        
        # Load CLIP for comparison
        self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_text_encoder = self.clip_text_encoder.to(self.device)
        self.clip_text_encoder.eval()
        
        print("✅ All models loaded successfully!")
    
    def get_geo_embedding(self, coords):
        """Get embedding from geo coordinates"""
        coords_tensor = torch.tensor([coords], dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            prompt_embed = self.geo_encoder(coords_tensor)  # [1, 77, 768]
            
        return prompt_embed
    
    def get_clip_embedding(self, text_prompt):
        """Get CLIP embedding from text"""
        clip_tokens = self.clip_tokenizer([text_prompt], return_tensors="pt", 
                                        padding="max_length", truncation=True, max_length=77).to(self.device)
        
        with torch.no_grad():
            text_outputs = self.clip_text_encoder(**clip_tokens)
            prompt_embed = text_outputs.last_hidden_state  # [1, 77, 768]
            
        return prompt_embed
    
    def generate_image(self, image_path, prompt_embed=None, text_prompt=None, 
                      strength=0.4, guidance_scale=4.0, output_path=None):
        """Generate image using either embedding or text prompt"""
        
        # Load and preprocess input image
        init_image = Image.open(image_path).convert("RGB").resize((512, 512))
        
        try:
            if prompt_embed is not None:
                # Use custom embedding
                output = self.pipe(
                    prompt_embeds=prompt_embed,
                    image=init_image,
                    strength=strength,
                    guidance_scale=guidance_scale
                )
            else:
                # Use text prompt
                output = self.pipe(
                    prompt=text_prompt,
                    image=init_image,
                    strength=strength,
                    guidance_scale=guidance_scale
                )
            
            # Save result if path provided
            generated_image = output.images[0]
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                generated_image.save(output_path)
                
            return generated_image
            
        except Exception as e:
            print(f"❌ Error generating image: {e}")
            return None
    
    def test_single_image(self, image_path, coords, location_name, 
                         strengths=[0.3, 0.4, 0.5], guidance_scales=[3.0, 4.0, 5.0]):
        """Test a single image with different parameters - applying geographic style transfer"""
        
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        # Create results directory that shows the style transfer: source_to_target
        results_dir = f"results/comprehensive_test/{image_name}_to_{location_name.replace(', ', '_').replace(' ', '_')}"
        
        print(f"\n🖼️  Testing image: {image_name}")
        print(f"🎨 Applying style from: {location_name} {coords}")
        print(f"📁 Results will be saved to: {results_dir}")
        
        # Get embeddings for the TARGET location (style we want to apply)
        geo_embed = self.get_geo_embedding(coords)
        clip_embed = self.get_clip_embedding(location_name)
        
        results = {
            'source_image': image_name,
            'target_style_location': location_name,
            'target_coordinates': coords,
            'style_transfer_description': f"Applying {location_name} style to {image_name} image",
            'results': []
        }
        
        # Test combinations of strength and guidance scale
        total_tests = len(strengths) * len(guidance_scales) * 3  # 3 methods
        pbar = tqdm(total=total_tests, desc=f"Testing {image_name}")
        
        for strength in strengths:
            for guidance_scale in guidance_scales:
                test_result = {
                    'strength': strength,
                    'guidance_scale': guidance_scale,
                    'outputs': {}
                }
                
                # Test 1: Geo embedding
                output_path = f"{results_dir}/geo_s{strength}_g{guidance_scale}.png"
                geo_image = self.generate_image(
                    image_path, prompt_embed=geo_embed, 
                    strength=strength, guidance_scale=guidance_scale,
                    output_path=output_path
                )
                if geo_image:
                    test_result['outputs']['geo'] = output_path
                pbar.update(1)
                
                # Test 2: CLIP embedding  
                output_path = f"{results_dir}/clip_s{strength}_g{guidance_scale}.png"
                clip_image = self.generate_image(
                    image_path, prompt_embed=clip_embed,
                    strength=strength, guidance_scale=guidance_scale,
                    output_path=output_path
                )
                if clip_image:
                    test_result['outputs']['clip'] = output_path
                pbar.update(1)
                
                # Test 3: Text prompt
                output_path = f"{results_dir}/text_s{strength}_g{guidance_scale}.png"
                text_image = self.generate_image(
                    image_path, text_prompt=location_name,
                    strength=strength, guidance_scale=guidance_scale, 
                    output_path=output_path
                )
                if text_image:
                    test_result['outputs']['text'] = output_path
                pbar.update(1)
                
                results['results'].append(test_result)
        
        pbar.close()
        
        # Save metadata
        with open(f"{results_dir}/metadata.json", 'w') as f:
            json.dump(results, f, indent=2)
            
        return results

def run_comprehensive_test():
    """Run comprehensive test on all available images"""
    
    # Test configurations - STYLE TRANSFER: image from one location, style from another
    test_configs = [
        # (input_image, source_location, target_coords, target_location_name)
        # Style transfer: Apply different geographic styles to each image
        ("tokyo.jpg", "Tokyo", [45.4408, 12.3155], "Venice, Italy"),           # Tokyo image → Venice style
        ("venice.jpg", "Venice", [35.6762, 139.6503], "Tokyo, Japan"),         # Venice image → Tokyo style  
        ("tokyo.jpg", "Tokyo", [24.2048, 55.2708], "Dubai Desert, UAE"),      # Tokyo image → Desert style
        ("venice.jpg", "Venice", [40.7589, -73.9851], "New York City, USA"),  # Venice image → NYC style
        ("tokyo.jpg", "Tokyo", [31.7683, 35.2137], "Jerusalem, Israel"),      # Tokyo image → Jerusalem style
    ]
    
    # Test parameters - comprehensive range
    strengths = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    guidance_scales = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    
    # Initialize pipeline
    pipeline = GeoStyleTransfer()
    
    print("🧪 Running comprehensive geo-style transfer test...")
    print(f"📊 Testing {len(test_configs)} images with {len(strengths)} strengths × {len(guidance_scales)} guidance scales")
    print(f"🎯 Total combinations: {len(test_configs) * len(strengths) * len(guidance_scales) * 3} outputs")
    
    all_results = []
    
    for image_file, source_location, target_coords, target_location_name in test_configs:
        image_path = f"{image_file}"  # Assuming we're running from the project root
        
        if os.path.exists(image_path):
            print(f"\n🎨 Style Transfer: {source_location} image → {target_location_name} style")
            results = pipeline.test_single_image(
                image_path, target_coords, target_location_name, 
                strengths=strengths, guidance_scales=guidance_scales
            )
            all_results.append(results)
        else:
            print(f"⚠️ Image not found: {image_path}")
    
    # Save comprehensive results
    os.makedirs("results/comprehensive_test", exist_ok=True)
    with open("results/comprehensive_test/all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Create summary report
    create_summary_report(all_results)
    
    print(f"\n🎉 Comprehensive test complete!")
    print(f"📁 Results saved to: results/comprehensive_test/")
    print(f"📊 Tested {len(all_results)} images successfully")
    
    return all_results

def create_summary_report(all_results):
    """Create an HTML summary report"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Geo-Style Transfer Test Results</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .image-section { border: 1px solid #ccc; margin: 20px 0; padding: 15px; }
            .parameter-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 10px; }
            .result-item { border: 1px solid #ddd; padding: 10px; text-align: center; }
            .result-item img { max-width: 100%; height: auto; }
            h1 { color: #333; }
            h2 { color: #555; }
            h3 { color: #777; }
        </style>
    </head>
    <body>
        <h1>🌍 Geo-Style Transfer Comprehensive Test Results</h1>
    """
    
    for result in all_results:
        html_content += f"""
        <div class="image-section">
            <h2>📸 {result['image']}</h2>
            <h3>📍 {result['location']} {result['coordinates']}</h3>
            <div class="parameter-grid">
        """
        
        for test in result['results']:
            strength = test['strength']
            guidance = test['guidance_scale']
            outputs = test['outputs']
            
            html_content += f"""
                <div class="result-item">
                    <h4>Strength: {strength}, Guidance: {guidance}</h4>
            """
            
            for method, path in outputs.items():
                if os.path.exists(path):
                    html_content += f"""
                        <div>
                            <strong>{method.upper()}</strong><br>
                            <img src="{path}" alt="{method}" style="max-width: 150px;">
                        </div>
                    """
            
            html_content += "</div>"
        
        html_content += "</div></div>"
    
    html_content += """
        </body>
    </html>
    """
    
    with open("results/comprehensive_test/summary_report.html", 'w') as f:
        f.write(html_content)
    
    print("📄 Summary report saved to: results/comprehensive_test/summary_report.html")

def run_quick_test():
    """Run a quick test with fewer parameters demonstrating style transfer"""
    # Style transfer examples - image from one location, style from another
    test_configs = [
        # (input_image, source_location, target_coords, target_location_name) 
        ("tokyo.jpg", "Tokyo", [45.4408, 12.3155], "Venice, Italy"),    # Tokyo image → Venice style
        ("venice.jpg", "Venice", [35.6762, 139.6503], "Tokyo, Japan"),  # Venice image → Tokyo style
    ]
    
    strengths = [0.3, 0.5]
    guidance_scales = [3.0, 5.0]
    
    pipeline = GeoStyleTransfer()
    
    print("🚀 Running quick style transfer test...")
    print("🎨 Demonstrating geographic style transfer: applying one location's style to another's image")
    
    for image_file, source_location, target_coords, target_location_name in test_configs:
        if os.path.exists(image_file):
            print(f"\n🎨 Style Transfer: {source_location} image → {target_location_name} style")
            pipeline.test_single_image(
                image_file, target_coords, target_location_name, 
                strengths=strengths, guidance_scales=guidance_scales
            )
        else:
            print(f"⚠️ Image not found: {image_file}")
    
    print("✅ Quick style transfer test complete!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run geo-style transfer tests")
    parser.add_argument("--quick", action="store_true", help="Run quick test with fewer parameters")
    parser.add_argument("--full", action="store_true", help="Run comprehensive test")
    
    args = parser.parse_args()
    
    if args.quick:
        run_quick_test()
    elif args.full:
        run_comprehensive_test()
    else:
        # Default: run quick test
        print("Running quick test by default. Use --full for comprehensive test.")
        run_quick_test()
