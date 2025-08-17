import torch
import torch.nn as nn
import torch.nn.functional as F

# Quick parameter count for mixture-of-bases architecture
def analyze_mixture_architecture():
    print("🧮 Parameter Analysis: Mixture-of-Bases Architecture")
    print("=" * 60)
    
    # Architecture parameters
    M = 8  # number of bases
    seq_length = 77
    output_dim = 768
    
    # Component breakdown
    components = {
        "Initial projection (512→768)": 512 * 768,
        "Prompt bases (M×77×768)": M * seq_length * output_dim,
        "Mix head layer 1 (768→64)": 768 * 64 + 64,
        "Mix head layer 2 (64→M)": 64 * M + M,
        "Positional encoding (1×77×768)": seq_length * output_dim,
        "Layer norm (768)": output_dim * 2,  # weight + bias
    }
    
    print("📊 Parameter Breakdown:")
    total_trainable = 0
    for name, count in components.items():
        print(f"   {name}: {count:,}")
        total_trainable += count
    
    print(f"\n🎯 Total Trainable Parameters: {total_trainable:,}")
    
    # Compare with alternatives
    print(f"\n🔄 Comparison with Other Architectures:")
    transformer_params = (
        768 * 768 * 3 +  # QKV projections
        768 * 768 +      # Output projection  
        768 * 1024 * 2 + # FFN
        768 * 4          # Layer norms
    ) * 2  # 2 layers
    
    conv_params = 2_000_000  # Estimated convolutional approach
    
    print(f"   Standard transformer (2 layers): ~{transformer_params:,}")
    print(f"   Convolutional approach: ~{conv_params:,}")
    print(f"   Mixture-of-bases: {total_trainable:,}")
    
    reduction_vs_transformer = (transformer_params - total_trainable) / transformer_params * 100
    reduction_vs_conv = (conv_params - total_trainable) / conv_params * 100
    
    print(f"\n✅ Efficiency Gains:")
    print(f"   vs Transformer: {reduction_vs_transformer:.1f}% fewer parameters")
    print(f"   vs Convolutional: {reduction_vs_conv:.1f}% fewer parameters")
    
    return total_trainable

if __name__ == "__main__":
    analyze_mixture_architecture()
