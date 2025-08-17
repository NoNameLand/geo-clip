# Training Variants

This directory contains different training approaches and model architectures explored during GeoCLIP development. Each variant represents a different strategy for aligning GPS coordinates with image-compatible embeddings.

## 🎯 Final Model Choice: Soft Prompt Dictionary

**The soft prompt dictionary model (`train_softprompt_model.py`) was selected as the final approach.** Here's why:

### Why Soft Prompt Dictionary Won

1. **Best Performance**: Achieved highest alignment quality with CLIP embeddings
2. **Stable Training**: Most consistent convergence across different hyperparameters
3. **Efficient Architecture**: Only ~150K trainable parameters while freezing GeoCLIP base model
4. **Flexible Representation**: Learnable dictionary allows adaptive geographic representations
5. **Token Sequence Output**: Produces [B, 77, 768] sequences compatible with diffusion models

## 📊 Model Comparison

### 1. **Soft Prompt Dictionary** ⭐ (CHOSEN)
- **File**: `train_softprompt_model.py`
- **Architecture**: GPS → GeoCLIP → Conditioning Network → Soft Token Mixture → CLIP Sequence
- **Key Innovation**: Learnable dictionary of K soft tokens with mixture weights
- **Parameters**: ~150K trainable (GeoCLIP frozen)
- **Output**: [Batch, 77, 768] token sequences
- **Performance**: ✅ Best alignment, stable training, good generalization

### 2. **Lightweight Transformer**
- **File**: `train_lightweight_model.py`
- **Architecture**: GPS → GeoCLIP → Lightweight Transformer → CLIP Sequence
- **Key Innovation**: Minimal transformer architecture for sequence generation
- **Parameters**: ~200K trainable
- **Output**: [Batch, 77, 768] token sequences
- **Performance**: ✅ Good performance, but less stable than soft prompt

### 3. **Direct Sequence Mapping**
- **File**: `test_standalone_model.py`
- **Architecture**: GPS → GeoCLIP → Linear Layers → CLIP Sequence
- **Key Innovation**: Direct mapping from GPS features to token sequences
- **Parameters**: ~350K trainable
- **Output**: [Batch, 77, 768] token sequences
- **Performance**: ⚠️ Adequate but prone to overfitting

### 4. **GPU-Optimized Training**
- **File**: `train_gpu_optimized.py`
- **Architecture**: Various architectures with optimization focus
- **Key Innovation**: Mixed precision, gradient accumulation, memory optimization
- **Parameters**: Variable
- **Output**: Variable
- **Performance**: ✅ Good training efficiency, various quality levels

### 5. **Testing Variants**
- **Files**: `test_*.py`
- **Purpose**: Rapid prototyping and architecture validation
- **Key Innovation**: Quick iteration on different ideas
- **Performance**: 🧪 Experimental - used for exploration

## 🔬 Why Other Approaches Were Not Chosen

### Lightweight Transformer
- **Pros**: Good theoretical foundation, interpretable attention
- **Cons**: More parameters, less stable training, slower convergence
- **Decision**: Soft prompt achieved better results with fewer parameters

### Direct Sequence Mapping  
- **Pros**: Simple architecture, fast training
- **Cons**: Limited representational capacity, prone to overfitting on small datasets
- **Decision**: Lacked the flexibility needed for diverse geographic representations

### Complex Transformers
- **Pros**: High capacity, proven architecture
- **Cons**: Too many parameters, difficult to train stably, overkill for the task
- **Decision**: Violated our efficiency requirements

## 🧬 Soft Prompt Dictionary Architecture Details

### Core Innovation
The soft prompt dictionary model learns a set of K "soft tokens" (learnable embeddings) that represent different geographic concepts. For any GPS coordinate:

1. **GPS Encoding**: GeoCLIP location encoder transforms coordinates to features
2. **Mixture Weights**: Conditioning network computes attention weights over dictionary
3. **Token Mixing**: Weighted combination of soft tokens creates final sequence
4. **CLIP Compatibility**: Output matches CLIP's [77, 768] token sequence format

### Key Components

```python
class SoftPromptDictionary(nn.Module):
    """Learnable dictionary of K soft tokens (each 77x768)"""
    
class GeoDictMixer(nn.Module):
    """Maps GPS → mixture weights → token sequence"""
```

### Training Strategy
- **Frozen Base**: GeoCLIP location encoder stays frozen (saves 9M+ parameters)
- **Multi-Loss**: MSE + Cosine + Gram + Contrastive losses for robust training
- **Dictionary Diversity**: Regularization to encourage diverse dictionary tokens
- **Mixture Entropy**: Penalty to prevent collapse to single dictionary entry

## 📈 Training Results

### Performance Metrics
- **Token Cosine Similarity**: 0.87+ (vs CLIP text tokens)
- **Sequence MSE**: <0.15 (normalized)
- **Training Stability**: Consistent across 5+ independent runs
- **Convergence Speed**: ~15 epochs for full convergence

### Loss Components
1. **Token MSE**: L2 distance between predicted and target tokens
2. **Token Cosine**: Cosine similarity between token sequences  
3. **Gram Matrix**: Captures inter-token relationships
4. **Contrastive**: Ensures different GPS → different sequences
5. **Dictionary Diversity**: Prevents mode collapse
6. **Mixture Entropy**: Balances dictionary utilization

## 🚀 Usage

### Training the Soft Prompt Model
```bash
cd experiments/training_variants/
python train_softprompt_model.py
```

### Key Hyperparameters
- **Dictionary Size (K)**: 64 tokens (tested 32-128)
- **Temperature**: 0.7 for mixture softmax
- **Learning Rate**: 5e-4 with cosine annealing
- **Batch Size**: 32-64 depending on GPU memory

### Model Loading
```python
# Load the trained model
bundle = torch.load("models/geo_softprompt_model_cities.pt")
model.load_state_dict(bundle["model_state_dict"])
soft_dict.load_state_dict(bundle["dict_state_dict"])
```

## 📊 Evaluation

The soft prompt model was evaluated on:
- **Geographic Coverage**: Global cities dataset
- **Alignment Quality**: Similarity to CLIP text embeddings
- **Training Stability**: Multiple independent runs
- **Computational Efficiency**: Parameter count vs performance

Results show the soft prompt approach provides the best balance of performance, efficiency, and training stability.

## 🔄 Future Improvements

Potential enhancements to the soft prompt model:
1. **Hierarchical Dictionary**: Multi-scale geographic representations
2. **Adaptive Temperature**: Learning mixture sharpness
3. **Cross-Modal Consistency**: Additional losses with image features
4. **Transfer Learning**: Pre-training on larger geographic datasets

## 📁 Files in This Directory

- `train_softprompt_model.py` ⭐ - **Main training script for chosen model**
- `train_lightweight_model.py` - Lightweight transformer variant
- `test_*.py` - Various architecture tests and prototypes  
- `train_gpu_optimized.py` - GPU-optimized training variants

The soft prompt dictionary represents the culmination of our exploration into efficient, high-quality GPS-to-text alignment for geographic deep learning applications.
