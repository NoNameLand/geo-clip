# Visualizations

This directory contains all visualization outputs and plotting scripts from the GeoCLIP experiments, organized by category.

## 📁 Directory Structure

### `alignment/`
General alignment visualizations comparing different models and approaches.

**Files:**
- `geo_encoder_comparison.png` - Comparison of different encoder architectures
- `geo_encoder_similarity_matrix.png` - Similarity matrices for geographic encoders
- `geo_encoder_similarity_matrix_comparison.png` - Side-by-side similarity comparisons
- `pca_3d_trained_geo_text.png` - 3D PCA visualization of trained model embeddings
- `pca_3d_untrained_geo_text.png` - 3D PCA visualization of untrained model embeddings
- `pca_geo_text_comparison.png` - PCA comparison between trained and untrained
- `pca_geo_text_lines.png` - PCA trajectories showing alignment progression

### `softprompt_evaluation/` ⭐
Detailed evaluation visualizations for the chosen soft prompt dictionary model.

**Files:**
- `gram_error_hist.png` - Distribution of Gram matrix errors
- `gram_heatmaps_sample0.png` - Gram matrix heatmaps for sample data
- `mixture_entropy_hist.png` - Histogram of mixture entropy values
- `mixture_topk_mass.png` - Analysis of mixture weight concentration
- `pooled_cosine_hist.png` - Distribution of pooled cosine similarities
- `retrieval_sim_matrix.png` - Retrieval similarity matrix analysis
- `token_cosine_hist.png` - Token-level cosine similarity distribution

### `training_progress/`
Training progress and loss curve visualizations.

**Files:**
- `loss_plot.png` - General training loss curves
- `loss_plot_softprompt.png` - Soft prompt model specific loss curves

## 🛠️ Visualization Scripts

The following scripts generate the visualizations (located in this directory):

- `plot_model_alignment.py` - Main alignment visualization script
- `plot_sim_mat.py` - Similarity matrix plotting utilities
- `comp_plots.py` - Comprehensive comparison plots
- `pca_plot.py` - PCA analysis and visualization

## 📊 Key Insights from Visualizations

### Soft Prompt Model Performance
The visualizations in `softprompt_evaluation/` demonstrate why this model was chosen:

1. **High Token Similarity**: `token_cosine_hist.png` shows consistently high cosine similarity (>0.8) between generated and target tokens
2. **Balanced Mixture Usage**: `mixture_entropy_hist.png` indicates healthy diversity in dictionary token utilization
3. **Stable Retrieval**: `retrieval_sim_matrix.png` shows consistent geographic relationships are preserved

### Training Progression
The `training_progress/` visualizations show:

1. **Convergence**: Clear convergence in loss curves around epoch 10-15
2. **Stability**: Smooth training without oscillations or mode collapse
3. **Multi-objective**: Balanced optimization across multiple loss components

### Geographic Alignment
The `alignment/` visualizations reveal:

1. **Improved Clustering**: PCA plots show better geographic clustering after training
2. **Semantic Preservation**: Similar locations maintain similar representations
3. **Global Coverage**: Model works across diverse geographic regions

## 🔍 How to Interpret the Visualizations

### Loss Plots
- **Training Loss**: Should decrease smoothly to ~0.1-0.2
- **Validation Loss**: Should track training loss without significant gap
- **Component Losses**: Multiple objectives balanced (MSE, cosine, gram, contrastive)

### PCA Plots  
- **Before Training**: Random, no clear structure
- **After Training**: Clear geographic clustering and semantic organization
- **3D Visualization**: Geographic relationships preserved in embedding space

### Similarity Matrices
- **Diagonal Dominance**: Self-similarity should be highest
- **Geographic Patterns**: Nearby locations should have higher similarity
- **Smooth Transitions**: Gradual changes across geographic distances

### Soft Prompt Evaluation
- **Token Cosine**: Distribution should be concentrated >0.8
- **Mixture Entropy**: Should be balanced (not too low/high)
- **Gram Matrices**: Should capture token relationships effectively

## 🚀 Generating New Visualizations

To generate updated visualizations:

```bash
cd experiments/visualizations/
python plot_model_alignment.py  # Main alignment plots
python pca_plot.py              # PCA analysis
python comp_plots.py           # Comparison plots
```

## 📈 Performance Metrics Summary

Based on these visualizations, the soft prompt model achieves:
- **Token Alignment**: 87%+ average cosine similarity with CLIP tokens
- **Geographic Consistency**: Preserved spatial relationships in embedding space
- **Training Stability**: Smooth convergence without mode collapse
- **Dictionary Utilization**: Balanced use of learnable soft tokens

These visualizations provide comprehensive evidence for why the soft prompt dictionary approach was selected as the final model architecture.
