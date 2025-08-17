# Experiments

This directory contains experimental code and research components used in the development and evaluation of GeoCLIP. These are primarily for research purposes and advanced users.

## Directory Structure

### `training_variants/` ⭐
**Contains the CHOSEN soft prompt dictionary model and alternative training approaches.**
- `train_softprompt_model.py` - **Main training script for the chosen model**
- `softprompt_config.py` - Optimal hyperparameters and configuration
- `softprompt_inference.py` - Inference script for the trained model
- `README.md` - Detailed comparison of all training approaches
- Other training variants and experimental architectures

### `alignment/`
Experiments with alignment between GPS coordinates and text descriptions.
- Training scripts for geo-text alignment models
- Dataset preparation and generation tools  
- Model checkpoints (moved training visualizations to `visualizations/`)

### `analysis/`
Analysis scripts for model evaluation and comparison.
- Parameter analysis tools
- Model comparison utilities
- Performance evaluation scripts
- Architecture analysis tools

### `stable_diffusion/`
Experiments combining GeoCLIP with Stable Diffusion.
- Custom embedding models for geographic conditioning
- Integration tests with diffusion pipelines
- Geographic prompt engineering experiments

### `visualizations/` 🎨
**All visualization outputs and plotting scripts, organized by category.**
- `alignment/` - General alignment visualizations
- `softprompt_evaluation/` - Detailed evaluation plots for chosen model
- `training_progress/` - Loss curves and training monitoring
- Plotting scripts and utilities

## Usage Notes

⚠️ **Important**: The code in this directory is experimental and may:
- Require specific dataset configurations
- Have dependencies not listed in main requirements
- Be work-in-progress implementations
- Not be fully documented or stable

These experiments were used in the research and development of GeoCLIP but are not part of the main stable API.

## Running Experiments

Each experiment directory may have its own requirements and setup. Generally:

1. Check for experiment-specific README files
2. Ensure required datasets are available
3. Install any additional dependencies
4. Review and modify paths/configurations as needed

## Research Context

These experiments relate to various aspects explored in the GeoCLIP research:

- **Alignment**: How to align geographic and semantic representations
- **Architecture**: Exploring efficient model designs
- **Integration**: Using GeoCLIP embeddings in other applications
- **Evaluation**: Comprehensive testing and analysis methodologies

## Contributing

When adding new experiments:
1. Place them in the appropriate subdirectory
2. Include a brief README explaining the purpose
3. Document any special requirements or datasets needed
4. Use clear, descriptive filenames
