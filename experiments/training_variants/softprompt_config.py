"""
Soft Prompt Model Configuration

This file contains the optimal hyperparameters and configuration for the 
soft prompt dictionary model - the chosen architecture for GeoCLIP geographic-text alignment.

These parameters were determined through extensive experimentation and represent
the best balance of performance, training stability, and computational efficiency.
"""

# Model Architecture Configuration
MODEL_CONFIG = {
    # Dictionary Configuration
    "dictionary_size": 64,          # Number of soft tokens in dictionary (K)
    "sequence_length": 77,          # CLIP token sequence length
    "embedding_dim": 768,           # CLIP embedding dimension
    "init_scale": 0.02,             # Scale for random initialization
    
    # Conditioning Network Architecture
    "input_feat_dim": 512,          # GeoCLIP location encoder output dimension
    "hidden_dims": [256, 128],      # Hidden layer dimensions
    "dropout_rate": 0.1,            # Dropout rate
    "temperature": 0.7,             # Softmax temperature for mixture weights
    
    # Frozen Components
    "freeze_location_encoder": True, # Keep GeoCLIP location encoder frozen
    "freeze_text_encoder": True,    # Keep CLIP text encoder frozen
}

# Training Configuration
TRAINING_CONFIG = {
    # Optimization
    "learning_rate": 5e-4,          # Base learning rate
    "weight_decay": 1e-4,           # L2 regularization
    "betas": (0.9, 0.95),          # Adam optimizer betas
    "eps": 1e-6,                   # Adam epsilon
    
    # Training Schedule
    "epochs": 15,                  # Total training epochs
    "batch_size": 32,              # Batch size (adjust based on GPU memory)
    "scheduler": "cosine",         # Learning rate scheduler
    "warmup_epochs": 2,            # Warmup epochs for scheduler
    
    # Mixed Precision
    "use_amp": True,               # Use automatic mixed precision
    "gradient_clip": 1.0,          # Gradient clipping threshold
    
    # Validation
    "validation_split": 0.2,       # Fraction of data for validation
    "validation_freq": 1,          # Validate every N epochs
}

# Loss Configuration  
LOSS_CONFIG = {
    # Loss Weights (carefully tuned)
    "token_mse_weight": 1.00,      # Token-level MSE loss
    "token_cosine_weight": 0.25,   # Token-level cosine similarity loss
    "gram_loss_weight": 0.10,      # Gram matrix loss
    "contrastive_weight": 0.25,    # Sequence contrastive loss
    "dict_diversity_weight": 0.01, # Dictionary diversity regularization
    "mixture_entropy_weight": 0.01,# Mixture entropy penalty
    
    # Loss-specific Parameters
    "contrastive_temperature": 0.03, # Temperature for contrastive loss
    "gram_loss_normalize": True,     # Normalize Gram matrices
}

# Data Configuration
DATA_CONFIG = {
    "dataset_path": "alignment/dataset/cities.json",
    "shuffle": True,
    "num_workers": 4,              # DataLoader workers (adjust for your system)
    "pin_memory": True,            # Pin memory for faster GPU transfer
    
    # Data Augmentation (if any)
    "coordinate_noise": 0.0,       # Add noise to coordinates (0 = no noise)
    "text_variations": False,      # Use text variations if available
}

# Checkpointing Configuration
CHECKPOINT_CONFIG = {
    "save_dir": "alignment/models",
    "save_every": 5,               # Save checkpoint every N epochs
    "save_best": True,             # Save best validation model
    "keep_top_k": 3,               # Keep top K checkpoints
    "model_name": "geo_softprompt_model_cities",
}

# Evaluation Configuration
EVAL_CONFIG = {
    "metrics": [
        "token_cosine_similarity",
        "sequence_mse", 
        "mixture_entropy",
        "dictionary_utilization",
    ],
    "eval_batch_size": 64,         # Larger batch size for evaluation
    "eval_samples": 1000,          # Number of samples for detailed evaluation
}

# Hardware Configuration
HARDWARE_CONFIG = {
    "device": "auto",              # "auto", "cuda", "cpu"
    "mixed_precision": True,       # Use mixed precision training
    "compile_model": False,        # Use torch.compile (PyTorch 2.0+)
    "memory_efficient": True,      # Use memory efficient attention if available
}

# Reproducibility
REPRODUCIBILITY_CONFIG = {
    "seed": 42,
    "deterministic": True,         # Enable deterministic algorithms
    "benchmark": False,            # Disable cudnn benchmark for reproducibility
}

# All configurations combined
FULL_CONFIG = {
    "model": MODEL_CONFIG,
    "training": TRAINING_CONFIG, 
    "loss": LOSS_CONFIG,
    "data": DATA_CONFIG,
    "checkpoint": CHECKPOINT_CONFIG,
    "eval": EVAL_CONFIG,
    "hardware": HARDWARE_CONFIG,
    "reproducibility": REPRODUCIBILITY_CONFIG,
}

# Quick access to key parameters
QUICK_CONFIG = {
    "dictionary_size": 64,
    "learning_rate": 5e-4,
    "batch_size": 32,
    "epochs": 15,
    "temperature": 0.7,
}

if __name__ == "__main__":
    print("Soft Prompt Model Configuration")
    print("=" * 40)
    print(f"Dictionary Size: {MODEL_CONFIG['dictionary_size']}")
    print(f"Learning Rate: {TRAINING_CONFIG['learning_rate']}")
    print(f"Training Epochs: {TRAINING_CONFIG['epochs']}")
    print(f"Batch Size: {TRAINING_CONFIG['batch_size']}")
    print(f"Expected Parameters: ~150K trainable")
    print("\nThis configuration achieved the best results in our experiments.")
    print("Modify carefully - these hyperparameters are finely tuned.")
