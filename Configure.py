# Configure.py
import torch

def get_config():
    return {
        'image_size': 32,
        'patch_size': 4,
        'in_channels': 3,
        'embed_dim': 512,
        'num_heads': 8,
        'mlp_dim': 1024,
        'num_layers': 4,
        'num_classes': 10,
        'dropout': 0.1,
        'batch_size': 32,
        'learning_rate': 0.003,
        'weight_decay': 0.0001,
        'num_epochs': 300,
        'device': "cuda" 
    }

