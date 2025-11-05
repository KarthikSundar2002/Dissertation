#!/usr/bin/env python3
"""
Simple test script for the autoencoder implementation
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import pytorch_lightning as L
    from torch import optim
    import numpy as np
    print("✓ All imports successful")
    
    # Import our autoencoder
    from networks.model.ae import AutoencoderLightning, test_autoencoder
    
    print("✓ Autoencoder import successful")
    
    # Test the autoencoder
    test_autoencoder()
    
    print("✓ Autoencoder test completed successfully!")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Please ensure PyTorch and PyTorch Lightning are installed")
except Exception as e:
    print(f"✗ Error during testing: {e}")
    import traceback
    traceback.print_exc()
