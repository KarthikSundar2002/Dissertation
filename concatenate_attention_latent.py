#!/usr/bin/env python3
"""
Script to concatenate all attention_latent files from various sources.

This script can handle:
1. attention_latent_{i}.pt files from /scratch/ks02450/ directory
2. attention_latent.pkl files from current directory
3. Any other attention_latent files in specified directories

Usage:
    python concatenate_attention_latent.py [--output OUTPUT_FILE] [--directories DIR1,DIR2,...]
"""

import torch
import pickle
import os
import glob
import argparse
from pathlib import Path
import numpy as np
from typing import List, Union, Tuple


def load_tensor_file(file_path: str) -> torch.Tensor:
    """
    Load a tensor from either .pt or .pkl file.
    
    Args:
        file_path: Path to the tensor file
        
    Returns:
        torch.Tensor: Loaded tensor
    """
    try:
        if file_path.endswith('.pt'):
            return torch.load(file_path, map_location='cpu')
        elif file_path.endswith('.pkl'):
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, torch.Tensor):
                    return data
                else:
                    # Convert to tensor if it's not already
                    return torch.tensor(data)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def find_attention_latent_files(directories: List[str]) -> List[str]:
    """
    Find all attention_latent files in the specified directories.
    
    Args:
        directories: List of directories to search
        
    Returns:
        List of file paths
    """
    files = []
    
    for directory in directories:
        if not os.path.exists(directory):
            print(f"Warning: Directory {directory} does not exist, skipping...")
            continue
            
        # Search for various patterns
        patterns = [
            os.path.join(directory, "attention_latent_*.pt"),
            #os.path.join(directory, "attention_latent_*.pkl"),
            os.path.join(directory, "attention_latent.pt"),
            #os.path.join(directory, "attention_latent.pkl"),
        ]
        
        for pattern in patterns:
            found_files = glob.glob(pattern)
            files.extend(found_files)
    
    # Remove duplicates and sort
    files = sorted(list(set(files)))
    return files


def concatenate_tensors(tensor_files: List[str]) -> Tuple[torch.Tensor, List[str]]:
    """
    Concatenate tensors from multiple files.
    
    Args:
        tensor_files: List of file paths containing tensors
        
    Returns:
        Tuple of (concatenated_tensor, list_of_loaded_files)
    """
    tensors = []
    loaded_files = []
    failed_files = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Found {len(tensor_files)} attention_latent files:")
    for file_path in tensor_files:
        print(f"  - {file_path}")
    
    print("\nLoading tensors...")
    for file_path in tensor_files:
        tensor = load_tensor_file(file_path)
        tensor = tensor.to(device)
        if tensor is not None:
            tensors.append(tensor)
            loaded_files.append(file_path)
            print(f"  ✓ Loaded {file_path}: shape {tensor.shape}")
        else:
            failed_files.append(file_path)
            print(f"  ✗ Failed to load {file_path}")
    
    if not tensors:
        raise ValueError("No tensors were successfully loaded!")
    
    if failed_files:
        print(f"\nWarning: Failed to load {len(failed_files)} files:")
        for file_path in failed_files:
            print(f"  - {file_path}")
    
    print(f"\nConcatenating {len(tensors)} tensors...")
    
    # Check if all tensors have compatible shapes
    shapes = [t.shape for t in tensors]
    print(f"Tensor shapes: {shapes}")
    
    # Concatenate along the first dimension (batch dimension)
    concatenated = torch.cat(tensors, dim=0)
    
    print(f"Concatenated tensor shape: {concatenated.shape}")
    print(f"Total elements: {concatenated.numel()}")
    
    return concatenated, loaded_files


def save_concatenated_tensor(tensor: torch.Tensor, output_path: str, format: str = 'pt'):
    """
    Save the concatenated tensor to file.
    
    Args:
        tensor: The concatenated tensor
        output_path: Output file path
        format: Output format ('pt' or 'pkl')
    """
    if format == 'pt':
        torch.save(tensor, output_path)
    elif format == 'pkl':
        with open(output_path, 'wb') as f:
            pickle.dump(tensor, f)
    else:
        raise ValueError(f"Unsupported output format: {format}")
    
    print(f"Saved concatenated tensor to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Concatenate attention_latent files')
    parser.add_argument('--output', '-o', default='concatenated_attention_latent.pt',
                       help='Output file path (default: concatenated_attention_latent.pt)')
    parser.add_argument('--directories', '-d', 
                       default='/scratch/ks02450,/user/HS400/ks02450/Dissertation',
                       help='Comma-separated list of directories to search (default: /scratch/ks02450,/user/HS400/ks02450/Dissertation)')
    parser.add_argument('--format', '-f', choices=['pt', 'pkl'], default='pt',
                       help='Output format (default: pt)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Parse directories
    directories = [d.strip() for d in args.directories.split(',')]
    
    print("Attention Latent Concatenation Script")
    print("=" * 40)
    print(f"Searching directories: {directories}")
    print(f"Output file: {args.output}")
    print(f"Output format: {args.format}")
    print()
    
    try:
        # Find all attention_latent files
        tensor_files = find_attention_latent_files(directories)
        
        if not tensor_files:
            print("No attention_latent files found!")
            return
        
        # Concatenate tensors
        concatenated_tensor, loaded_files = concatenate_tensors(tensor_files)
        
        # Save the result
        save_concatenated_tensor(concatenated_tensor, args.output, args.format)
        
        print(f"\nSuccessfully concatenated {len(loaded_files)} files!")
        print(f"Output saved to: {args.output}")
        
        # Show summary
        print(f"\nSummary:")
        print(f"  - Files processed: {len(loaded_files)}")
        print(f"  - Final tensor shape: {concatenated_tensor.shape}")
        print(f"  - Data type: {concatenated_tensor.dtype}")
        print(f"  - Device: {concatenated_tensor.device}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
