#!/usr/bin/env python3
"""
Simple script to concatenate all attention_latent files from various sources.
This version uses only standard library modules and numpy.

Usage:
    python concatenate_attention_latent_simple.py [--output OUTPUT_FILE] [--directories DIR1,DIR2,...]
"""

import pickle
import os
import glob
import argparse
from pathlib import Path
import numpy as np
from typing import List, Union, Tuple


def load_tensor_file(file_path: str):
    """
    Load a tensor from either .pt or .pkl file.
    For .pt files, we'll try to load as pickle first, then as numpy.
    
    Args:
        file_path: Path to the tensor file
        
    Returns:
        Loaded data (numpy array or tensor-like object)
    """
    try:
        if file_path.endswith('.pkl'):
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                # Convert to numpy array if possible
                if hasattr(data, 'numpy'):
                    return data.numpy()
                elif hasattr(data, 'detach'):
                    return data.detach().cpu().numpy()
                else:
                    return np.array(data)
        elif file_path.endswith('.pt'):
            # Try to load as pickle first (PyTorch files are often pickle-based)
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    if hasattr(data, 'numpy'):
                        return data.numpy()
                    elif hasattr(data, 'detach'):
                        return data.detach().cpu().numpy()
                    else:
                        return np.array(data)
            except:
                # If pickle fails, try to read as binary and convert
                with open(file_path, 'rb') as f:
                    # This is a fallback - might not work for all .pt files
                    print(f"Warning: Could not load {file_path} as pickle, skipping...")
                    return None
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
            os.path.join(directory, "attention_latent_*.pkl"),
            os.path.join(directory, "attention_latent.pt"),
            os.path.join(directory, "attention_latent.pkl"),
        ]
        
        for pattern in patterns:
            found_files = glob.glob(pattern)
            files.extend(found_files)
    
    # Remove duplicates and sort
    files = sorted(list(set(files)))
    return files


def concatenate_arrays(array_files: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Concatenate arrays from multiple files.
    
    Args:
        array_files: List of file paths containing arrays
        
    Returns:
        Tuple of (concatenated_array, list_of_loaded_files)
    """
    arrays = []
    loaded_files = []
    failed_files = []
    
    print(f"Found {len(array_files)} attention_latent files:")
    for file_path in array_files:
        print(f"  - {file_path}")
    
    print("\nLoading arrays...")
    for file_path in array_files:
        array = load_tensor_file(file_path)
        if array is not None:
            arrays.append(array)
            loaded_files.append(file_path)
            print(f"  ✓ Loaded {file_path}: shape {array.shape}")
        else:
            failed_files.append(file_path)
            print(f"  ✗ Failed to load {file_path}")
    
    if not arrays:
        raise ValueError("No arrays were successfully loaded!")
    
    if failed_files:
        print(f"\nWarning: Failed to load {len(failed_files)} files:")
        for file_path in failed_files:
            print(f"  - {file_path}")
    
    print(f"\nConcatenating {len(arrays)} arrays...")
    
    # Check if all arrays have compatible shapes
    shapes = [a.shape for a in arrays]
    print(f"Array shapes: {shapes}")
    
    # Concatenate along the first dimension (batch dimension)
    concatenated = np.concatenate(arrays, axis=0)
    
    print(f"Concatenated array shape: {concatenated.shape}")
    print(f"Total elements: {concatenated.size}")
    
    return concatenated, loaded_files


def save_concatenated_array(array: np.ndarray, output_path: str, format: str = 'npy'):
    """
    Save the concatenated array to file.
    
    Args:
        array: The concatenated array
        output_path: Output file path
        format: Output format ('npy', 'pkl', or 'npz')
    """
    if format == 'npy':
        np.save(output_path, array)
    elif format == 'pkl':
        with open(output_path, 'wb') as f:
            pickle.dump(array, f)
    elif format == 'npz':
        np.savez_compressed(output_path, attention_latent=array)
    else:
        raise ValueError(f"Unsupported output format: {format}")
    
    print(f"Saved concatenated array to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Concatenate attention_latent files (simple version)')
    parser.add_argument('--output', '-o', default='concatenated_attention_latent.npy',
                       help='Output file path (default: concatenated_attention_latent.npy)')
    parser.add_argument('--directories', '-d', 
                       default='/scratch/ks02450,/user/HS400/ks02450/Dissertation',
                       help='Comma-separated list of directories to search')
    parser.add_argument('--format', '-f', choices=['npy', 'pkl', 'npz'], default='npy',
                       help='Output format (default: npy)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Parse directories
    directories = [d.strip() for d in args.directories.split(',')]
    
    print("Attention Latent Concatenation Script (Simple Version)")
    print("=" * 50)
    print(f"Searching directories: {directories}")
    print(f"Output file: {args.output}")
    print(f"Output format: {args.format}")
    print()
    
    try:
        # Find all attention_latent files
        array_files = find_attention_latent_files(directories)
        
        if not array_files:
            print("No attention_latent files found!")
            return
        
        # Concatenate arrays
        concatenated_array, loaded_files = concatenate_arrays(array_files)
        
        # Save the result
        save_concatenated_array(concatenated_array, args.output, args.format)
        
        print(f"\nSuccessfully concatenated {len(loaded_files)} files!")
        print(f"Output saved to: {args.output}")
        
        # Show summary
        print(f"\nSummary:")
        print(f"  - Files processed: {len(loaded_files)}")
        print(f"  - Final array shape: {concatenated_array.shape}")
        print(f"  - Data type: {concatenated_array.dtype}")
        print(f"  - Memory usage: {concatenated_array.nbytes / (1024**2):.2f} MB")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
