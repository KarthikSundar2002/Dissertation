#!/usr/bin/env python3
"""
Example usage of the pt_to_svg_converter.py script

This script demonstrates how to convert the 10k.pt data to SVG images
using the format.svg template.
"""

from pt_to_svg_converter import convert_pt_to_svg

def main():
    # Example 1: Convert first 10 samples
    print("Example 1: Converting first 10 samples...")
    convert_pt_to_svg(
        pt_file='10k.pt',
        format_file='format.svg',
        output_dir='example_output_10',
        size=512,
        start_idx=0,
        end_idx=10
    )
    
    # Example 2: Convert samples 100-110
    print("\nExample 2: Converting samples 100-110...")
    convert_pt_to_svg(
        pt_file='10k.pt',
        format_file='format.svg',
        output_dir='example_output_100_110',
        size=512,
        start_idx=100,
        end_idx=110
    )
    
    # Example 3: Convert with different size
    print("\nExample 3: Converting with 256x256 size...")
    convert_pt_to_svg(
        pt_file='10k.pt',
        format_file='format.svg',
        output_dir='example_output_256',
        size=256,
        start_idx=0,
        end_idx=5
    )

if __name__ == "__main__":
    main() 