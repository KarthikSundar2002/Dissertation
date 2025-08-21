# PT to SVG Converter

This script converts stroke data from `.pt` files into SVG images using a format template.

## Overview

The script loads stroke data from a PyTorch `.pt` file (like `10k.pt`) and converts each data point into an SVG image using the `format.svg` template. Each stroke is represented as a quadratic Bézier curve in the SVG.

## Data Format

The `.pt` file contains a list of tensors, where each tensor represents a drawing:
- Each tensor has shape `[N, 6]` where N is the number of strokes
- Each stroke is represented by 6 values: `[x1, y1, x2, y2, x3, y3]`
- These represent the control points of a quadratic Bézier curve
- The values are normalized to the range [-1, 1]

## Usage

### Command Line Interface

```bash
# Convert all samples
python pt_to_svg_converter.py

# Convert first 100 samples
python pt_to_svg_converter.py --start_idx 0 --end_idx 100

# Convert 50 samples starting from index 500
python pt_to_svg_converter.py --start_idx 500 --num_samples 50

# Use different output directory and size
python pt_to_svg_converter.py --output_dir my_svgs --size 256
```

### Command Line Arguments

- `--pt_file`: Path to the .pt file (default: `10k.pt`)
- `--format_file`: Path to the format.svg template (default: `format.svg`)
- `--output_dir`: Output directory for SVG files (default: `converted_svgs`)
- `--size`: Size of SVG canvas (default: 512)
- `--start_idx`: Starting index for conversion (default: 0)
- `--end_idx`: Ending index for conversion (default: None, converts all)
- `--num_samples`: Number of samples to convert (alternative to end_idx)

### Python API

```python
from pt_to_svg_converter import convert_pt_to_svg

# Convert first 10 samples
convert_pt_to_svg(
    pt_file='10k.pt',
    format_file='format.svg',
    output_dir='my_output',
    size=512,
    start_idx=0,
    end_idx=10
)
```

## Example Usage

See `example_usage.py` for complete examples of how to use the converter.

## Output

The script creates SVG files named `sample_XXXXXX.svg` where XXXXXX is the zero-padded index number. Each SVG file contains:

- A 512x512 (or custom size) canvas
- Multiple path elements representing the strokes
- Each stroke is a quadratic Bézier curve with black color and rounded line caps

## Dependencies

- `torch`: For loading .pt files
- `lxml`: For XML/SVG processing
- `tqdm`: For progress bars
- `argparse`: For command line argument parsing

## Installation

```bash
pip install torch lxml tqdm
```

## Notes

- The script automatically creates the output directory if it doesn't exist
- Invalid stroke data (outside the [0,1] range after normalization) is filtered out
- The stroke width is calculated as `size / 128`
- The script includes error handling for individual samples that fail to process 