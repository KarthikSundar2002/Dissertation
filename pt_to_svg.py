import torch
import os
from lxml import etree
import argparse
from tqdm import tqdm

def format_template(format_path):
    """Load and parse the format.svg template"""
    tree = etree.parse(format_path)
    root = tree.getroot()
    d = etree.tostring(root[1])
    d = d.decode(encoding='utf_8')
    data = d.split()
    template = data
    return template

def filter_stroke(stroke):
    """Filter and normalize stroke data"""
    values = []
    strokes = stroke.tolist()
    for i in strokes:
        for j in range(len(i)):
            i[j] = (i[j] + 1) / 2
        if max(i) < 1 and min(i) > 0:
            values.append(i)
    return values

def rebuild_svg(vectors, template, size, stroke_thickness):
    """Rebuild SVG paths from vector data"""
    svg = []
    for i in vectors:
        template[3] = str(i[0] * size) + ','
        template[4] = str(i[1] * size)
        template[6] = str(i[2] * size) + ','
        template[7] = str(i[3] * size) + ','
        template[8] = str(i[4] * size) + ','
        template[9] = str(i[5] * size)
        template[16] = 'stroke-width="' + str(stroke_thickness) + '"/>\n  '
        svg.append(bytes(' '.join(template), 'utf-8'))
    return svg

def save_svg(svg_elements, size, filename):
    """Save SVG elements to file"""
    new_svg = etree.XML(
        '<svg width="{}" height="{}" version="1.1" xmlns="http://www.w3.org/2000/svg"></svg>'.format(size, size))
    for i in svg_elements:
        new_svg.append(etree.fromstring(i))
    tree = etree.ElementTree(new_svg)
    tree.write(filename, pretty_print=True)

def draw_svg(format_path, size, filename, stroke):
    """Convert stroke data to SVG using the format template"""
    template = format_template(format_path)
    #print(template)
    data = filter_stroke(stroke)
    svg = rebuild_svg(data, template, size, size / 128)
    save_svg(svg, size, filename)

def convert_pt_to_svg(pt_file, format_file, output_dir, size=512, start_idx=0, end_idx=None):
    """
    Convert .pt file data to SVG images
    
    Args:
        pt_file: Path to the .pt file containing stroke data
        format_file: Path to the format.svg template
        output_dir: Directory to save SVG files
        size: Size of the SVG canvas (default: 512)
        start_idx: Starting index for conversion (default: 0)
        end_idx: Ending index for conversion (default: None, converts all)
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the data
    print(f"Loading data from {pt_file}...")
    data = torch.load(pt_file)
    
    # Determine the range to process
    if end_idx is None:
        end_idx = len(data)
    
    print(f"Converting {end_idx - start_idx} samples from index {start_idx} to {end_idx-1}")
    
    # Process each sample
    for idx in tqdm(range(start_idx, end_idx), desc="Converting to SVG"):
        try:
            # Get the stroke data for this sample
            stroke_data = data[idx]
            print(stroke_data.shape)
            # Create filename
            filename = os.path.join(output_dir, f"sample_{idx:06d}.svg")
            
            # Convert to SVG
            draw_svg(format_file, size, filename, stroke_data)
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    print(f"Conversion complete! SVG files saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Convert .pt file data to SVG images')
    parser.add_argument('--pt_file', type=str, default='10k.pt', 
                       help='Path to the .pt file (default: 10k.pt)')
    parser.add_argument('--format_file', type=str, default='format.svg',
                       help='Path to the format.svg template (default: format.svg)')
    parser.add_argument('--output_dir', type=str, default='converted_svgs',
                       help='Output directory for SVG files (default: converted_svgs)')
    parser.add_argument('--size', type=int, default=512,
                       help='Size of SVG canvas (default: 512)')
    parser.add_argument('--start_idx', type=int, default=0,
                       help='Starting index for conversion (default: 0)')
    parser.add_argument('--end_idx', type=int, default=None,
                       help='Ending index for conversion (default: None, converts all)')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples to convert (alternative to end_idx)')
    
    args = parser.parse_args()
    
    # Handle num_samples parameter
    if args.num_samples is not None:
        args.end_idx = args.start_idx + args.num_samples
    
    # Check if files exist
    if not os.path.exists(args.pt_file):
        print(f"Error: PT file {args.pt_file} not found!")
        return
    
    if not os.path.exists(args.format_file):
        print(f"Error: Format file {args.format_file} not found!")
        return
    
    # Convert the data
    convert_pt_to_svg(
        pt_file=args.pt_file,
        format_file=args.format_file,
        output_dir=args.output_dir,
        size=args.size,
        start_idx=args.start_idx,
        end_idx=args.end_idx
    )

if __name__ == "__main__":
    main() 