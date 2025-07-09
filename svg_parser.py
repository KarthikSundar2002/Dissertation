from lxml import etree
import torch
from os import listdir
from os.path import join
import re

def parse_path_data(d):
    """Parse SVG path data into a list of coordinates"""
    # Split the path data into commands and their parameters
    commands = re.findall(r'([A-Za-z])([^A-Za-z]*)', d)
    points = [1.0,]
    
    for cmd, params in commands:
        params = [float(x) for x in params.strip().split()]
        if cmd == 'C':  # Cubic Bezier curve
            # C command has 6 parameters: x1,y1 x2,y2 x,y
            points.extend([params[0], params[1], params[2], params[3], params[4], params[5]])
        elif cmd == 'M':  # Move to
            points.extend([params[0], params[1]])
        elif cmd == 'L':  # Line to
            points.extend([params[0], params[1]])
    
    return points

def parse_circle(cx, cy, r):
    """Convert circle to a series of points"""
    # Create 8 points around the circle
    points = [-1.0]
    for i in range(8):
        angle = i * (2 * 3.14159 / 8)
        x = cx + r * torch.cos(torch.tensor(angle))
        y = cy + r * torch.sin(torch.tensor(angle))
        points.extend([x, y])
    return points

def parse_rect(x, y, width, height):
    """Convert rectangle to a series of points"""
    # Return the four corners of the rectangle
    return [x, y, x + width, y, x + width, y + height, x, y + height]

def normalize_points(points, size=256):
    """Normalize points to [-1, 1] range"""
    normalized = []
    for point in points:
        normalized.append((point / size) * 2 - 1)
    return normalized

def pad_and_type(points, cmd_type):
    # Pad or trim to 6 points, then add the command type as the 7th attribute
    padded = list(points[:6]) + [-1.0] * (6 - len(points))
    padded.append(float(cmd_type))
    return padded


def parse_svg_element(element, size=256):
    """Recursively parse SVG elements and return a list of [6 points + type] for all shapes found."""
    all_points = []
    # Handle shape elements
    if element.tag.endswith('path'):
        d = element.get('d')
        points = parse_path_data(d)
        normalized = normalize_points(points, size)
        # Split normalized into chunks of 6, pad if needed, type=0
        for i in range(0, len(normalized), 6):
            chunk = normalized[i:i+6]
            all_points.append(pad_and_type(chunk, 0))
    elif element.tag.endswith('circle'):
        cx = float(element.get('cx', 0))
        cy = float(element.get('cy', 0))
        r = float(element.get('r', 0))
        points = [cx, cy, r]
        normalized = normalize_points(points, size)
        # For circle, type=1
        all_points.append(pad_and_type(normalized, 1))
    # elif element.tag.endswith('rect'):
        
        # x = float(element.get('x', 0))
        # y = float(element.get('y', 0))
        # width = float(element.get('width', 0))
        # height = float(element.get('height', 0))
        # points = parse_rect(x, y, width, height)
        # normalized = normalize_points(points, size)
        # # For rect, type=2
        # all_points.append(pad_and_type(normalized, 2))
    # Recursively handle group elements
    elif element.tag.endswith('g') or element.tag.endswith('svg'):
        for child in element:
            all_points.extend(parse_svg_element(child, size))
    # Ignore <defs> and other non-shape elements
    return all_points


def parse_svg_file(file_path, size=256):
    """Parse an SVG file and return a tensor of shape (N, 7)"""
    tree = etree.parse(file_path)
    root = tree.getroot()
    all_points = parse_svg_element(root, size)
    return torch.FloatTensor(all_points)

def process_svg_directory(input_dir, output_file, size=256):
    """Process all SVG files in a directory and save as a single PyTorch tensor of shape [num_images, N, 7]"""
    all_shapes = []
    max_len = 0
    # First, parse all SVGs and find the max N
    for file in listdir(input_dir):
        if file.endswith('.svg'):
            file_path = join(input_dir, file)
            points = parse_svg_file(file_path, size)
            all_shapes.append(points)
            if points.shape[0] > max_len:
                max_len = points.shape[0]
    # Pad all to max_len
    padded_shapes = []
    for points in all_shapes:
        n = points.shape[0]
        if n < max_len:
            pad = torch.full((max_len - n, 7), -1.0)
            padded = torch.cat([points, pad], dim=0)
        else:
            padded = points
        padded_shapes.append(padded)
    # Stack into [num_images, N, 7]
    final_tensor = torch.stack(padded_shapes, dim=0)
    torch.save(final_tensor, output_file)

if __name__ == "__main__":
    # Example usage
    input_directory = "svg_files"
    output_file = "./shapes_without_rect.pt"
    process_svg_directory(input_directory, output_file) 