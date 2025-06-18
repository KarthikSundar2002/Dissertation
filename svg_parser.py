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

def parse_svg_file(file_path, size=256):
    """Parse an SVG file and return normalized points"""
    tree = etree.parse(file_path)
    root = tree.getroot()
    all_points = []
    
    for element in root:
        if element.tag.endswith('path'):
            d = element.get('d')
            points = parse_path_data(d)
            normalized = normalize_points(points, size)
            all_points.extend(normalized)
            
        elif element.tag.endswith('circle'):
            cx = float(element.get('cx', 0))
            cy = float(element.get('cy', 0))
            r = float(element.get('r', 0))
            points = [cx,cy,r]
            normalized = normalize_points(points, size)
            normalized = [1.0, *normalized]
            all_points.extend(normalized)
            
        elif element.tag.endswith('rect'):
            x = float(element.get('x', 0))
            y = float(element.get('y', 0))
            width = float(element.get('width', 0))
            height = float(element.get('height', 0))
            points = parse_rect(x, y, width, height)
            normalized = normalize_points(points, size)
            all_points.extend(normalized)
    
    return torch.FloatTensor(all_points)

def process_svg_directory(input_dir, output_file, size=256):
    """Process all SVG files in a directory and save as PyTorch tensor"""
    all_shapes = []
    
    for file in listdir(input_dir):
        if file.endswith('.svg'):
            file_path = join(input_dir, file)
            points = parse_svg_file(file_path, size)
            all_shapes.append(points)
    
    torch.save(all_shapes, output_file)

if __name__ == "__main__":
    # Example usage
    input_directory = "svg_files"
    output_file = "Data/shapes.pt"
    process_svg_directory(input_directory, output_file) 