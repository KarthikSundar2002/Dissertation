import json
import torch
import numpy as np

NDJSON_FILE = 'full_simplified_spider.ndjson'
PT_FILE = 'spider.pt'


def find_max_strokes_points(ndjson_file):
    max_strokes = 0
    max_points = 0
    with open(ndjson_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                drawing = data['drawing']
                num_strokes = len(drawing)
                max_strokes = max(max_strokes, num_strokes)
                for stroke in drawing:
                    num_points = min(len(stroke[0]), len(stroke[1]))
                    max_points = max(max_points, num_points)
            except Exception:
                continue
    return max_strokes, max_points

def process_stroke(stroke, max_points):
    # stroke: [x_list, y_list]
    points = list(zip(stroke[0], stroke[1]))
    mask = [1.0] * min(len(points), max_points)
    # Truncate or pad to max_points
    if len(points) >= max_points:
        points = points[:max_points]
    else:
        points += [(0.0, 0.0)] * (max_points - len(points))
        mask += [0.0] * (max_points - len(mask))
    # Flatten to [x1, y1, x2, y2, ..., xN, yN]
    flat = [coord for pt in points for coord in pt]
    # Mask for x and y
    mask_flat = []
    for m in mask:
        mask_flat.extend([m, m])
    return flat, mask_flat

def process_drawing(drawing, max_strokes, max_points):
    strokes = []
    mask_strokes = []
    for stroke in drawing:
        flat, mask_flat = process_stroke(stroke, max_points)
        strokes.append(flat)
        mask_strokes.append(mask_flat)
    # Pad strokes if fewer than max_strokes
    while len(strokes) < max_strokes:
        strokes.append([0.0] * (max_points * 2))
        mask_strokes.append([0.0] * (max_points * 2))
    # Truncate if more than max_strokes
    strokes = strokes[:max_strokes]
    mask_strokes = mask_strokes[:max_strokes]
    return strokes, mask_strokes

def main():
    # First pass: find max_strokes and max_points
    max_strokes, max_points = find_max_strokes_points(NDJSON_FILE)
    print(f"Max strokes: {max_strokes}, Max points: {max_points}")
    
    drawings = []
    masks = []
    with open(NDJSON_FILE, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                drawing = data['drawing']
                strokes, mask_strokes = process_drawing(drawing, max_strokes, max_points)
                drawings.append(strokes)
                masks.append(mask_strokes)
            except Exception:
                continue
    if not drawings:
        print("No valid drawings found")
        return
    # Convert to numpy arrays
    drawings_np = np.array(drawings, dtype=np.float32)  # (N, max_strokes, max_points*2)
    masks_np = np.array(masks, dtype=np.float32)        # (N, max_strokes, max_points*2)
    # Save as tuple
    torch.save((torch.from_numpy(drawings_np), torch.from_numpy(masks_np)), PT_FILE)
    print(f"Saved {len(drawings)} drawings to {PT_FILE} with shape {drawings_np.shape} and mask shape {masks_np.shape}")

if __name__ == "__main__":
    main() 