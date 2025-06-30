import json
import torch
import numpy as np

NDJSON_FILE = 'airplane.ndjson'
PT_FILE = 'airplane.pt'
NUM_STROKES = 4
NUM_POINTS = 17


def process_stroke(stroke):
    # stroke: [x_list, y_list]
    points = list(zip(stroke[0], stroke[1]))
    # Truncate or pad to NUM_POINTS
    if len(points) >= NUM_POINTS:
        points = points[:NUM_POINTS]
    else:
        points += [(0.0, 0.0)] * (NUM_POINTS - len(points))
    # Flatten to [x1, y1, x2, y2, ..., xN, yN]
    flat = [coord for pt in points for coord in pt]
    return flat

def process_drawing(drawing):
    # drawing: list of strokes
    strokes = []
    for stroke in drawing[:NUM_STROKES]:
        strokes.append(process_stroke(stroke))
    # Pad strokes if fewer than NUM_STROKES
    while len(strokes) < NUM_STROKES:
        strokes.append([0.0] * (NUM_POINTS * 2))
    return strokes

def main():
    with open(NDJSON_FILE, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                drawing = data['drawing']
                # Only include drawings with exactly NUM_STROKES strokes
                if len(drawing) != NUM_STROKES:
                    continue
                # Process just the first valid drawing
                processed = process_drawing(drawing)
                # Convert to numpy array with shape (1, NUM_STROKES, NUM_POINTS*2)
                drawing_np = np.array([processed], dtype=np.float32)
                torch.save(torch.from_numpy(drawing_np), PT_FILE)
                print(f"Saved first drawing to {PT_FILE}")
                return  # Exit after processing the first drawing
            except Exception:
                continue
    print("No valid drawings found")

if __name__ == "__main__":
    main() 