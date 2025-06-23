import json
import numpy as np
from collections import Counter

NDJSON_FILE = 'airplane.ndjson'
percentile = 90

point_counts = []
stroke_counts = []

with open(NDJSON_FILE, 'r') as f:
    for line in f:
        try:
            drawing_data = json.loads(line)
            drawing = drawing_data['drawing']
            stroke_counts.append(len(drawing))
            for stroke in drawing:
                # stroke: [x_list, y_list]
                num_points = len(stroke[0])  # or len(stroke[1]), both are same
                point_counts.append(num_points)
        except Exception:
            continue

if point_counts:
    perc = np.percentile(point_counts, percentile)
    print(f"{percentile}th percentile of number of points in each stroke: {perc}")
    mode_points = Counter(point_counts).most_common(1)[0][0]
    print(f"Mode number of points per stroke: {mode_points}")
else:
    print("No strokes found.")

if stroke_counts:
    mean_strokes = np.mean(stroke_counts)
    mode_strokes = Counter(stroke_counts).most_common(1)[0][0]
    print(f"Mean number of strokes per drawing: {mean_strokes}")
    print(f"Mode number of strokes per drawing: {mode_strokes}")
else:
    print("No drawings found.") 