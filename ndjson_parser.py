import json

def load_strokes_from_ndjson(filepath):
    """
    Loads drawings from a .ndjson file and yields the strokes for each drawing.
    """
    with open(filepath, 'r') as f:
        for line in f:
            try:
                drawing_data = json.loads(line)
                yield drawing_data['drawing']
            except json.JSONDecodeError:
                # Handle potential empty or malformed lines
                continue

# Path to your downloaded .ndjson file
ndjson_file = 'airplane.ndjson'

# Process the strokes for each drawing
for i, strokes in enumerate(load_strokes_from_ndjson(ndjson_file)):
    print(f"Drawing {i+1}:")
    print(f"  Number of strokes: {len(strokes)}")

    # Reshape the strokes into a more intuitive list of (x, y) points
    reshaped_strokes = []
    for stroke in strokes:
        # The zip function pairs the x and y coordinates together
        reshaped_stroke = list(zip(stroke[0], stroke[1]))
        reshaped_strokes.append(reshaped_stroke)
        print(f"    Stroke {len(reshaped_strokes)}: {len(reshaped_stroke)} points")

    # Now 'reshaped_strokes' is a list of strokes,
    # where each stroke is a list of (x, y) tuples.
    # You can now use this data for your specific application.
    # For example, you could save it to a different format,
    # or use it to train a model.

    if i == 4: # Stop after the first 5 drawings for demonstration
        break