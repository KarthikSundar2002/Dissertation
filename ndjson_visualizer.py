import json
import matplotlib.pyplot as plt


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
                continue

def plot_drawing(strokes, drawing_idx):
    plt.figure(figsize=(6, 6))
    for stroke in strokes:
        x, y = stroke[0], stroke[1]
        plt.plot(x, y, marker='o')
    plt.gca().invert_yaxis()
    plt.title(f"Drawing {drawing_idx+1}")
    plt.axis('equal')
    plt.show()


def main():
    ndjson_file = 'airplane.ndjson'
    for i, strokes in enumerate(load_strokes_from_ndjson(ndjson_file)):
        plot_drawing(strokes, i)
        if i == 4:  # Show only the first 5 drawings
            break

if __name__ == "__main__":
    main() 