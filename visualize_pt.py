import torch
import matplotlib.pyplot as plt
import numpy as np

PT_FILE = 'airplane.pt'
NUM_STROKES = 5
NUM_POINTS = 17


def plot_drawing(drawing, drawing_idx):
    plt.figure(figsize=(6, 6))
    for stroke in drawing:
        points = np.array(stroke).reshape(NUM_POINTS, 2)
        # Only plot segments where neither endpoint is (0,0)
        for i in range(NUM_POINTS - 1):
            x0, y0 = points[i]
            x1, y1 = points[i + 1]
            if (x0, y0) != (0, 0) and (x1, y1) != (0, 0):
                plt.plot([x0, x1], [y0, y1], marker='o')
    plt.gca().invert_yaxis()
    plt.title(f"Drawing {drawing_idx+1}")
    plt.axis('equal')
    plt.show()

def main():
    drawings = torch.load(PT_FILE)
    drawings = drawings.numpy()  # (N, 5, 34)
    for i, drawing in enumerate(drawings):
        plot_drawing(drawing, i)
        if i == 4:  # Show only the first 5 drawings
            break

if __name__ == "__main__":
    main() 