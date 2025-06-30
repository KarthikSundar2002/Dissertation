from options.args import argument_parser
import os
import torch
from utils import draw_points_svg

if __name__ == "__main__":
    stroke = torch.load('drawing_3.pt')
    stroke = stroke.unsqueeze(0)
    draw_points_svg('drawing_3.svg', stroke, num_strokes=4, num_points=15)