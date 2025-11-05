#!/usr/bin/env python3

import argparse
import math
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import List, Tuple, Iterable, Optional, Dict, Set
import os
import glob


Number = float
Point = Tuple[Number, Number]
BBox = Tuple[float, float, float, float]
Segment = Tuple[Point, Point]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze stroke overlaps/intersections in an SVG or a folder of SVGs")
    parser.add_argument("svg_path", help="Path to an SVG file or a directory of SVG files")
    parser.add_argument("--mode", choices=["intersections", "raster"], default="intersections",
                        help="Analysis mode: intersections (default) or raster")
    parser.add_argument("--step", type=float, default=1.0,
                        help="Sampling step in SVG units along curves (smaller = denser)")
    parser.add_argument("--max-steps", type=int, default=2048,
                        help="Safety cap for samples per path to avoid runaway work")
    parser.add_argument("--dpi-scale", type=float, default=1.0,
                        help="Scale factor (raster mode only)")
    parser.add_argument("--limit", type=int, default=None,
                        help="If provided, limit to first N <path> elements per file for speed")
    parser.add_argument("--tol", type=float, default=1e-6,
                        help="Tolerance for intersection deduplication and robust checks")
    parser.add_argument("--grid", type=float, default=None,
                        help="Grid cell size for spatial binning (defaults to step)")
    parser.add_argument("--recursive", action="store_true",
                        help="If svg_path is a directory, search recursively for *.svg files")
    parser.add_argument("--verbose", action="store_true",
                        help="Print progress details")
    parser.add_argument("--debug", action="store_true",
                        help="Print detailed debugging information")
    return parser.parse_args()


# --- SVG path parsing (supports only absolute M, L, Q) ---

COMMAND_RE = re.compile(r"([MLQZ])|(-?\d*\.?\d+(?:[eE][+-]?\d+)?)|[,\s]+")


def tokenize_path_d(d: str) -> List[str]:
    tokens: List[str] = []
    i = 0
    while i < len(d):
        m = COMMAND_RE.match(d, i)
        if not m:
            raise ValueError(f"Unrecognized path data at position {i}: ...{d[i:i+20]}")
        if m.group(1):
            tokens.append(m.group(1))
        elif m.group(2):
            tokens.append(m.group(2))
        i = m.end()
    return tokens


def consume_number(tokens: List[str], idx: int) -> Tuple[float, int]:
    if idx >= len(tokens):
        raise ValueError("Unexpected end of tokens while expecting number")
    try:
        value = float(tokens[idx])
    except ValueError as exc:
        raise ValueError(f"Expected number, got {tokens[idx]}") from exc
    return value, idx + 1


def parse_path_commands(d: str) -> List[Tuple[str, List[Point]]]:
    tokens = tokenize_path_d(d)
    i = 0
    cmds: List[Tuple[str, List[Point]]] = []
    current_cmd: Optional[str] = None
    last_point: Optional[Point] = None

    def append_segment(cmd: str, pts: List[Point]):
        nonlocal cmds, last_point
        cmds.append((cmd, pts))
        if cmd == 'M':
            last_point = pts[0]
        elif cmd == 'L':
            last_point = pts[0]
        elif cmd == 'Q':
            last_point = pts[1]

    while i < len(tokens):
        t = tokens[i]
        if t in ('M', 'L', 'Q', 'Z'):
            current_cmd = t
            i += 1
            if current_cmd == 'Z':
                continue
        if current_cmd is None:
            raise ValueError("Path data must begin with a command letter")

        if current_cmd == 'M':
            x, i = consume_number(tokens, i)
            y, i = consume_number(tokens, i)
            append_segment('M', [(x, y)])
            while i < len(tokens) and tokens[i] not in ('M', 'L', 'Q', 'Z'):
                x, i = consume_number(tokens, i)
                y, i = consume_number(tokens, i)
                append_segment('L', [(x, y)])
        elif current_cmd == 'L':
            while i < len(tokens) and tokens[i] not in ('M', 'L', 'Q', 'Z'):
                x, i = consume_number(tokens, i)
                y, i = consume_number(tokens, i)
                append_segment('L', [(x, y)])
        elif current_cmd == 'Q':
            while True:
                if i >= len(tokens) or tokens[i] in ('M', 'L', 'Q', 'Z'):
                    break
                cx, i = consume_number(tokens, i)
                cy, i = consume_number(tokens, i)
                x, i = consume_number(tokens, i)
                y, i = consume_number(tokens, i)
                append_segment('Q', [(cx, cy), (x, y)])
        else:
            raise ValueError(f"Unsupported command: {current_cmd}")

    return cmds


# --- Geometry helpers ---


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def quad_bezier(p0: Point, p1: Point, p2: Point, t: float) -> Point:
    x = (1 - t) * (1 - t) * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0]
    y = (1 - t) * (1 - t) * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1]
    return (x, y)


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


# --- Polyline sampling ---


def sample_polyline_from_commands(cmds: List[Tuple[str, List[Point]]], step: float) -> List[Point]:
    points: List[Point] = []
    cursor: Optional[Point] = None

    for cmd, pts in cmds:
        if cmd == 'M':
            cursor = pts[0]
            points.append(cursor)
        elif cmd == 'L':
            if cursor is None:
                cursor = pts[0]
                points.append(cursor)
                continue
            p1 = cursor
            p2 = pts[0]
            seg_len = max(distance(p1, p2), 1e-6)
            n = max(1, int(math.ceil(seg_len / max(step, 1e-6))))
            for k in range(1, n + 1):
                t = k / n
                x = lerp(p1[0], p2[0], t)
                y = lerp(p1[1], p2[1], t)
                points.append((x, y))
            cursor = p2
        elif cmd == 'Q':
            if cursor is None:
                cursor = pts[1]
                points.append(cursor)
                continue
            c = pts[0]
            p2 = pts[1]
            approx_len = 0.0
            prev = cursor
            tmp_n = 10
            for k in range(1, tmp_n + 1):
                t = k / tmp_n
                q = quad_bezier(cursor, c, p2, t)
                approx_len += distance(prev, q)
                prev = q
            n = max(1, int(math.ceil(approx_len / max(step, 1e-6))))
            n = min(n, 4096)
            for k in range(1, n + 1):
                t = k / n
                q = quad_bezier(cursor, c, p2, t)
                points.append(q)
            cursor = p2
        else:
            continue
    return points


# --- Intersection utilities ---


def bbox_of_segment(seg: Segment) -> BBox:
    (x1, y1), (x2, y2) = seg
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def expand_bbox(b: BBox, other: BBox) -> BBox:
    return (min(b[0], other[0]), min(b[1], other[1]), max(b[2], other[2]), max(b[3], other[3]))


def on_segment(a: Point, b: Point, p: Point, tol: float) -> bool:
    (x1, y1), (x2, y2) = a, b
    px, py = p
    if (min(x1, x2) - tol <= px <= max(x1, x2) + tol and
        min(y1, y2) - tol <= py <= max(y1, y2) + tol):
        area = abs((x2 - x1) * (py - y1) - (y2 - y1) * (px - x1))
        return area <= tol * (abs(x2 - x1) + abs(y2 - y1) + 1.0)
    return False


def segment_intersection(a1: Point, a2: Point, b1: Point, b2: Point, tol: float) -> Tuple[bool, Optional[Point], bool]:
    x1, y1 = a1
    x2, y2 = a2
    x3, y3 = b1
    x4, y4 = b2

    if max(min(x1, x2), min(x3, x4)) - min(max(x1, x2), max(x3, x4)) > tol:
        return False, None, False
    if max(min(y1, y2), min(y3, y4)) - min(max(y1, y2), max(y3, y4)) > tol:
        return False, None, False

    def cross(ax, ay, bx, by):
        return ax * by - ay * bx

    dx1 = x2 - x1
    dy1 = y2 - y1
    dx2 = x4 - x3
    dy2 = y4 - y3
    dx3 = x1 - x3
    dy3 = y1 - y3

    denom = cross(dx1, dy1, dx2, dy2)
    num_t = cross(-dx3, -dy3, dx2, dy2)
    num_u = cross(-dx3, -dy3, dx1, dy1)

    if abs(denom) <= tol:
        area = cross(dx1, dy1, dx3, dy3)
        if abs(area) <= tol:
            if (max(min(x1, x2), min(x3, x4)) <= min(max(x1, x2), max(x3, x4)) + tol and
                max(min(y1, y2), min(y3, y4)) <= min(max(y1, y2), max(y3, y4)) + tol):
                return True, None, True
        return False, None, False

    t = num_t / denom
    u = num_u / denom

    if -tol <= t <= 1 + tol and -tol <= u <= 1 + tol:
        ix = x1 + t * dx1
        iy = y1 + t * dy1
        return True, (ix, iy), False

    return False, None, False


def quantize_point(p: Point, tol: float) -> Tuple[int, int]:
    return (int(round(p[0] / tol)), int(round(p[1] / tol)))


def build_segments_by_path(polylines: List[List[Point]], tol: float) -> Tuple[List[Tuple[int, Segment, BBox]], BBox]:
    segs: List[Tuple[int, Segment, BBox]] = []
    overall: Optional[BBox] = None
    for pi, pts in enumerate(polylines):
        if len(pts) < 2:
            continue
        for i in range(len(pts) - 1):
            a = pts[i]
            b = pts[i + 1]
            if distance(a, b) < tol:
                continue
            s = (a, b)
            bb = bbox_of_segment(s)
            segs.append((pi, s, bb))
            overall = bb if overall is None else expand_bbox(overall, bb)
    if overall is None:
        overall = (0.0, 0.0, 0.0, 0.0)
    return segs, overall


def grid_cells_for_bbox(bb: BBox, origin: Point, cell: float) -> Iterable[Tuple[int, int]]:
    ox, oy = origin
    min_cx = int(math.floor((bb[0] - ox) / cell))
    max_cx = int(math.floor((bb[2] - ox) / cell))
    min_cy = int(math.floor((bb[1] - oy) / cell))
    max_cy = int(math.floor((bb[3] - oy) / cell))
    for cy in range(min_cy, max_cy + 1):
        for cx in range(min_cx, max_cx + 1):
            yield (cx, cy)


def count_intersections(svg_path: str, step: float, limit: Optional[int], tol: float,
                        grid_size: Optional[float], verbose: bool, max_steps: int, debug: bool = False) -> Tuple[int, int, int]:
    tree = ET.parse(svg_path)
    root = tree.getroot()
    ns = ''
    if root.tag.startswith('{'):
        ns = root.tag.split('}', 1)[0] + '}'

    paths = root.findall(f'.//{ns}path')
    if limit is not None:
        paths = paths[:limit]

    polylines: List[List[Point]] = []
    processed = 0
    for p in paths:
        d = p.attrib.get('d')
        if not d:
            continue
        try:
            cmds = parse_path_commands(d)
        except Exception as exc:
            if verbose:
                print(f"Skipping path (parse error): {exc}", file=sys.stderr)
            continue
        pts = sample_polyline_from_commands(cmds, step)
        if len(pts) > max_steps:
            pts = pts[:max_steps]
        polylines.append(pts)
        processed += 1
        if verbose and processed % 50 == 0:
            print(f"Sampled {processed} paths...")

    total_points = sum(len(pts) for pts in polylines)

    if len(polylines) < 2:
        return len(polylines), total_points, 0

    # Find overlapping sample points between different paths
    # Use spatial indexing to efficiently find nearby points
    cell = grid_size if (grid_size and grid_size > 0) else max(step, tol * 10)
    
    # Build spatial grid: cell -> list of (path_idx, point_idx, point)
    grid: Dict[Tuple[int, int], List[Tuple[int, int, Point]]] = defaultdict(list)
    
    for path_idx, pts in enumerate(polylines):
        for point_idx, pt in enumerate(pts):
            q = quantize_point(pt, cell)
            grid[q].append((path_idx, point_idx, pt))

    # Find overlapping points (same cell, different paths)
    overlapping_locations: Set[Tuple[int, int]] = set()
    
    for cell_idx, points_in_cell in grid.items():
        if len(points_in_cell) < 2:
            continue
            
        # Group by path to ensure we only count overlaps between different paths
        path_groups: Dict[int, List[Tuple[int, Point]]] = defaultdict(list)
        for path_idx, point_idx, pt in points_in_cell:
            path_groups[path_idx].append((point_idx, pt))
        
        # Only consider cells with points from multiple paths
        if len(path_groups) < 2:
            continue
            
        # Check for actual overlaps within tolerance
        all_points_in_cell = [(path_idx, point_idx, pt) for path_idx, point_idx, pt in points_in_cell]
        
        for i in range(len(all_points_in_cell)):
            path_i, point_i, pt_i = all_points_in_cell[i]
            for j in range(i + 1, len(all_points_in_cell)):
                path_j, point_j, pt_j = all_points_in_cell[j]
                
                # Only count overlaps between different paths
                if path_i == path_j:
                    continue
                    
                # Check if points are within tolerance
                if distance(pt_i, pt_j) <= tol:
                    # Use the quantized location as the overlap point
                    overlapping_locations.add(cell_idx)

    if debug:
        print(f"DEBUG: {len(polylines)} paths, {total_points} sample points")
        print(f"DEBUG: {len(overlapping_locations)} overlapping sample point locations")
        if len(polylines) > 0:
            avg_points_per_path = total_points / len(polylines)
            print(f"DEBUG: Average {avg_points_per_path:.1f} points per path")
        if len(overlapping_locations) > 0 and total_points > 0:
            print(f"DEBUG: Ratio = {len(overlapping_locations)}/{total_points} = {len(overlapping_locations)/total_points:.3f}")

    return len(polylines), total_points, len(overlapping_locations)


# --- Raster mode (legacy) ---


def rasterize_points(points: Iterable[Point], stroke_width: float, scale: float,
                      counts: defaultdict) -> None:
    r = max(0.0, (stroke_width * scale) / 2.0)
    if r <= 0.25:
        for (x, y) in points:
            px = int(round(x * scale))
            py = int(round(y * scale))
            counts[(px, py)] += 1
        return

    r_int = int(math.ceil(r))
    r2 = r * r
    for (x, y) in points:
        cx = x * scale
        cy = y * scale
        min_x = int(math.floor(cx - r_int))
        max_x = int(math.ceil(cx + r_int))
        min_y = int(math.floor(cy - r_int))
        max_y = int(math.ceil(cy + r_int))
        for py in range(min_y, max_y + 1):
            dy2 = (py + 0.5 - cy) ** 2
            if dy2 > r2:
                continue
            dx = math.sqrt(max(0.0, r2 - dy2))
            span_min_x = int(math.floor((cx - dx) - 0.5))
            span_max_x = int(math.floor((cx + dx) - 0.5))
            for px in range(span_min_x, span_max_x + 1):
                counts[(px, py)] += 1


def measure_overlap(svg_path: str, step: float, max_steps: int, dpi_scale: float,
                    limit: Optional[int], verbose: bool) -> Tuple[int, int, float]:
    tree = ET.parse(svg_path)
    root = tree.getroot()

    ns = ''
    if root.tag.startswith('{'):
        ns = root.tag.split('}', 1)[0] + '}'

    counts: defaultdict = defaultdict(int)

    paths = root.findall(f'.//{ns}path')
    if limit is not None:
        paths = paths[:limit]

    processed = 0
    for p in paths:
        d = p.attrib.get('d')
        if not d:
            continue
        stroke_width = float(p.attrib.get('stroke-width', '1'))
        try:
            cmds = parse_path_commands(d)
        except Exception as exc:
            if verbose:
                print(f"Skipping path (parse error): {exc}", file=sys.stderr)
            continue
        pts = sample_polyline_from_commands(cmds, step)
        if len(pts) > max_steps:
            pts = pts[:max_steps]
        rasterize_points(pts, stroke_width, dpi_scale, counts)
        processed += 1
        if verbose and processed % 50 == 0:
            print(f"Processed {processed} paths...")

    total_pixels = len(counts)
    overlap_pixels = sum(1 for v in counts.values() if v >= 2)
    ratio = (overlap_pixels / total_pixels) if total_pixels > 0 else 0.0
    return total_pixels, overlap_pixels, ratio


def list_svg_files(path: str, recursive: bool) -> List[str]:
    if os.path.isdir(path):
        pattern = os.path.join(path, "**", "*.svg") if recursive else os.path.join(path, "*.svg")
        return sorted(glob.glob(pattern, recursive=recursive))
    return [path]


def main() -> None:
    args = parse_args()
    if args.mode == "intersections":
        files = list_svg_files(args.svg_path, args.recursive)
        if len(files) > 1:
            ratios: List[float] = []
            counted = 0
            for fp in files:
                try:
                    num_paths, total_points, num_pts = count_intersections(
                        svg_path=fp,
                        step=args.step,
                        limit=args.limit,
                        tol=args.tol,
                        grid_size=args.grid,
                        verbose=False,
                        max_steps=args.max_steps,
                        debug=args.debug,
                    )
                except Exception as exc:
                    if args.verbose:
                        print(f"Failed {fp}: {exc}", file=sys.stderr)
                    continue
                ratio = (num_pts / total_points) if total_points > 0 else 0.0
                ratios.append(ratio)
                counted += 1
                if args.verbose:
                    print(f"{fp}: ratio={ratio:.6f} (paths={num_paths}, points={total_points}, intersections={num_pts})")
            avg = (sum(ratios) / counted) if counted > 0 else 0.0
            print(f"Files processed: {counted}/{len(files)}")
            print(f"Average Intersection Ratio: {avg:.6f}")
        else:
            num_paths, total_points, num_pts = count_intersections(
                svg_path=files[0],
                step=args.step,
                limit=args.limit,
                tol=args.tol,
                grid_size=args.grid,
                verbose=args.verbose,
                max_steps=args.max_steps,
                debug=args.debug,
            )
            ratio = (num_pts / total_points) if total_points > 0 else 0.0
            print(f"Paths considered: {num_paths}")
            print(f"Total sampled points: {total_points}")
            print(f"Unique intersection points: {num_pts}")
            print(f"Intersection ratio (unique/total): {ratio:.6f}")
    else:
        total, overlap, ratio = measure_overlap(
            svg_path=args.svg_path,
            step=args.step,
            max_steps=args.max_steps,
            dpi_scale=args.dpi_scale,
            limit=args.limit,
            verbose=args.verbose,
        )
        print(f"Total stroke pixels: {total}")
        print(f"Overlapped pixels (>=2): {overlap}")
        print(f"Overlap ratio: {ratio:.6f}")


if __name__ == "__main__":
    main()
