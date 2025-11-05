#!/usr/bin/env python3
import argparse
import colorsys
import math
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Color-code each stroked element in an SVG with a distinct color."
    )
    parser.add_argument("input", help="Path to input SVG file")
    parser.add_argument("output", help="Path to output SVG file")
    parser.add_argument(
        "--palette",
        default="rainbow",
        choices=["rainbow"],
        help="Color palette to use",
    )
    parser.add_argument(
        "--saturation",
        type=float,
        default=0.85,
        help="Saturation for generated colors (0-1)",
    )
    parser.add_argument(
        "--value",
        type=float,
        default=0.85,
        help="Value/Brightness for generated colors (0-1)",
    )
    parser.add_argument(
        "--only-if-stroked",
        action="store_true",
        help=(
            "Only recolor elements that explicitly define a stroke (attribute or style). "
            "If not set, the script still only targets elements that appear to have stroke, "
            "but will be conservative and avoid non-graphical elements."
        ),
    )
    parser.add_argument(
        "--overlaps-only",
        action="store_true",
        help=(
            "Recolor only <path> elements that overlap/intersect with any other <path>. "
            "Detection is approximate via sampling."
        ),
    )
    parser.add_argument(
        "--step",
        type=float,
        default=1.0,
        help="Sampling step in SVG units for overlap detection (smaller = denser)",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help="Tolerance for considering two sampled points overlapping",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=2048,
        help="Safety cap for samples per path to avoid runaway work",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="If provided, limit to first N <path> elements when detecting overlaps",
    )
    return parser.parse_args()


def hsv_palette(n: int, s: float, v: float) -> List[Tuple[int, int, int]]:
    if n <= 0:
        return []
    colors = []
    for i in range(n):
        h = (i / n) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, max(0.0, min(1.0, s)), max(0.0, min(1.0, v)))
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors


STYLE_PAIR_RE = re.compile(r"\s*([^:\s]+)\s*:\s*([^;]*?)\s*(?:;|$)")


def parse_style(style_value: str) -> dict:
    props = {}
    if not style_value:
        return props
    for m in STYLE_PAIR_RE.finditer(style_value):
        key = m.group(1)
        val = m.group(2)
        props[key] = val
    return props


def serialize_style(props: dict) -> str:
    if not props:
        return ""
    return ";".join(f"{k}:{v}" for k, v in props.items())


def hex_color(rgb: Tuple[int, int, int]) -> str:
    return "#%02x%02x%02x" % rgb


def is_graphical_with_stroke(elem: ET.Element) -> bool:
    tag = elem.tag.split('}')[-1]
    graphical_tags = {
        "path",
        "line",
        "polyline",
        "polygon",
        "circle",
        "ellipse",
        "rect",
        "text",
        "use",
    }
    if tag not in graphical_tags:
        return False

    stroke_attr = elem.attrib.get("stroke")
    style_attr = elem.attrib.get("style")
    if stroke_attr is not None and stroke_attr.strip().lower() != "none":
        return True
    if style_attr:
        style = parse_style(style_attr)
        stroke_val = style.get("stroke")
        if stroke_val is not None and stroke_val.strip().lower() != "none":
            return True
    return False


def collect_stroked_elements(root: ET.Element, only_if_stroked: bool) -> List[ET.Element]:
    target_elements: List[ET.Element] = []
    for elem in root.iter():
        if only_if_stroked:
            if is_graphical_with_stroke(elem):
                target_elements.append(elem)
        else:
            # Be conservative: prefer elements that either specify stroke or are likely to be stroked
            if is_graphical_with_stroke(elem):
                target_elements.append(elem)
    return target_elements


def set_element_stroke(elem: ET.Element, color_hex: str) -> None:
    style_attr = elem.attrib.get("style")
    if style_attr:
        style = parse_style(style_attr)
        style["stroke"] = color_hex
        elem.set("style", serialize_style(style))
    # Set explicit stroke attribute as well to override inheritance
    elem.set("stroke", color_hex)


# ---- Overlap detection utilities (approximate via sampling) ----

COMMAND_RE = re.compile(r"([MLQZ])|(-?\d*\.?\d+(?:[eE][+-]?\d+)?)|[,\s]+")


Number = float
Point = Tuple[Number, Number]


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

    def append_segment(cmd: str, pts: List[Point]):
        cmds.append((cmd, pts))

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


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def quad_bezier(p0: Point, p1: Point, p2: Point, t: float) -> Point:
    x = (1 - t) * (1 - t) * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0]
    y = (1 - t) * (1 - t) * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1]
    return (x, y)


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


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


def quantize_point(p: Point, cell: float) -> Tuple[int, int]:
    return (int(round(p[0] / cell)), int(round(p[1] / cell)))


def detect_overlapping_path_indices(root: ET.Element, step: float, tol: float, max_steps: int, limit: Optional[int]) -> Set[int]:
    ns = ''
    if root.tag.startswith('{'):
        ns = root.tag.split('}', 1)[0] + '}'

    paths = root.findall(f'.//{ns}path')
    if limit is not None:
        paths = paths[:limit]

    polylines: List[List[Point]] = []
    for p in paths:
        d = p.attrib.get('d')
        if not d:
            polylines.append([])
            continue
        try:
            cmds = parse_path_commands(d)
        except Exception:
            polylines.append([])
            continue
        pts = sample_polyline_from_commands(cmds, step)
        if len(pts) > max_steps:
            pts = pts[:max_steps]
        polylines.append(pts)

    cell = max(step, tol * 10)
    grid: Dict[Tuple[int, int], List[Tuple[int, int, Point]]] = defaultdict(list)

    for path_idx, pts in enumerate(polylines):
        for point_idx, pt in enumerate(pts):
            q = quantize_point(pt, cell)
            grid[q].append((path_idx, point_idx, pt))

    overlapping_paths: Set[int] = set()
    for points_in_cell in grid.values():
        if len(points_in_cell) < 2:
            continue
        for i in range(len(points_in_cell)):
            p_i, idx_i, pt_i = points_in_cell[i]
            for j in range(i + 1, len(points_in_cell)):
                p_j, idx_j, pt_j = points_in_cell[j]
                if p_i == p_j:
                    continue
                if distance(pt_i, pt_j) <= tol:
                    overlapping_paths.add(p_i)
                    overlapping_paths.add(p_j)
    return overlapping_paths


def main() -> int:
    args = parse_args()
    try:
        # Parse with namespace awareness
        ET.register_namespace("", "http://www.w3.org/2000/svg")
        tree = ET.parse(args.input)
        root = tree.getroot()
    except Exception as e:
        print(f"Failed to parse SVG: {e}", file=sys.stderr)
        return 1

    if args.overlaps_only:
        # Find only <path> elements that overlap with any other
        try:
            overlapping_indices = detect_overlapping_path_indices(
                root=root,
                step=args.step,
                tol=args.tol,
                max_steps=args.max_steps,
                limit=args.limit,
            )
        except Exception as e:
            print(f"Failed during overlap detection: {e}", file=sys.stderr)
            return 1

        ns = ''
        if root.tag.startswith('{'):
            ns = root.tag.split('}', 1)[0] + '}'
        all_paths = root.findall(f'.//{ns}path')
        if args.limit is not None:
            all_paths = all_paths[:args.limit]
        elements = [el for idx, el in enumerate(all_paths) if idx in overlapping_indices]
    else:
        elements = collect_stroked_elements(root, args.only_if_stroked)
    count = len(elements)
    if count == 0:
        try:
            tree.write(args.output, encoding="utf-8", xml_declaration=True)
            print("No target elements found. Wrote unmodified SVG.")
            return 0
        except Exception as e:
            print(f"Failed to write SVG: {e}", file=sys.stderr)
            return 1

    colors = hsv_palette(count, args.saturation, args.value)
    for idx, elem in enumerate(elements):
        set_element_stroke(elem, hex_color(colors[idx]))

    try:
        tree.write(args.output, encoding="utf-8", xml_declaration=True)
    except Exception as e:
        print(f"Failed to write SVG: {e}", file=sys.stderr)
        return 1

    print(f"Recolored {count} stroked element(s). Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


