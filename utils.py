from lxml import etree
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "mps"

def format(x):
    tree = etree.parse((x))
    root = tree.getroot()
    d = etree.tostring(root[1])
    d = d.decode(encoding='utf_8')
    data = d.split()
    template = data
    return template

def Rebuild(Vectors, template, size, stroke_thickness):
    svg = []
    for i in Vectors:
        v0 = i[0] * size
        v1 = i[1] * size
        v2 = i[2] * size
        v3 = i[3] * size
        v4 = i[4] * size
        v5 = i[5] * size
        
        template[3] = str(v0) + ','
        template[4] = str(v1)
        template[6] = str(v2) + ','
        template[7] = str(v3) + ','
        template[8] = str(v4) + ','
        template[9] = str(v5)
        template[16] = 'stroke-width="' + str(stroke_thickness) + '"/>' + '\n  '

        #Variable stroke width option
        # template[16] = 'stroke-width="' + str(i[6]) + '"/>' + '\n  '
        svg.append(bytes(' '.join(template), 'utf-8'))
    return svg

def save(s, dim, filename):
    New = etree.XML(
        '<svg width= "{}" height= "{}" version="1.1" xmlns="http://www.w3.org/2000/svg"></svg>'.format(dim, dim))
    for i in s:
        New.append(etree.fromstring(i))
    tree = etree.ElementTree(New)
    tree.write(filename, pretty_print=True)

def filter(stroke):
    values = []
    strokes = stroke.tolist()
    for i in strokes:
        for j in range(len(i)):
           # print(f"i[j]: {i[j]}")
            i[j] = (i[j] + 1) / 2
        if max(i) < 1 and min(i) > 0:
            values.append(i)
    return values

def draw(format_path, size, filename, stroke):
    template = format(format_path)
    print(f"template: {template}")
    stroke = stroke[0,:,:]
    data = filter(stroke)
    svg = Rebuild(data, template, size, size / 128)
    save(svg, size, filename)

def sample(samples, steps, model, noise_scheduler, condition, dim_in):
    stroke = torch.randn(1, samples, dim_in).to(device)
    c = condition[0,:]
    for i, t in enumerate(steps):
        t = torch.full((samples,), t, dtype=torch.long).to(device)
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                residual = model(stroke, t, c)
                
                stroke = noise_scheduler.step(residual, t[0], stroke)[0]

    return stroke

def l_sample(timesteps, model, noise_scheduler, encoded_dim, number_of_strokes):
    model.eval()
    latent = torch.randn(1, number_of_strokes, encoded_dim).to(device)
    for i, t in enumerate(timesteps):
        t = torch.full((1,), t, dtype=torch.long).to(device)
        with torch.no_grad():
            residual = model(latent, t)
            latent = noise_scheduler.step(residual, t[0], latent)[0]
            #latent =torch.unsqueeze(latent, 0)
    return latent

def input_sample(model, set_transformer_encoder, noise_scheduler, condition, dim_per_stroke, number_of_strokes, timesteps):
    inp = torch.randn(1, number_of_strokes, dim_per_stroke).to(device)
    for i, t in enumerate(timesteps):
        t = torch.full((number_of_strokes,), t, dtype=torch.long).to(device)
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
           #inp_enc, condition, mu, sigma = set_transformer_encoder(inp)
                # print(f"inp is a tensor of shape {inp.shape}")
                # print(f"t is a tensor of shape {t.shape}")
                # print(f"condition is a tensor of shape {condition.shape}")
                residual = model(inp, t, condition)
                inp = noise_scheduler.step(residual, t[0], inp)[0]
    return inp

def draw_points_svg(filename, drawing, num_strokes=5, num_points=17):
    """
    Draws a batch of drawings (shape [1, num_strokes, num_points*2]) as SVG using matplotlib, similar to visualize_pt.py.
    Accepts drawing as a torch.Tensor on GPU or CPU.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    if isinstance(drawing, torch.Tensor):
        drawing = drawing.detach().cpu().numpy()
    drawing = drawing[0]  # Remove batch dimension if present
    plt.figure(figsize=(6, 6))
    for stroke in drawing:
        points = np.array(stroke).reshape(num_points, 2)
        for i in range(num_points - 1):
            x0, y0 = points[i]
            x1, y1 = points[i + 1]
            if (x0, y0) != (0, 0) and (x1, y1) != (0, 0):
                plt.plot([x0, x1], [y0, y1], marker='o')
    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.axis('off')
    plt.savefig(filename, format='svg', bbox_inches='tight')
    plt.close()

def tensor_to_svg(tensor, filename=None, size=256):
    """
    Convert a tensor of shape [N, 7] (or [num_images, N, 7]) back to an SVG string or file.
    If filename is provided, saves the SVG to that file.
    """
    import numpy as np
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().numpy()
    if tensor.ndim == 3:  # batch mode
        results = []
        for i, t in enumerate(tensor):
            fname = None
            if filename is not None:
                fname = filename.replace('.svg', f'_{i}.svg')
            results.append(tensor_to_svg(t, fname, size))
        return results

    svg_elements = []
    for row in tensor:
        if np.all(row == -1):
            continue
        cmd_type = int(row[6])
        vals = row[:6]
        if cmd_type == 0:  # path
            # Interpret as a move-to and cubic bezier if enough points, else as a polyline
            if np.count_nonzero(vals != -1) >= 6:
                d = f'M {vals[0]*size/2+size/2} {vals[1]*size/2+size/2} C {vals[2]*size/2+size/2} {vals[3]*size/2+size/2} {vals[4]*size/2+size/2} {vals[5]*size/2+size/2} {vals[4]*size/2+size/2} {vals[5]*size/2+size/2}'
            else:
                # fallback: just move to the first point
                d = f'M {vals[0]*size/2+size/2} {vals[1]*size/2+size/2}'
            svg_elements.append(f'<path d="{d}" fill="none" stroke="black" stroke-width="2"/>')
        elif cmd_type == 1:  # circle
            cx = vals[0]*size/2+size/2
            cy = vals[1]*size/2+size/2
            r = abs(vals[2]*size/2)
            svg_elements.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="red" stroke-width="2"/>')
        elif cmd_type == 2:  # rect
            x = vals[0]*size/2+size/2
            y = vals[1]*size/2+size/2
            w = abs(vals[2]*size/2)
            h = abs(vals[3]*size/2)
            svg_elements.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" stroke="blue" stroke-width="2"/>')
        # Add more types if needed

    svg_str = f'<svg width="{size}" height="{size}" xmlns="http://www.w3.org/2000/svg">' + ''.join(svg_elements) + '</svg>'
    if filename is not None:
        with open(filename, 'w') as f:
            f.write(svg_str)
    return svg_str