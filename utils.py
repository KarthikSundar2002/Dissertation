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
        template[3] = str(i[0] * size) + ','
        template[4] = str(i[1] * size)
        template[6] = str(i[2] * size) + ','
        template[7] = str(i[3] * size) + ','
        template[8] = str(i[4] * size) + ','
        template[9] = str(i[5] * size)
        template[16] = 'stroke-width="' + str(stroke_thickness) + '"/>\n  '

        #Variable stroke width option
        # template[16] = 'stroke-width="' + str(i[6]) + '"/>\n  '
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
            i[j] = (i[j] + 1) / 2
        if max(i) < 1 and min(i) > 0:
            values.append(i)
    return values

def draw(format_path, size, filename, stroke):
    template = format(format_path)
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
                print(stroke.shape)
                print(t.shape)
                print(c.shape)
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