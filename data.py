import json
import os
import shutil

import torch
import torchvision.transforms as transforms

from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from os import path

import numpy as np
import cv2
import kagglehub

from PIL import Image
from networks.LineArtGenerator.dataset import UnpairedDepthDataset
from networks.LineArtGenerator.model import Generator


def download_data(out_dir="./data/"):
    with open("./Secrets/kaggle.json", "r") as file:
        data = json.load(file)
    kagglehub.auth.set_kaggle_credentials(data["username"], data["key"])

    tempPath = kagglehub.dataset_download(
        "subinium/highresolution-anime-face-dataset-512x512"
    )
    filenames = os.listdir(tempPath)
    for file_name in filenames:
        shutil.move(os.path.join(tempPath, file_name), out_dir)
    print("Downloaded Data")


def generate_line_art(input_dir, out_dir):
    net = Generator(3,1,3)
    net.load_state_dict(torch.load("./weights/line_art/anime_style/netG_A_latest.pth", map_location='cpu'))
    net.eval()

    transforms_r = [transforms.Resize((512, 512),Image.BICUBIC), transforms.ToTensor()]
    transforms_r = transforms.Compose(transforms_r)

    test_data = UnpairedDepthDataset(root=input_dir, root2='', transforms_r=transforms_r) 
    dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

    full_output_dir = os.path.join(out_dir, "line_art")

    if not os.path.exists(full_output_dir):
        os.makedirs(full_output_dir)
    
    for i, batch in enumerate(dataloader):
        img_r  = Variable(batch['r'])
        img_depth  = Variable(batch['depth'])
        real_A = img_r
        name = batch['name'][0]
        input_image = real_A
        image = net(input_image)
        save_image(image.data, full_output_dir+'/%s_out.png' % name)

def generate_svg(line_art_dir, out_dir, device="cpu"):

    model_udf, args_udf, model_ndc, args_ndc = load_model(device)
    for img_name in os.listdir(line_art_dir):
        if ("png" in img_name or "jpg" in img_name or "bmp" in img_name):
            name, _ = os.path.splitext(img_name)
            if os.path.exists(
                os.path.join(out_dir, name + ".svg")
            ):
                continue
            if 'res' in img_name or 'keypt' in img_name or 'usm' in img_name:
                continue
            
            if device == 'cuda':
                    # if True:
                ww, hh = Image.open(path.join(line_art_dir, img_name)).size
                longer = ww if ww > hh else hh
                resize = True if longer > 512 else False
                if resize:
                    ratio = 512 / longer
                    print("log:\timage size (%dx%d) is too large, resize to (%dx%d)" % (
                        hh, ww, int(hh * ratio), int(ww * ratio)))
            else:
                resize = False
                ratio = 1
            
            img, img_np, canvas_size, _ = load_img(path.join(line_art_dir, img_name), device, args_udf.up_scale,
                                                       thin=None,
                                                       line_extractor=None,
                                                       resize=resize,
                                                       path_to_out=out_dir,
                                                       resize_to=512,
                                                       ui_mode=False)

            tensor_h, tensor_w = img.shape[2], img.shape[3]

            # predict UDF from sketch
            udf_topo_pre, usm_pre_, keypt_pre_dict, keypt_pre_np = predict_UDF(
                img, img_np, model_udf, None, name)
            
            linemap_pre_x, linemap_pre_y, pt_map_pre, edge_maps_pre_xy = predict_SVG(
                    udf_topo_pre, model_ndc, args_udf, args_ndc, (tensor_h, tensor_w), out_path=out_dir, refine=False, name=name, to_npz=False)

            lines_pre, _, usm_applied, usm_uncertain = refine_topology(
                    D(edge_maps_pre_xy),
                    pt_map_pre,
                    usm_pre_,
                    linemap_pre_x,
                    linemap_pre_y,
                    keypt_pre_dict,
                    down_rate=8,
                    downsample=True,
                    full_auto_mode=True)
            usm_applied = usm_applied != 0
            lines_to_svg(
                    lines_pre * 2,
                    tensor_w * 2,
                    tensor_h * 2,
                    path.join(
                        out_dir,
                        name + "_refine.svg"))
            simplify_SVG(
                    path.join(
                        out_dir,
                        name + "_refine.svg"),
                    keypt_pre_dict,
                    bezier=False,
                    rdp_simplify=True,
                    epsilon=0.4,
                    skip_len=4)
            svg_pre = path.join(
                    out_dir,
                    name + "_raw.svg")
            res_pre = svg_to_numpy(svg_pre)
            if res_pre is None:
                    res_pre = np.ones((tensor_h, tensor_w, 3)) * 255
            else:
                    res_pre = res_pre[..., np.newaxis].repeat(3, axis=-1)
            h, w = res_pre.shape[0], res_pre.shape[1]
            img_np = cv2.resize(
                img_np, (w, h), interpolation=cv2.INTER_AREA)
            keypt_pre_np = cv2.resize(
                keypt_pre_np, (w, h), interpolation=cv2.INTER_AREA)

            res_pre_keypt = blend_skeletons(res_pre, (usm_applied.astype(
                int), (usm_uncertain != 0).astype(int)), usm_mode=True)
            res_pre_keypt = add_keypt(
                res_pre_keypt, keypt_pre_dict, (tensor_h, tensor_w))
            keypt_pre_list = []
            keypt_to_color = {
                "end_point": "green",
                "sharp_turn": "red",
                "junc": "blue"}
            color_list = []
            for key in keypt_pre_dict:
                for i in range(len(keypt_pre_dict[key])):
                    keypt_pre_list.append(
                        complex(*keypt_pre_dict[key][i]))
                    color_list.append(keypt_to_color[key])
            pt_num = len(keypt_pre_list)
            assert len(keypt_pre_list) == len(color_list)
            Image.fromarray(
                res_pre_keypt.astype(
                    np.uint8)).save(
                path.join(
                    out_dir,
                    name + "_usm.png"))