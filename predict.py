import logging
import os

import hydra
import argparse
import io
import multiprocessing
import os
import time
from distutils.util import strtobool
from typing import Union

import cv2
import numpy as np
import torch
from flask import Flask, request, send_file
from flask_cors import CORS
from src.netModule.InpatingModule import InpatingModule

from lama_cleaner.helper import (download_model, load_img, norm_img,
                                 numpy_to_bytes, pad_img_to_modulo,
                                 resize_max_size)

NUM_THREADS = str(multiprocessing.cpu_count())

os.environ["OMP_NUM_THREADS"] = NUM_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = NUM_THREADS
os.environ["MKL_NUM_THREADS"] = NUM_THREADS
os.environ["VECLIB_MAXIMUM_THREADS"] = NUM_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = NUM_THREADS
if os.environ.get("CACHE_DIR"):
    os.environ["TORCH_HOME"] = os.environ["CACHE_DIR"]

BUILD_DIR = os.environ.get("LAMA_CLEANER_BUILD_DIR", "./lama_cleaner/app/build")

app = Flask(__name__, static_folder=os.path.join(BUILD_DIR, "static"))
app.config["JSON_AS_ASCII"] = False
CORS(app)

model = None
device = None


@app.route("/inpaint", methods=["POST"])
def process():
    input = request.files
    image = load_img(input["image"].read())
    original_shape = image.shape
    interpolation = cv2.INTER_CUBIC

    size_limit: Union[int, str] = request.form.get("sizeLimit", "1080")
    if size_limit == "Original":
        size_limit = max(image.shape)
    else:
        size_limit = int(size_limit)

    print(f"Origin image shape: {original_shape}")
    image = resize_max_size(image, size_limit=size_limit, interpolation=interpolation)
    print(f"Resized image shape: {image.shape}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = norm_img(image)

    mask = load_img(input["mask"].read(), gray=True)
    mask = resize_max_size(mask, size_limit=size_limit, interpolation=interpolation)
    mask = norm_img(mask)

    res_np_img = run(image, mask)

    # resize to original size
    res_np_img = cv2.resize(
        res_np_img,
        dsize=(original_shape[1], original_shape[0]),
        interpolation=interpolation,
    )

    return send_file(
        io.BytesIO(numpy_to_bytes(res_np_img)),
        mimetype="image/jpeg",
        as_attachment=True,
        attachment_filename="result.jpeg",
    )


@app.route("/")
def index():
    return send_file(os.path.join(BUILD_DIR, "index.html"))


def run(image, mask):
    """
    image: [C, H, W]
    mask: [1, H, W]
    return: BGR IMAGE
    """
    origin_height, origin_width = image.shape[1:]
    image = pad_img_to_modulo(image, mod=8)
    mask = pad_img_to_modulo(mask, mod=8)
    # image = image.reshape(image.shape[2], image.shape[0], image.shape[1])

    image = torch.from_numpy(image).unsqueeze(0).to('cuda')
    mask = torch.from_numpy(mask).unsqueeze(0).to('cuda')

    start = time.time()
    batch={
        'image':image,
        'mask':mask
    }
    with torch.no_grad():
        batch = model(batch)
    print(f"process time: {(time.time() - start)*1000}ms")
    cur_res = batch['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy()
    cur_res = cur_res[0:origin_height, 0:origin_width, :]
    cur_res = np.clip(cur_res * 255, 0, 255).astype("uint8")
    return cur_res

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=8080, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()

@hydra.main(config_path="/home/lxl/scc/projects/Inpating/lama_cleaner/model", config_name="config", version_base='1.1')
def main(config):
    global model
    model_path= '/home/lxl/scc/projects/Inpating/lama_cleaner/model/last.ckpt'
    model=InpatingModule(config)
    state = torch.load(model_path, map_location='cuda')
    model.load_state_dict(state['state_dict'], strict=True)
    model.on_load_checkpoint(state)
    model.freeze()
    model.to('cuda')
    args = get_args_parser()
    app.run(host="0.0.0.0", port=args.port, debug=args.debug)

if __name__ == '__main__':
    main()