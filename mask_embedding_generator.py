# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import numpy as np
from PIL import Image
import open_clip
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import *

from detectron2.config import get_cfg

from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from open_vocab_seg import add_ovseg_config

from open_vocab_seg.utils import VisualizationDemo
from einops import rearrange

# constants
WINDOW_NAME = "Open vocabulary segmentation"

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_ovseg_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

def matching_sort_key(match: list):
    return match[2]

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for open vocabulary segmentation")
    parser.add_argument(
        "--config-file",
        default="configs/ovseg_swinB_vitL_demo.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--class-names",
        nargs="+",
        help="A list of user-defined class_names"
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)
    class_names = args.class_names

    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32')

    matching_num = 5

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        tokens_list = []
        bounding_boxes_list = []
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output, pred_mask = demo.run_on_image(img, class_names)

            masks = []
            bounding_boxes = []
            tokens = []
            pic_h, pic_w = pred_mask.shape
            for i in range(len(predictions['sem_seg'])):
                masks.append(np.zeros(pred_mask.shape))
                masks[i][pred_mask == i] = 1
                masks[i] = np.uint8(masks[i])
                contours, hierarchy = cv2.findContours(masks[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                mask_image = np.zeros((pic_h, pic_w, 3))
                mask_image = img.copy()
                mask_image[masks[i] == 0] = 0
                
                circum = []
                tot_circum = 0
                for contour in contours:
                    circum.append(contour.shape[0])
                    tot_circum += contour.shape[0]
                for j, contour in enumerate(contours):
                    if circum[j] / tot_circum > 0.1:
                        bbox = cv2.boundingRect(contour)
                        bounding_boxes.append(bbox)
                        [x, y, w, h] = bbox
                        mask_image_crop = mask_image[y : (y + h), x : (x + w), :]
                        with torch.no_grad():
                            token = model.encode_image(preprocess(Image.fromarray(np.uint8(mask_image_crop))).unsqueeze(0)) 
                            token /= token.norm(dim=-1, keepdim=True)
                        tokens.append(token)
            
            bounding_boxes_list.append(bounding_boxes)
            tokens_list.append(tokens)

        nearest = []
        for idx0, token0 in enumerate(tokens_list[0]):
            min_idx = 0
            min_dis2 = 1000
            for idx1, token1 in enumerate(tokens_list[1]):
                dis2 = 0
                for i in range(len(token0[0])):
                    dis2 = dis2 + (token0[0][i] - token1[0][i]) ** 2
                if dis2 < min_dis2:
                    min_dis2 = dis2
                    min_idx = idx1
            nearest.append((idx0, min_idx, np.sqrt(min_dis2)))
        nearest.sort(key=matching_sort_key)
        # print(nearest)

        fig = plt.figure(figsize=(10, 5))
        image0 = np.clip(read_image(args.input[0], format="RGB"), 0, 255).astype(np.uint8)
        fig0 = fig.add_subplot(211)
        image1 = np.clip(read_image(args.input[1], format="RGB"), 0, 255).astype(np.uint8)
        fig1 = fig.add_subplot(212)

        for i in range(matching_num):
            idx0, idx1, dis = nearest[i]
            [x0, y0, w0, h0] = bounding_boxes_list[0][idx0]
            [x1, y1, w1, h1] = bounding_boxes_list[1][idx1]
            mask_color = (int(255 - 400 * dis), int(0), int(400 * dis))
            mask_color = tuple(mask_color)
            cv2.rectangle(image0, (x0, y0), (x0 + w0, y0 + h0), mask_color, 2)
            cv2.rectangle(image1, (x1, y1), (x1 + w1, y1 + h1), mask_color, 2)
            xy0 = (x0 + w0 / 2, y0 + h0 / 2)
            xy1 = (x1 + w1 / 2, y1 + h1 / 2)
            line_color = (float(1 - dis), 0, float(dis))
            line_color = tuple(line_color)
            fig1.add_artist(ConnectionPatch(xyA=xy0, xyB=xy1, axesA=fig0, axesB=fig1, coordsA="data", coordsB="data", color=line_color))
            
        fig0.imshow(image0)
        fig1.imshow(image1)
        plt.show()
    else:
        raise NotImplementedError
    
# python mask_embedding_generator.py --config-file configs/ovseg_swinB_vitL_mask.yaml --class-names 'Tree' 'Car' 'Window' 'Road' 'House'  --input ./data/230217213829381562.png ./data/230217214233166508.png --output ./match --opts MODEL.WEIGHTS ./checkpoints/ovseg_swinbase_vitL14_ft_mpt.pth
