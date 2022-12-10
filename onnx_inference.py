#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os

import cv2

import time
import logging

from yolox.demo_utils import YOLOXONNX, YOLOXONNX_VIS


def make_parser():
    parser = argparse.ArgumentParser("onnxruntime inference sample")
    parser.add_argument(
        "-mo",
        "--mode",
        type=str,
        default="video",
        help="Inputfile format",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="model/yolox_tiny.onnx",
        help="Input your onnx model.",
    )
    parser.add_argument(
        "-i",
        "--input_path",
        type=str,
        default='test_image.png',
        help="Path to your input image.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default='output',
        help="Path to your output directory.",
    )
    parser.add_argument(
        "-s",
        "--score_thr",
        type=float,
        default=0.3,
        help="Score threshould to filter the result.",
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default="416,416",
        help="Specify an input shape for inference.",
    )
    parser.add_argument(
        "--with_p6",
        action="store_true",
        help="Whether your model uses p6 in FPN/PAN.",
    )
    return parser


def infer_image(args,yolox_onnx, yolox_vis):
    origin_img = cv2.imread(args.input_path)
    start = time.time()
    predictions, ratio = yolox_onnx.inference(origin_img)
    logging.info(f'Infer time: {(time.time()-start)*1000:.2f} [ms]')
    
    result_img = yolox_vis.visual(origin_img, predictions, ratio)
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, os.path.basename(args.input_path))
    logging.info(f'save_path: {output_path}')
    cv2.imwrite(output_path, result_img)
    
    logging.info(f'Inference Finish!')


def infer_video(args,yolox_onnx, yolox_vis):
    cap = cv2.VideoCapture(args.input_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir,os.path.basename(args.input_path))
    
    vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )
    
    frame_id = 1
    while True:
        ret_val, origin_img = cap.read()
        if not ret_val:
            break
        
        start = time.time()
        predictions, ratio = yolox_onnx.inference(origin_img)
        logging.info(f'Frame: {frame_id}/{frame_count}, Infer time: {(time.time()-start)*1000:.2f} [ms]')
        
        result_img = yolox_vis.visual(origin_img, predictions, ratio)
        
        vid_writer.write(result_img)
        
        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break
        
        frame_id+=1
    
    logging.info(f'save_path: {save_path}')
    logging.info(f'Inference Finish!')


def main():
    args = make_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s:%(name)s - %(message)s")
    
    input_shape = tuple(map(int, args.input_shape.split(',')))
    
    yolox_onnx = YOLOXONNX(
        model_path= args.model,
        input_shape= input_shape,
        class_score_th=args.score_thr,
        with_p6=args.with_p6,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
    )
    yolox_vis = YOLOXONNX_VIS(
        class_score_th=args.score_thr
    )
    
    if args.mode == 'image':
        infer_image(args,yolox_onnx, yolox_vis)
    elif args.mode == 'video':
        infer_video(args,yolox_onnx, yolox_vis)


if __name__ == '__main__':
    main()