#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os

import cv2
import numpy as np

import onnxruntime
import time
import logging

from yolox.data_augment import preproc as preprocess
from yolox.demo_utils import mkdir, demo_postprocess, visual


def make_parser():
    parser = argparse.ArgumentParser("onnxruntime inference sample")
    parser.add_argument(
        "-mo",
        "--mode",
        type=str,
        default="image",
        help="Inputfile format",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="yolox.onnx",
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
        default='demo_output',
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
        default="640,640",
        help="Specify an input shape for inference.",
    )
    parser.add_argument(
        "--with_p6",
        action="store_true",
        help="Whether your model uses p6 in FPN/PAN.",
    )
    return parser


def infer_image(args,input_shape):
    origin_img = cv2.imread(args.input_path)
    img, ratio = preprocess(origin_img, input_shape)
    
    start = time.time()
    logging.info(f'onnx model: {os.path.basename(args.model)}')
    session = onnxruntime.InferenceSession(args.model)
    
    ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
    output = session.run(None, ort_inputs)
    predictions = demo_postprocess(output[0], input_shape, p6=args.with_p6)[0]
    logging.info(f'Infer time: {time.time()-start:.4f} [s]')
    
    result_img = visual(origin_img, predictions, ratio, args)
    
    mkdir(args.output_dir)
    output_path = os.path.join(args.output_dir, os.path.basename(args.input_path))
    logging.info(f'save_path: {output_path}')
    cv2.imwrite(output_path, result_img)
    
    logging.info(f'Inference Finish!')


def infer_video(args,input_shape):
    cap = cv2.VideoCapture(args.input_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    output_dir = args.output_dir
    mkdir(output_dir)
    save_path = os.path.join(output_dir,os.path.basename(args.input_path))
    
    vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )
    logging.info(f'onnx model: {os.path.basename(args.model)}')
    
    frame_id = 1
    while True:
        ret_val, origin_img = cap.read()
        if ret_val:
            img, ratio = preprocess(origin_img, input_shape)
            
            start = time.time()
            session = onnxruntime.InferenceSession(args.model)
            
            ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
            output = session.run(None, ort_inputs)
            predictions = demo_postprocess(output[0], input_shape, p6=args.with_p6)[0]
            logging.info(f'Frame: {frame_id}/{frame_count}, Infer time: {time.time()-start:.5f} [s]')
            
            result_img = visual(origin_img, predictions, ratio, args)
            
            vid_writer.write(result_img)
            
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        
        frame_id+=1
    
    logging.info(f'save_path: {save_path}')
    logging.info(f'Inference Finish!')


def main():
    args = make_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s:%(name)s - %(message)s")
    logging.info(f'Inference mode: {args.mode}')
    
    input_shape = tuple(map(int, args.input_shape.split(',')))
    logging.info(f'Input Size: {input_shape}')
    
    if args.mode == 'image':
        infer_image(args,input_shape)
    elif args.mode == 'video':
        infer_video(args,input_shape)


if __name__ == '__main__':
    main()