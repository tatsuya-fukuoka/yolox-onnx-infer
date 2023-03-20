#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import logging

import cv2
import yaml

from yolox.utils import YOLOXONNX


def infer_image(project_config, yolox_onnx):
    input_path = project_config["input_path"]
    output_dir = project_config["output_dir"]

    origin_img = cv2.imread(input_path)
    h, w, c = origin_img.shape
    logging.info(f'Input info - width: {w}, height: {h}')

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(input_path))
    
    start = time.time()
    result_img = yolox_onnx.inference(origin_img)
    logging.info(f'Infer time: {(time.time()-start)*1000:.2f} [ms]')
    
    cv2.imwrite(output_path, result_img)
    
    logging.info(f'save_path: {output_path}')
    logging.info(f'Inference Finish!')


def infer_video(project_config, yolox_onnx):
    input_path = project_config["input_path"]
    output_dir = project_config["output_dir"]

    cap = cv2.VideoCapture(input_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logging.info(f'Input info - width: {width}, height: {height}, fps: {fps}')

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, os.path.basename(input_path))
    
    writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )
    
    frame_id = 1
    while True:
        ret_val, origin_img = cap.read()
        if not ret_val:
            break
        
        start = time.time()
        result_img = yolox_onnx.inference(origin_img)
        logging.info(f'Frame: {frame_id}/{frame_count}, Infer time: {(time.time()-start)*1000:.2f} [ms]')
        
        writer.write(result_img)
        
        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break
        
        frame_id+=1
        
    writer.release()
    cv2.destroyAllWindows()
    
    logging.info(f'save_path: {save_path}')
    logging.info(f'Inference Finish!')


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s:%(name)s - %(message)s")

    with open('config.yaml', 'r') as yml:
        config = yaml.safe_load(yml)
        logging.info(config)
    
    project_config = config["project_config"]

    yolox_onnx = YOLOXONNX(config["yolox_config"])
    
    if project_config["mode"] == 'image':
        infer_image(project_config, yolox_onnx)
    else:
        infer_video(project_config, yolox_onnx)


if __name__ == '__main__':
    main()