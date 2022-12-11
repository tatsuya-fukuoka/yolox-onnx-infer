# YOLOX_onnx_inference
Code to infer images and videos using YOLOX's onnx model

YOLOXでonnxモデルを用いて画像と動画を推論するコード

## 1. Building the environment
### 1.1 pip install
```bash
pip install -U pip && pip install -r requirements.txt
```
### 1.2 Dockerfile
```bash
docker build -t tatsuya060504/yolox-onnx-infer:raspberrypi .
docker run -it --name=yolox-onnx-infer -v /home/<user>/yolox-onnx-infer:/home tatsuya060504/yolox-onnx-infer:raspberrypi
```
### 1.3 Docker Hub
```bash
docker pull tatsuya060504/yolox-onnx-infer:raspberrypi
```

## 2. execution command
### image
```bash
#templete
python onnx_inference.py -mo <data type> -m <onnx model path> -i <image path> -o <input dir> -s <score threshold> --input_shape <input size>

#example
python onnx_inference.py -mo image -m model/yolox_tiny.onnx -i sample_image.jpg -o outputs -s 0.3 --input_shape 416,416
```
### video
```bash
#templete
python onnx_inference.py -mo <data type> -m <onnx model path> -i <video path> -o outputs -s <score threshold> --input_shape <input size>

#example
python onnx_inference.py -mo video -m model/yolox_tiny.onnx -i sample.mp4 -o outputs -s 0.3 --input_shape 416,416
```
