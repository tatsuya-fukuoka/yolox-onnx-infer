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
docker build -t tatsuya060504/yolox-onnx-infer:v1.0.0 .
```
### 1.3 Docker Hub
https://hub.docker.com/repository/docker/tatsuya060504/yolox-onnx-infer
```bash
docker pull tatsuya060504/yolox-onnx-infer:raspberrypi
#or
docker pull tatsuya060504/yolox-onnx-infer:wsl2
```
### 1.4 Docker run
```
docker run -it --name=yolox-onnx-infer -v $(pwd):/home tatsuya060504/yolox-onnx-infer:v1.0.0
```

## 2. Inference model download
```
cd model
sh download_yolox_<type>_onnx.sh
```
* [yolox_nano.onnx](https://drive.google.com/file/d/17_U9mVqb6-07P2uQ-_A5fd-F2Dp7_-j-/view?usp=share_link)
* [yolox_tiny.onnx](https://drive.google.com/file/d/1uLZMCrYzt-bDunqO6xByPqym9bfk_5q3/view?usp=share_link)
* [yolox_s.onnx](https://drive.google.com/file/d/1kb2wgrNOp15AWYiVI70f1ll4vbvMqRqh/view?usp=share_link)

## 3. execution command
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
