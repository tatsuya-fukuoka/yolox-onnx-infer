# YOLOX onnx inference
Code to infer images and videos using YOLOX's onnx model

YOLOXでonnxモデルを用いて画像と動画を推論するコード

# 1. Building the environment
## 1.1 pip install
```bash
pip install -U pip && pip install -r requirements.txt
```
## 1.2 Docker
### 1.2.1 docker build
```Dockerfile
FROM ubuntu:20.04
USER root

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

LABEL version="1.0"
LABEL description="Build operating environment for yolox-onnx-infer"

RUN apt-get update && \
    apt-get -y install python3-pip && \
    apt-get -y install git && \
    apt-get -y install libgl1-mesa-dev && \
    apt-get -y install libglib2.0-0 && \
    pip install -U pip && \
    pip install onnxruntime==1.13.1 opencv-python==4.6.0.66 pyyaml==6.0
```
```Dockerfile
FROM debian:stable-slim
USER root

LABEL version="1.0"
LABEL description="Build operating environment for yolox-onnx-infer"

RUN apt-get update && \
    apt-get -y install python3-pip && \
    apt-get -y install git && \
    apt-get -y install libgl1-mesa-dev && \
    apt-get -y install libglib2.0-0 && \
    pip install -U pip && \
    pip install onnxruntime==1.13.1 opencv-python==4.6.0.66 pyyaml==6.0
```
```bash
docker build -t tatsuya060504/yolox-onnx-infer:v1.0.0 .
```
### 1.2.2 Docker Hub
https://hub.docker.com/repository/docker/tatsuya060504/yolox-onnx-infer
```bash
docker pull tatsuya060504/yolox-onnx-infer:raspberrypi
#or
docker pull tatsuya060504/yolox-onnx-infer:wsl2
```
### 1.2.3 Docker run
```
docker run -it --name=yolox-onnx-infer -v $(pwd):/home tatsuya060504/yolox-onnx-infer:v1.0.0
```

# 2. Inference model download
You can download the yolox-onnx model by executing the shell script in the model folder.
```
cd model
sh download_yolox_<type>_onnx.sh
```
You can also get the model from the Google Drive link below.
* [yolox_nano.onnx](https://drive.google.com/file/d/17_U9mVqb6-07P2uQ-_A5fd-F2Dp7_-j-/view?usp=share_link)
* [yolox_tiny.onnx](https://drive.google.com/file/d/1uLZMCrYzt-bDunqO6xByPqym9bfk_5q3/view?usp=share_link)
* [yolox_s.onnx](https://drive.google.com/file/d/1kb2wgrNOp15AWYiVI70f1ll4vbvMqRqh/view?usp=share_link)

# 3. Inference
## 3.1 Yaml file
```yaml
project_config:
  mode: video
  input_path: sample.mp4
  output_dir: outputs

yolox_config:
  model_path: model/yolox_tiny.onnx
  class_score_thr: 0.3
  input_shape: 416,416
  with_p6: False
  device: cpu
```

## 3.2 Inference command
```bash
python onnx_inference.py
```
