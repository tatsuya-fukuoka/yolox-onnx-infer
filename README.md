# YOLOX_onnx_inference
YOLOXでonnxモデルを用いて画像と動画を推論するコード

## 1. 環境構築
```bash
pip install -U pip && pip install -r requirements.txt
```

## 2. 実行コマンド
### 画像:image
```bash
#templete
python onnx_inference.py -mo <data type> -m <onnx model path> -i <image path> -o <input dir> -s <score threshold> --input_shape <input size>

#example
python onnx_inference.py -mo image -m model/yolox_tiny.onnx -i sample_image.jpg -o outputs -s 0.3 --input_shape 416,416
```
### 動画:video
```bash
#templete
python onnx_inference.py -mo <data type> -m <onnx model path> -i <video path> -o outputs -s <score threshold> --input_shape <input size>

#example
python onnx_inference.py -mo video -m model/yolox_tiny.onnx -i sample_video.mp4 -o outputs -s 0.3 --input_shape 416,416
```
