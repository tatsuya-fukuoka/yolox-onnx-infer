# YOLOX_onnx_inference
YOLOXでonnxモデルを用いて画像と動画を推論するコード

以下の方法ではなく、[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)のdemo/ONNXRnutimeフォルダに置いて実行しても良い。

## 1. 環境構築
```bash
pip install -U pip && pip install -r requirements.txt
```

## 2. 実行コマンド
### 画像:image
```bash
#templete
python onnx_inference_custom.py -mo <data type> -m <onnx model path> -i <image path> -o <input dir> -s <score threshold> --input_shape <input size>

#example
python onnx_inference_custom.py -mo image -m model/yolox_tiny.onnx -i sample_image.jpg -o outputs -s 0.3 --input_shape 416,416
```
### 動画:video
```bash
#templete
python onnx_inference_custom.py -mo <data type> -m <onnx model path> -i <video path> -o outputs -s <score threshold> --input_shape <input size>

#example
python onnx_inference_custom.py -mo video -m model/yolox_tiny.onnx -i sample_video.mp4 -o outputs -s 0.3 --input_shape 416,416
```
