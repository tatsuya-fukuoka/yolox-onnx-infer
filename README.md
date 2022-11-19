# YOLOX_onnx_inference
YOLOXでonnxモデルを用いて画像と動画を推論するコード

# 実行コマンド
## 画像:image
```bash
python demo/ONNXRuntime/onnx_inference_custom.py -mo image -m yolox_tiny.onnx -i sample_image.jpg -o outputs -s 0.3 --input_shape 416,416
```
## 動画:video
```bash
python demo/ONNXRuntime/onnx_inference_custom.py -mo video -m yolox_tiny.onnx -i sample_video.mp4 -o outputs -s 0.3 --input_shape 416,416
```
