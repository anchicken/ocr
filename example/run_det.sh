#!/usr/bin/env bash
set -x

# 准备测试模型和测试资源
cp -r ../model* . 

# 如果是 文本检测模型
# 在CPU上使用Paddle Inference推理
python infer_det.py --model model --image test1.jpg --device cpu --backend paddle
# 在CPU上使用OpenVINO推理
python infer_det.py --model model --image test1.jpg --device cpu --backend openvino
# 在CPU上使用ONNX Runtime推理
python infer_det.py --model model --image test1.jpg --device cpu --backend ort
# 在GPU上使用Paddle Inference推理
python infer_det.py --model model --image test1.jpg --device gpu --backend paddle
# 在GPU上使用Paddle TensorRT推理
python infer_det.py --model model --image test1.jpg --device gpu --backend pptrt
# 在GPU上使用ONNX Runtime推理
python infer_det.py --model model --image test1.jpg --device gpu --backend ort
# 在GPU上使用Nvidia TensorRT推理
python infer_det.py --model model --image test1.jpg --device gpu --backend trt

set +x