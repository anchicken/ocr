# PaddleOCR Python部署示例

本说明文档主要介绍部署示例使用方法。阅读本文档前建议提前阅读部署包目录下的部署包使用文档。

**使用部署示例前请确认已按照使用文档要求安装 FastDeploy**

## 1. 运行部署示例

|参数|含义|默认值|  
|---|---|---|
|--model|指定模型文件夹所在的路径|None|
|--rec_label_file|指定识别真值文件所在的路径（仅用于文本识别模型）|None|
|--image|指定测试图片所在的路径|None|  
|--device|指定即将运行的硬件类型，支持的值为`[cpu, gpu]`，当设置为cpu时，可运行在x86 cpu/arm cpu等cpu上|cpu|
|--backend|部署模型时使用的后端, 支持的值为`[paddle,pptrt,pplite,ort,openvino,trt]` |openvino|

**注意：FastDeploy优先保证模型使用Paddle Inference和Paddle-TensorRT后端部署的正确性，推荐优先使用这两个后端，其他后端可能出现部署错误，请知悉。**

```bash
# 找到部署包内的模型路径，例如 model

# 准备一张测试图片，例如test.jpg

# 如果是 文本识别模型
# 在CPU上使用Paddle Inference推理
python infer_rec.py --model model --rec_label_file ppocr_keys_v1.txt --image test.jpg --device cpu --backend paddle
# 在GPU上使用Paddle Inference推理（需要安装GPU版本的FastDeploy）
python infer_rec.py --model model --rec_label_file ppocr_keys_v1.txt --image test.jpg --device gpu --backend paddle

# 在GPU上使用Paddle TensorRT推理（需要安装GPU版本的FastDeploy）
# Paddle TensorRT推理后端可能存在TensorRT建图耗时较长情况，请留意
python infer_rec.py --model model --rec_label_file ppocr_keys_v1.txt --image test.jpg --device gpu --backend pptrt

# 如果是 文本检测模型
# 在CPU上使用Paddle Inference推理
python infer_det.py --model model --image test1.jpg --device cpu --backend paddle
# 在GPU上使用Paddle Inference推理（需要安装GPU版本的FastDeploy）
python infer_det.py --model model --image test1.jpg --device gpu --backend paddle

# 在GPU上使用Paddle TensorRT推理（需要安装GPU版本的FastDeploy）
# Paddle TensorRT推理后端可能存在TensorRT建图耗时较长情况，请留意
python infer_det.py --model model --image test1.jpg --device gpu --backend pptrt
```


## 2. 更多指南
- [PaddleOCR系列 Python API查阅](https://www.paddlepaddle.org.cn/fastdeploy-api-doc/cpp/html/namespacefastdeploy_1_1vision_1_1ocr.html)
