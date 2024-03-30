# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import fastdeploy as fd
import cv2
import os

def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rec_label_file",
        required=True,
        help="Path of Recognization model of PPOCR.")
    parser.add_argument(
        "--model", required=True, help="Path of PaddleClas model.")
    parser.add_argument(
        "--image", type=str, required=True, help="Path of test image file.")
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        help="Type of inference device, support 'cpu' or 'gpu' or 'ipu' or 'kunlunxin' or 'ascend' ."
    )
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="Define which GPU card used to run model.")
    parser.add_argument(
        "--backend",
        type=str,
        default="default",
        help="Type of inference backend, support ort/trt/paddle/openvino, default 'openvino' for cpu, 'tensorrt' for gpu"
    )
    return parser.parse_args()

def build_option(args):

    option = fd.RuntimeOption()

    if args.device.lower() == "gpu":
        option.use_gpu(args.device_id)

    if args.backend.lower() == "trt":
        assert args.device.lower(
        ) == "gpu", "TensorRT backend require inference on device GPU."
        option.use_trt_backend()
        option.trt_option.set_shape("x", [1, 3, 48, 10], [1, 3, 48, 320],
                                    [1, 3, 48, 2304])
        
    elif args.backend.lower() == "pptrt":
        assert args.device.lower(
        ) == "gpu", "Paddle-TensorRT backend require inference on device GPU."
        option.use_paddle_infer_backend()
        option.paddle_infer_option.enable_trt = True
        option.paddle_infer_option.collect_trt_shape = True
        option.trt_option.set_shape("x", [1, 3, 48, 10], [1, 3, 48, 320],
                                    [1, 3, 48, 2304])

    elif args.backend.lower() == "ort":
        option.use_ort_backend()

    elif args.backend.lower() == "paddle":
        option.use_paddle_infer_backend()
        option.paddle_infer_option.enable_mkldnn = False

    elif args.backend.lower() == "openvino":
        assert args.device.lower(
        ) == "cpu", "OpenVINO backend require inference on device CPU."
        option.use_openvino_backend()

    elif args.backend.lower() == "pplite":
        assert args.device.lower(
        ) == "cpu", "Paddle Lite backend require inference on device CPU."
        option.use_lite_backend()

    return option

args = parse_arguments()

# 配置runtime，加载模型
option = build_option(args)
model_file = os.path.join(args.model, "inference.pdmodel")
params_file = os.path.join(args.model, "inference.pdiparams")
rec_label_file = args.rec_label_file
model = fd.vision.ocr.Recognizer(
    model_file, params_file, rec_label_file, runtime_option=option)
im = cv2.imread(args.image)
result = model.predict(im)
print(result)
