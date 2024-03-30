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
        "--model", required=True, help="Path of Detection model of PPOCR.")
    parser.add_argument(
        "--image", type=str, required=True, help="Path of test image file.")
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        help="Type of inference device, support 'cpu', 'kunlunxin' or 'gpu'.")
    parser.add_argument(
        "--backend",
        type=str,
        default="default",
        help="Type of inference backend, support ort/trt/paddle/openvino, default 'openvino' for cpu, 'tensorrt' for gpu"
    )
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="Define which GPU card used to run model.")
    return parser.parse_args()


def build_option(args):

    option = fd.RuntimeOption()

    if args.device.lower() == "gpu":
        option.use_gpu(args.device_id)

    if args.backend.lower() == "trt":
        assert args.device.lower(
        ) == "gpu", "TensorRT backend require inference on device GPU."
        option.use_trt_backend()
        option.trt_option.set_shape("x", [1, 3, 64, 64], [1, 3, 640, 640],
                                    [1, 3, 960, 960])

    elif args.backend.lower() == "pptrt":
        assert args.device.lower(
        ) == "gpu", "Paddle-TensorRT backend require inference on device GPU."
        option.use_paddle_infer_backend()
        option.paddle_infer_option.enable_trt = True
        option.paddle_infer_option.collect_trt_shape = True
        option.trt_option.set_shape("x", [1, 3, 64, 64], [1, 3, 640, 640],
                                    [1, 3, 960, 960])

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

det_model_file = os.path.join(args.model, "inference.pdmodel")
det_params_file = os.path.join(args.model, "inference.pdiparams")

# Set the runtime option
det_option = build_option(args)

# Create the det_model
det_model = fd.vision.ocr.DBDetector(
    det_model_file, det_params_file, runtime_option=det_option)

# Set the preporcessing parameters
det_model.preprocessor.max_side_len = 960
det_model.postprocessor.det_db_thresh = 0.3
det_model.postprocessor.det_db_box_thresh = 0.6
det_model.postprocessor.det_db_unclip_ratio = 1.5
det_model.postprocessor.det_db_score_mode = "slow"
det_model.postprocessor.use_dilation = False

# Read the image
im = cv2.imread(args.image)

# Predict and return the results
result = det_model.predict(im)

print(result)

# Visualize the results
# vis_im = fd.vision.vis_ppocr(im, result)
# cv2.imwrite("visualized_result.jpg", vis_im)
# print("Visualized result save in ./visualized_result.jpg")
