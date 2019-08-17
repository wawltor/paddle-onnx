# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import numpy as np

import paddle.fluid as fluid
from onnx import helper, checker, load

import fluid_onnx.ops as ops
from fluid_onnx.variables import paddle_variable_to_onnx_tensor
from fluid_onnx.variables import PADDLE_TO_ONNX_DTYPE


def parse_args():
    # Read arguments: path to model.
    parser = argparse.ArgumentParser("Use dummy data in the interval [a, b] "
                                     "as inputs to verify the conversion.")
    parser.add_argument(
        "--fluid_model",
        required=True,
        help="The path to PaddlePaddle Fluid model.")
    parser.add_argument(
        "--onnx_model", required=True, help="The path to ONNX model.")
    parser.add_argument(
        "--a",
        type=float,
        default=0.0,
        help="Left boundary of dummy data. (default: %(default)f)")
    parser.add_argument(
        "--b",
        type=float,
        default=1.0,
        help="Right boundary of dummy data. (default: %(default)f)")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Batch size. (default: %(default)d)")
    parser.add_argument(
        "--expected_decimal",
        type=int,
        default=5,
        help="The expected decimal accuracy. (default: %(default)d)")
    parser.add_argument(
        "--backend",
        type=str,
        choices=['caffe2', 'tensorrt'],
        default='caffe2',
        help="The ONNX backend used for validation. (default: %(default)s)")
    parser.add_argument(
        "--use_data_reader",
        type=bool,
        action='store_true',
        default=False,
        help="Use the real image or nlp data to valid model,(default: %(default)b)"
        )
    parser.add_argument(
        "--image_path",
        type=str,
        default="",
        help="The image path to read"
        )
    args = parser.parse_args()
    return args


def print_arguments(args):
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).iteritems()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')

def random_reader(program, feed_target_names):
    a = 0.0
    b = 1.0
    input_shapes = [
        program.global_block().var(var_name).shape
        for var_name in feed_target_names
    ]
    input_shapes = [
        shape if shape[0] > 0 else (1, ) + shape[1:]
        for shape in input_shapes
    ]
    for i in range(1):
        # Generate dummy data as inputs
        inputs = [
            (b - a) * np.random.random(shape).astype("float32") + a
            for shape in input_shapes
        ]
        yield inputs
        
       

def validate(args):
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    # load the paddle and onnx model 
    [fluid_infer_program, feed_target_names,
     fetch_targets] = fluid.io.load_inference_model(args.fluid_model, exe)

    # ONNX inference, using caffe2 as the backend
    onnx_model = load(args.onnx_model)
    if args.backend == 'caffe2':
        from caffe2.python.onnx.backend import Caffe2Backend
        rep = Caffe2Backend.prepare(onnx_model, device='CPU')
    else:
        import onnx_tensorrt.backend as backend
        rep = backend.prepare(onnx_model, device='CUDA:0')
    # select the random data reader or use define reader
    reader = random_reader
    fluid_results_all = []
    onnx_results_all = []
    for inputs in reader():
        # Fluid inference 
        fluid_results = exe.run(fluid_infer_program,
                                feed=dict(zip(feed_target_names, inputs)),
                                fetch_list=fetch_targets)
        fluid_results_all.append(fluid_results)

        #ONNX inference
        onnx_results = rep.run(inputs)
        onnx_results_all.append(onnx_results)


    # Remove these prints some day
    print("Inference results for fluid model:")
    print(fluid_results)
    print('\n')


    print("Inference results for ONNX model:")
    print(onnx_results)
    print('\n')
    
    #check result diff in paddle and onnx model
    batch_id = 0
    for fluid_results, onnx_results in zip(fluid_results_all, onnx_results_all): 
        print("The btach id is %d"%(batch_id))
        batch_inner_id = 0
        for ref, hyp in zip(fluid_results, onnx_results):
            print("The inner id is %d"%(batch_inner_id))
            np.testing.assert_almost_equal(ref, hyp, decimal=args.expected_decimal)
            batch_inner_id += 1
        batch_id += 1

    print("The exported model achieves {}-decimal precision.".format(
                args.expected_decimal))


if __name__ == "__main__":
    args = parse_args()
    print_arguments(args)
    validate(args)
