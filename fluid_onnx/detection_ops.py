# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import math
import onnx
import numpy as np
from functools import partial
from onnx import TensorProto
from onnx.helper import make_node, make_tensor
from paddle.fluid.executor import _fetch_var as fetch_var
from fluid.utils import op_io_info, get_old_name
from fluid_onnx.variables import PADDLE_TO_ONNX_DTYPE, paddle_onnx_shape

def multiclass_nms_op(operator, block):
    """
    Convert the paddle multiclass_nms to onnx op.
    This op is get the select boxes from origin boxes.
    """
    inputs, attrs, outputs = op_io_info(operator)
    #convert the paddle attribute to onnx tensor 
    name_score_threshold = [outputs['Out'][0] + "@score_threshold"]
    name_iou_threshold = [outputs['Out'][0] + "@iou_threshold"]
    name_keep_top_k = [outputs['Out'][0] + '@keep_top_k ']

    node_score_threshold = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=name_score_threshold,
        value=onnx.helper.make_tensor(
             name=name_score_threshold[0]+"@const",
             data_type=onnx.TensorProto.FLOAT,
             dims=(),
             vals=[float(attrs['score_threshold'])]))

    node_iou_threshold = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=name_iou_threshold,
        value=onnx.helper.make_tensor(
             name=name_iou_threshold[0]+"@const",
             data_type=onnx.TensorProto.FLOAT,
             dims=(),
             vals=[float(attrs['nms_threshold'])]))

    node_keep_top_k = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=name_keep_top_k,
        value=onnx.helper.make_tensor(
             name=name_keep_top_k[0]+"@const",
             data_type=onnx.TensorProto.INT64,
             dims=(),
             vals=[np.int64(attrs['keep_top_k'])]))

    # the paddle data format is x1,y1,x2,y2
    kwargs = {'center_point_box' : 0}

    name_select_nms = [outputs['Out'][0] + "@select_index"]
    node_select_nms= onnx.helper.make_node(
        'NonMaxSuppression',
        inputs=inputs['BBoxes'] + inputs['Scores'] + name_keep_top_k +\
            name_iou_threshold + name_score_threshold,
        outputs=name_select_nms)
    # step 1 nodes select the nms class 
    node_list = [node_score_threshold, node_iou_threshold, node_keep_top_k, node_select_nms]
    return tuple(node_list)

    """
    ## this op is slice the class from nms 
    # create the const value to slice 
    Constant_684 = ["Constant_684"]
    Constant_685 = ["Constant_685"]
    Constant_686 = ["Constant_686"]
    Constant_687 = ["Constant_687"]
    name_slice_step = [outputs['Out'] + "@slice_step"]

    node_Constant_684 = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=name_slice_start,
        value=onnx.helper.make_tensor(
             name=Constant_684+"@const",
             data_type=onnx.TensorProto.INT32,
             dims=(),
             vals=[]))

    node_Constant_685 = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=name_slice_start,
        value=onnx.helper.make_tensor(
             name=Constant_685+"@const",
             data_type=onnx.TensorProto.INT32,
             dims=(),
             vals=[]))
    node_Constant_686 = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=name_slice_start,
        value=onnx.helper.make_tensor(
             name=Constant_686+"@const",
             data_type=onnx.TensorProto.INT32,
             dims=(),
             vals=[]))
    node_Constant_687 = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=name_slice_start,
        value=onnx.helper.make_tensor(
             name=Constant_687+"@const",
             data_type=onnx.TensorProto.INT32,
             dims=(),
             vals=[]))
    node_step_2 = [node_Constant_684, node_Constant_685, node_Constant_686, node_Constant_687]
    node_list.extend(node_step_2)
    return (node_score_threshold, node_iou_threshold, node_keep_top_k, node_slect_nms)
    """





def ExpandAspectRations(input_aspect_ratior, flip):
    expsilon = 1e-6
    output_ratios = [1.0]
    for input_ratio in input_aspect_ratior:
        already_exis = False
        for output_ratio in output_ratios:
            if abs(input_ratio - output_ratio) < expsilon:
                already_exis = True
                break
        if already_exis == False:
             output_ratios.append(input_ratio) 
             if flip:
                 output_ratios.append(1.0/input_ratio)
    return output_ratios


def prior_box_op(operator, block):
    """
    In this function, use the attribute to get the prior box, because we do not use 
    the image data and feature map, wo could the python code to create the varaible, 
    and to create the onnx tensor as output.
    """
    inputs, attrs, outputs = op_io_info(operator)
    flip = bool(attrs['flip'])
    clip = bool(attrs['clip'])
    min_max_aspect_ratios_order = bool(attrs['min_max_aspect_ratios_order'])
    min_sizes = [float(size) for size in attrs['min_sizes']]
    max_sizes = [float(size) for size in attrs['max_sizes']]
    if isinstance(attrs['aspect_ratios'], list):
        aspect_ratios = [float(ratio) for ratio in attrs['aspect_ratios']]
    else:
        aspect_ratios = [float(attrs['aspect_ratios'])]
    variances = [float(var) for var in attrs['variances']]
    # set min_max_aspect_ratios_order = false 
    output_ratios = ExpandAspectRations(aspect_ratios, flip)

    step_w = float(attrs['step_w'])
    step_h = float(attrs['step_h'])
    offset = float(attrs['offset'])

    input_shape = block.vars[get_old_name(inputs['Input'][0])].shape
    image_shape = block.vars[get_old_name(inputs['Image'][0])].shape

    img_width =  image_shape[3]
    img_height =  image_shape[2]
    feature_width = input_shape[3]
    feature_height = input_shape[2]

    step_width = 1.0 
    step_height = 1.0

    if step_w == 0.0 or step_h == 0.0:
        step_width = float(img_width/feature_width)
        step_height = float(img_height/feature_height)
    else:
        step_width = step_w 
        step_height = step_h 

    num_priors = len(output_ratios) * len(min_sizes)
    if len(max_sizes) > 0:
       num_priors += len(max_sizes)
    out_dim = (feature_height, feature_width, num_priors, 4)
    out_boxes = np.zeros(out_dim).astype('float32')
    out_var = np.zeros(out_dim).astype('float32')

    idx = 0
    for h in range(feature_height):
        for w in range(feature_width):
            c_x = (w + offset) * step_w
            c_y = (h + offset) * step_h
            idx = 0
            for s in range(len(min_sizes)):
                min_size = min_sizes[s]
                if not min_max_aspect_ratios_order:
                    # rest of priors
                    for r in range(len(output_ratios)):
                        ar = output_ratios[r]
                        c_w = min_size * math.sqrt(ar) / 2
                        c_h = (min_size / math.sqrt(ar)) / 2
                        out_boxes[h, w, idx, :] = [
                            (c_x - c_w) / img_width, (c_y - c_h) /
                            img_height, (c_x + c_w) / img_width,
                            (c_y + c_h) / img_height
                        ]
                        idx += 1

                    if len(max_sizes) > 0:
                        max_size = max_sizes[s]
                        # second prior: aspect_ratio = 1,
                        c_w = c_h = math.sqrt(min_size * max_size) / 2
                        out_boxes[h, w, idx, :] = [
                            (c_x - c_w) / img_width, (c_y - c_h) /
                            img_height, (c_x + c_w) / img_width,
                            (c_y + c_h) / img_height
                        ]
                        idx += 1
                else:
                    c_w = c_h = min_size / 2.
                    out_boxes[h, w, idx, :] = [(c_x - c_w) / img_width,
                                               (c_y - c_h) / img_height,
                                               (c_x + c_w) / img_width,
                                               (c_y + c_h) / img_height]
                    idx += 1
                    if len(max_sizes) > 0:
                        max_size = max_sizes[s]
                        # second prior: aspect_ratio = 1,
                        c_w = c_h = math.sqrt(min_size * max_size) / 2
                        out_boxes[h, w, idx, :] = [
                            (c_x - c_w) / img_width, (c_y - c_h) /
                            img_height, (c_x + c_w) / img_width,
                            (c_y + c_h) / img_height
                        ]
                        idx += 1

                    # rest of priors
                    for r in range(len(output_ratios)):
                        ar = output_ratios[r]
                        if abs(ar - 1.) < 1e-6:
                            continue
                        c_w = min_size * math.sqrt(ar) / 2
                        c_h = (min_size / math.sqrt(ar)) / 2
                        out_boxes[h, w, idx, :] = [
                            (c_x - c_w) / img_width, (c_y - c_h) /
                            img_height, (c_x + c_w) / img_width,
                            (c_y + c_h) / img_height
                        ]
                        idx += 1

    if clip:
        out_boxes = np.clip(out_boxes, 0.0, 1.0)
    # set the variance.
    out_var = np.tile(variances, (feature_height, feature_width,  num_priors,1)) 

    #make node that 
    node_boxes = onnx.helper.make_node('Constant',
                      inputs=[],
                      outputs=outputs['Boxes'],
                      value=onnx.helper.make_tensor(
                      name=outputs['Boxes'][0]+"@const",
                      data_type=onnx.TensorProto.FLOAT,
                      dims=out_boxes.shape,
                      vals=out_boxes.flatten()))
    node_vars = onnx.helper.make_node('Constant',
                      inputs=[],
                      outputs=outputs['Variances'],
                      value=onnx.helper.make_tensor(
                      name=outputs['Variances'][0]+"@const",
                      data_type=onnx.TensorProto.FLOAT,
                      dims=out_var.shape,
                      vals=out_var.flatten()))
    return (node_boxes, node_vars)







