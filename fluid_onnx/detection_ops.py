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


def box_coder_op(operator, block):
   """
   In this function, we will use the decode the prior box to target box,
   we just use the decode mode to transform this op.
   """
   inputs, attrs, outputs = op_io_info(operator)
   node_list = []

   prior_var = block.vars[get_old_name(inputs['PriorBox'][0])]
   t_size = block.vars[get_old_name(inputs['TargetBox'][0])].shape
   p_size = prior_var.shape

   # get the outout_name 
   result_name = outputs['OutputBox'][0]
   # n is size of batch, m is boxes num of targe_boxes
   n = t_size[0]
   m = t_size[0]

   axis = int(attrs['axis'])


   #norm
   norm = bool(attrs['box_normalized'])

   name_slice_x1 = outputs['OutputBox'][0]+"@x1"
   name_slice_y1 = outputs['OutputBox'][0]+"@y1"
   name_slice_x2 = outputs['OutputBox'][0]+"@x2"
   name_slice_y2 = outputs['OutputBox'][0]+"@y2"

   #make onnx tensor to save the intermeidate reslut 
   name_slice_indices = [[outputs['OutputBox'][0] + "@slice_" + str(i)] for i in range(1, 3)] 
   node_slice_indices = [ None for i in range(1,3)]
      
   # create the range(0, 4) const data to slice 
   for i in range(1, 3):
       node = onnx.helper.make_node('Constant',
           inputs=[],
           outputs=name_slice_indices[i-1],
           value=onnx.helper.make_tensor(
           name=name_slice_indices[i-1][0]+"@const",
           data_type=onnx.TensorProto.FLOAT,
           dims=(),
           vals=[i]))
       node_list.append(node)
   # make node split data 
   name_box_split = [name_slice_x1, name_slice_y1, name_slice_x2, name_slice_y2]
   split_shape = list(p_size)
   split_shape[-1] = 1

   node_split_prior_node = onnx.helper.make_node('Split', 
       inputs=inputs['PriorBox'],
       outputs=name_box_split,
       axis=1)
   node_list.append(node_split_prior_node)

   # make node get centor node for decode
   final_outputs_vars = []
   if not norm: 
      name_centor_w_tmp = [outputs['OutputBox'][0] + "@centor_w_tmp"]
      name_centor_h_tmp = [outputs['OutputBox'][0] + "@centor_h_tmp"]
      node_centor_w_tmp = None 
      node_centor_h_tmp = None 
      name_centor_tmp_list = [name_centor_w_tmp, name_centor_h_tmp]
      node_centor_tmp_list = [node_centor_w_tmp, node_centor_h_tmp]

      count = 2
      for (name, node) in zip(name_centor_tmp_list, node_centor_tmp_list): 
          node = onnx.helper.make_node('Add', 
                 inputs=[outputs['OutputBox'][0] + "@slice_" + str(1)]\
                     + [name_box_split[count]],
                 outputs=name)
          node_list.append(node)
          count = count + 1
   if not norm:
       inputs_sub = [[name_centor_w_tmp[0], name_box_split[0]], [name_centor_h_tmp[0], name_box_split[1]]]
   else:
       inputs_sub = [[name_box_split[2], name_box_split[0]], [name_box_split[3], name_box_split[1]]]
   outputs_sub = [result_name+"@pb_w", result_name+"@pb_h"]  
   for i in range(0, 2):
      node = onnx.helper.make_node('Sub', 
          inputs=inputs_sub[i],
          outputs=[outputs_sub[i]])
      node_list.append(node)
   # according to prior_box height and weight to get centor x, y 
   name_half_value = [result_name + "@half_value"]
   node_half_value = onnx.helper.make_node('Constant',
       inputs=[],
       outputs=name_half_value,
       value=onnx.helper.make_tensor(
           name=name_slice_indices[i][0]+"@const",
           data_type=onnx.TensorProto.FLOAT,
           dims=(),
           vals=[0.5]))
   node_list.append(node_half_value)
   outputs_half_wh = [[result_name + "@pb_w_half"], [result_name + "@pb_h_half"]]
   inputs_half_wh = [[result_name+"@pb_w", name_half_value[0]],
                     [result_name+"@pb_h", name_half_value[0]]]
      
   for i in range(0, 2):
       node = onnx.helper.make_node('Mul', 
           inputs=inputs_half_wh[i],
           outputs=outputs_half_wh[i])
       node_list.append(node)

   inputs_centor_xy = [[outputs_half_wh[0][0], name_slice_x1],
                       [outputs_half_wh[1][0], name_slice_y1]]

   outputs_centor_xy = [[result_name + "@pb_x"],
                        [result_name + "@pb_y"]]

   # final calc the centor x ,y 
   for i in range(0, 2):
       node = onnx.helper.make_node('Add', inputs=inputs_centor_xy[i],
           outputs=outputs_centor_xy[i])
       node_list.append(node)
   # reshape the data
   shape = (1, split_shape[0]) if axis == 0 else (split_shape[0], 1)
   
   # need to reshape the data
   inputs_transpose_pb = [[result_name+ "@pb_w"],
                        [result_name+ "@pb_h"],
                        [result_name+ "@pb_x"],
                        [result_name+ "@pb_y"],
                        ]
   outputs_transpose_pb = [[result_name+ "@pb_w_transpose"],
                        [result_name+ "@pb_h_transpose"],
                        [result_name+ "@pb_x_transpose"],
                        [result_name+ "@pb_y_transpose"],
                        ]
   if axis == 0:
       name_reshape_pb = [result_name + "@pb_transpose"]
       # reshape the data 
       for i in range(0, 4):
           node = onnx.helper.make_node('Transpose',
                 inputs=inputs_transpose_pb[i],
                 outputs=outputs_transpose_pb[i])
           node_list.append(node) 
   # decoder the box according to the target_box and variacne  
   name_variance_raw = [result_name + "@variance_raw"]
   name_variance_unsqueeze = [result_name + "@variance_unsqueeze"]
   shape = []
   # make node to extend the data 
   var_split_axis = 0
   var_split_inputs_name = []
   if 'PriorBoxVar' in inputs and len(inputs['PriorBoxVar']) > 0: 
       if axis == 1:
           raise Exception("The op box_coder has variable do not support aixs broadcast")
       prior_variance_var = block.vars[get_old_name(inputs['PriorBoxVar'][0])]
       axes = []
       var_split_inputs_name = [result_name + "@variance_split"]
       node = onnx.helper.make_node('Transpose', inputs = inputs['PriorBoxVar'],
           outputs=var_split_inputs_name)
       node_list.append(node)
       var_split_axis = 0
   else:
       variances = [1.0, 1.0, 1.0, 1.0]
       if 'variance' in attrs and len(attrs['variance']) > 0:
           variances = [float(var) for var in attrs['variance']]
       node_variance_create = onnx.helper.make_node('Constant',
           inputs=[],
           outputs=name_variance_raw,
           value=onnx.helper.make_tensor(
           name=name_variance_raw[0]+"@const",
           data_type=onnx.TensorProto.FLOAT,
           dims=[len(variances)],
           vals=variances))
       node_list.append(node_variance_create)
       var_split_axis = 0
       var_split_inputs_name = name_variance_raw

   # decode the result 
   outputs_split_variance = [result_name + "@variance_split" + str(i) for i in range(0, 4)]
   outputs_split_targebox = [result_name + "@targebox_split" + str(i) for i in range(0, 4)]
   node_split_var = onnx.helper.make_node('Split',
       inputs=var_split_inputs_name,
       outputs=outputs_split_variance,
       axis=var_split_axis)
   node_split_target = onnx.helper.make_node('Split',
       inputs=inputs['TargetBox'],
       outputs=outputs_split_targebox,
       axis=2)
   node_list.extend([node_split_var, node_split_target])

   outputs_squeeze_targebox = [result_name + "@targebox_squeeze" + str(i) for i in range(0, 4)]
   for (input_name, output_name) in zip(outputs_split_targebox, outputs_squeeze_targebox):
       node = onnx.helper.make_node('Squeeze',
       inputs=[input_name],
       outputs=[output_name],
       axes=[2])
       node_list.append(node)

   output_shape_step1 = list(t_size)[:-1]

   inputs_tb_step1 = [[outputs_squeeze_targebox[0], outputs_split_variance[0]],
        [outputs_squeeze_targebox[1], outputs_split_variance[1]],
        [outputs_squeeze_targebox[2], outputs_split_variance[2]],
        [outputs_squeeze_targebox[3], outputs_split_variance[3]]] 
   outputs_tb_step1 = [[result_name + "@decode_x_step1"],
                      [result_name + "@decode_y_step1"],
                      [result_name + "@decode_w_step1"],
                      [result_name + "@decode_h_step1"]]

   for input_step1, output_step_1 in zip(inputs_tb_step1, outputs_tb_step1):
       node = onnx.helper.make_node('Mul',
           inputs=input_step1,
           outputs=output_step_1)
       node_list.append(node)
   if axis == 0:
       inputs_tbxy_step2 = [[outputs_tb_step1[0][0], outputs_transpose_pb[0][0]],
           [outputs_tb_step1[1][0], outputs_transpose_pb[1][0]]]
   else:
       inputs_tbxy_step2 = [[outputs_tb_step1[0][0], inputs_transpose_pb[0][0]],
           [outputs_tb_step1[1][0], inputs_transpose_pb[1][0]]]
  
   outputs_tbxy_step2 = [[result_name + "@decode_x_step2"],
                         [result_name + "@decode_y_step2"]]
 

   for input_step2, output_step_2 in zip(inputs_tbxy_step2, outputs_tbxy_step2):
       node = onnx.helper.make_node('Mul',
           inputs=input_step2,
           outputs=output_step_2)
       node_list.append(node)
   if axis == 0:
       inputs_tbxy_step3 = [[outputs_tbxy_step2[0][0], outputs_transpose_pb[2][0]],
          [outputs_tbxy_step2[1][0], outputs_transpose_pb[3][0]]]
   else:
       inputs_tbxy_step3 = [[outputs_tbxy_step2[0][0], inputs_transpose_pb[2][0]],
          [outputs_tbxy_step2[1][0], inputs_transpose_pb[3][0]]]
  
   outputs_tbxy_step3 = [[result_name + "@decode_x_step3"],
                       [result_name + "@decode_y_step3"]]

   for input_step3, output_step_3 in zip(inputs_tbxy_step3, outputs_tbxy_step3):
       node = onnx.helper.make_node('Add',
           inputs=input_step3,
           outputs=output_step_3)
       node_list.append(node)  

   # deal with width & height
   inputs_tbwh_step2 = [outputs_tb_step1[2],
                        outputs_tb_step1[3]]
   outputs_tbwh_step2 =[[result_name + "@decode_w_step2"],
                        [result_name + "@decode_h_step2"]] 

   for input_name, output_name in zip(inputs_tbwh_step2, outputs_tbwh_step2):
       node = onnx.helper.make_node('Exp',
           inputs=input_name,
           outputs=output_name)
       node_list.append(node)

   if axis == 0:
        inputs_tbwh_step3 = [[outputs_tbwh_step2[0][0], outputs_transpose_pb[0][0]],
            [outputs_tbwh_step2[1][0], outputs_transpose_pb[1][0]]]
   else:
       inputs_tbwh_step3 = [[outputs_tbwh_step2[0][0], inputs_transpose_pb[0][0]],
           [outputs_tbwh_step2[1][0], inputs_transpose_pb[1][0]]]

   outputs_tbwh_step3 = [[result_name + "@decode_w_step3"],
                         [result_name + "@decode_h_step3"]] 
        
   for input_name, output_name in zip(inputs_tbwh_step3, outputs_tbwh_step3):
       node = onnx.helper.make_node('Mul',
            inputs=input_name,
            outputs=output_name)
       node_list.append(node)


   # final step to calc the result, and concat the result to output 
   # return the output box, [(x1, y1), (x2, y2)]

   inputs_half_tbwh_step4 = [[outputs_tbwh_step3[0][0], result_name + "@slice_2"],
        [outputs_tbwh_step3[1][0], result_name + "@slice_2"]]

   outputs_half_tbwh_step4 = [[result_name + "@decode_half_w_step4"],
       [result_name + "@decode_half_h_step4"]]
   for inputs_name, outputs_name in zip(inputs_half_tbwh_step4, outputs_half_tbwh_step4):
       node = onnx.helper.make_node('Div', 
            inputs=inputs_name,
            outputs=outputs_name)
       node_list.append(node)
   inputs_output_point1 = [[outputs_tbxy_step3[0][0], outputs_half_tbwh_step4[0][0]],
       [outputs_tbxy_step3[1][0], outputs_half_tbwh_step4[1][0]]]
   
   outputs_output_point1 = [[result_name + "@ouput_x1"],
       [result_name + "@output_y1"]]
   for input_name, output_name in zip(inputs_output_point1, outputs_output_point1):
       node = onnx.helper.make_node('Sub',
           inputs=input_name, 
           outputs=output_name)
       node_list.append(node)

   inputs_output_point2 = [[outputs_tbxy_step3[0][0], outputs_half_tbwh_step4[0][0]],
       [outputs_tbxy_step3[1][0], outputs_half_tbwh_step4[1][0]]]
   
   outputs_output_point2 = [[result_name + "@ouput_x2"],
       [result_name + "@output_y2"]]

   for input_name, output_name in zip(inputs_output_point2, outputs_output_point2):
       node = onnx.helper.make_node('Add',
           inputs=input_name, 
           outputs=output_name)
       node_list.append(node)
   if not norm:
      inputs_unnorm_point2 = [[outputs_output_point2[0][0], result_name + "@slice_1"],
                              [outputs_output_point2[1][0], result_name + "@slice_1"]] 
      outputs_unnorm_point2 = [[result_name + "@ouput_unnorm_x2"],
                               [result_name + "@ouput_unnorm_y2"]]

      for input_name, output_name in zip(inputs_unnorm_point2, outputs_unnorm_point2):
          node = onnx.helper.make_node('Sub',
               inputs=input_name, 
               outputs=output_name)
          node_list.append(node)
      outputs_output_point2 = outputs_unnorm_point2
   
   outputs_output_point1.extend(outputs_output_point2)
   ouputs_points_unsqueeze = [[result_name + "@points_unsqueeze_x1"],
       [result_name + "points_unsqueeze_y1"],
       [result_name + "points_unsqueeze_x2"],
       [result_name + "points_unsqueeze_y2"]] 

   for input_name, output_name in zip(outputs_output_point1, ouputs_points_unsqueeze):
       node = onnx.helper.make_node('Unsqueeze',
           inputs=input_name,
           outputs=output_name,
           axes=[len(output_shape_step1)])
       node_list.append(node)
   outputs_points_unsqueeze_list = [output[0] for output in ouputs_points_unsqueeze] 
   node_point_final = onnx.helper.make_node('Concat', 
      inputs=outputs_points_unsqueeze_list,
      outputs=outputs['OutputBox'],
      axis=len(output_shape_step1))
   node_list.append(node_point_final)
  #@ output_shape_step1.append(4)
  #@ for output_name in  outputs_points_final:
  #@     value_info = onnx.helper.make_tensor_value_info(output_name,
  #@         PADDLE_TO_ONNX_DTYPE[prior_var.dtype], output_shape_step1) 
  #@     final_outputs_vars.append(value_info)
   return tuple(node_list)
