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
# -*- coding:utf-8 -*-
import onnx
from onnx import helper
from fluid_onnx.variables import paddle_variable_to_onnx_tensor, paddle_onnx_weight

def split_model(model, outputs, global_block):
    """
    Takes a model and changes its outputs.

    :param model: *ONNX* model
    :param outputs: new outputs
    :return: modified model
    The function removes unneeded files.
    """
    if outputs is None:
        raise RuntimeError("outputs and inputs are None")
    if outputs == model.graph.output[0].name:
        return model

    nodes = model.graph.node

    keep_nodes = []
    # We mark all the nodes we need to keep.
    for node in nodes:
        if node.output[0] == outputs:
            keep_nodes.append(node)
            break
        keep_nodes.append(node)

    #infer_shapes = onnx.shape_inference.infer_shapes(model)

    var_out = []
    value_infos = []
    for value_info in model.graph.value_info:
        if value_info.name == outputs:
            var_out.append(value_info)
            value_infos.append(value_info)
            break
        value_infos.append(value_info)
    
    layer_mode_name = model.graph.name + outputs
    var_out = [paddle_variable_to_onnx_tensor(outputs, global_block)] 
    graph = helper.make_graph(nodes=keep_nodes, 
                              name=layer_mode_name,
                              inputs=model.graph.input,
                              initializer=model.graph.initializer,
                              outputs=var_out) 
    onnx_model = helper.make_model(graph)
    
    """
    onnx_model.ir_version = model.ir_version
    onnx_model.producer_name = model.producer_name
    onnx_model.producer_version = model.producer_version
    onnx_model.domain = model.domain
    onnx_model.model_version = model.model_version
    onnx_model.doc_string = model.doc_string
    """

    if len(onnx_model.graph.input) != len(model.graph.input):
        raise RuntimeError("Input mismatch {} != {}".format(
            len(onnx_model.input), len(model.input)))
    return onnx_model
