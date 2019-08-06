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
import onnx
import numpy as np
from functools import partial
from onnx import TensorProto
from onnx.helper import make_node, make_tensor
from paddle.fluid.executor import _fetch_var as fetch_var
from fluid.utils import op_io_info, get_old_name
from fluid_onnx.variables import PADDLE_TO_ONNX_DTYPE, paddle_onnx_shape
"""
Priority of ops (uniques) to figure out support for.

test_fit_a_line.py
- mean
- mul
- elementwise_add
- elementwise_sub
- fill_constant

^ Try to make this run before proceeding.

test_machine_translation.py
- lookup_table
- tanh
- lstm
- sequence_pool
- lookup_table
- lod_rank_table
- max_sequence_len
- less_than
- lod_tensor_to_array
- write_to_array
- while
- array_to_lod_tensor
- cross_entropy
- lod_tensor_to_array
- read_from_array
- sum
- scale
- adagrad
- shrink_rnn_memory
- softmax
- write_to_array
- increment
"""

__onnx_ver__ = onnx.version.version

__name_prefix__ = ""

def init_name_prefix(name_prefix=""):
    gloab

def activation_ops(act_type, operator, block):
    """ Convert common activations with type specified by 'act_type', including
        'abs', 'ceil', 'exp', 'floor', 'log', 'reciprocal', 'relu', 'sigmoid',
        'softplus', 'softsign', 'sqrt' and 'tanh'.
    """

    inputs, _, outputs = op_io_info(operator)
    return make_node(
        act_type, inputs=list(inputs.values())[0], outputs=list(outputs.values())[0])


def argmax_op():
    pass


def argmin_op():
    pass


def batch_norm_op(operator, block):
    inputs, attrs, outputs = op_io_info(operator)

    x_shape = block.vars[get_old_name(inputs['X'][0])].shape
    nodes = ()
    # Batch norm in ONNX only supports input dim. >= 3, for input tensor with 
    # dim. == 2 supported by Fluid, reshape it into dim. == 4.
    if len(x_shape) == 2:
        new_shape = [0, x_shape[1], 1, 1]
        reshaped_x = [inputs['X'][0] + '@reshape_0']
        if __onnx_ver__ == '1.0.1':
            reshape_node = make_node(
                'Reshape',
                inputs=inputs['X'],
                shape=new_shape,
                outputs=reshaped_x)
        else:
            new_shape_name = [inputs['X'][0] + '@shape_tensor_0']
            new_shape_node = make_node(
                'Constant',
                inputs=[],
                outputs=new_shape_name,
                value=make_tensor(
                    name=new_shape_name[0],
                    data_type=TensorProto.INT64,
                    dims=(4, ),
                    vals=new_shape))
            reshape_node = make_node(
                'Reshape',
                inputs=inputs['X'] + new_shape_name,
                outputs=reshaped_x)
            nodes = (new_shape_node, )
        nodes += (reshape_node, )
    else:
        reshaped_x = inputs['X']

    kwargs = {
        'epsilon': attrs['epsilon'],
        'momentum': attrs['momentum']
    }
    # In v1.0.1, need to set input(Mean) and input(Variance) to be consumed 
    # explicitly. 
    if __onnx_ver__ == '1.0.1':
        kwargs['consumed_inputs'] = [0, 0, 0, 1, 1]

    bn_node = make_node(
        'BatchNormalization',
        inputs=reshaped_x + inputs['Scale'] + inputs['Bias'] + inputs['Mean'] +
        inputs['Variance'],
        outputs=outputs['Y'],
        **kwargs)

    return nodes + (bn_node, )


def cast_op(operator, block):
    inputs, attrs, outputs = op_io_info(operator)
    return make_node(
        'Cast',
        inputs=inputs['X'],
        outputs=outputs['Out'],
        to=PADDLE_TO_ONNX_DTYPE[attrs['out_dtype']])


def clip_op(operator, block):
    inputs, attrs, outputs = op_io_info(operator)
    return make_node(
        'Clip',
        inputs=inputs['X'],
        outputs=outputs['Out'],
        min=attrs['min'],
        max=attrs['max'])


def compare_ops(op_type, operator, block):
    ''' Conversion for compare ops, including 'Less', 'Equal', 'Greater'
    '''
    inputs, attrs, outputs = op_io_info(operator)
    return make_node(
        op_type, inputs=inputs['X'] + inputs['Y'], outputs=outputs['Out'])


def concat_op(operator, block):
    inputs, attrs, outputs = op_io_info(operator)
    return make_node(
        'Concat',
        inputs=inputs['X'],
        outputs=outputs['Out'],
        axis=attrs['axis'])


def constant_op(var, scope):
    data = fetch_var(var.name, scope)
    constant_node = make_node(
        'Constant',
        inputs=[],
        outputs=[var.name],
        value=make_tensor(
            name=var.name,
            dims=var.shape,
            data_type=PADDLE_TO_ONNX_DTYPE[var.dtype],
            vals=data.flatten().tolist()))
    return constant_node


def conv2d_op(operator, block):
    inputs, attrs, outputs = op_io_info(operator)

    kernel_shape = block.vars[get_old_name(inputs['Filter'][0])].shape
    conv2d = make_node(
        'Conv',
        inputs=inputs['Input'] + inputs['Filter'],
        outputs=outputs['Output'],
        dilations=attrs['dilations'],
        kernel_shape=kernel_shape[-2:],
        strides=attrs['strides'],
        group=attrs['groups'],
        pads=attrs['paddings'] + attrs['paddings'])
    return conv2d


def conv2d_transpose_op(operator, block):
    inputs, attrs, outputs = op_io_info(operator)

    kernel_shape = block.vars[get_old_name(inputs['Filter'][0])].shape
    conv2d_transpose = make_node(
        'ConvTranspose',
        inputs=inputs['Input'] + inputs['Filter'],
        outputs=outputs['Output'],
        dilations=attrs['dilations'],
        kernel_shape=kernel_shape[-2:],
        strides=attrs['strides'],
        group=1,
        pads=attrs['paddings'] + attrs['paddings'])
    return conv2d_transpose


def depthtospace_op():
    pass


def dropout_op(operator, block):
    inputs, attrs, outputs = op_io_info(operator)
    scale_input = [outputs['Out'][0] + '@dropout']
    dropout_node = make_node(
        'Dropout',
        inputs=inputs['X'],
        outputs=scale_input + outputs['Mask'],
        is_test=attrs['is_test'],
        ratio=attrs['dropout_prob'])

    ## Fluid and ONNX use different dropout formula
    # ONNX 1.0.1 doesn't support Scale op
    if __onnx_ver__ == '1.0.1':
        scale_val = [outputs['Out'][0] + '@scale']
        constant_node = make_node(
            'Constant',
            inputs=[],
            outputs=scale_val,
            value=make_tensor(
                name=scale_val[0],
                dims=(),
                data_type=TensorProto.FLOAT,
                vals=[1.0 - attrs['dropout_prob']]))
        mul_node = make_node(
            'Mul',
            inputs=scale_input + scale_val,
            outputs=outputs['Out'],
            broadcast=1)
        nodes = (dropout_node, constant_node, mul_node)
    else:
        scale_node = make_node(
            'Scale',
            inputs=scale_input,
            outputs=outputs['Out'],
            scale=1.0 - attrs['dropout_prob'])
        nodes = (dropout_node, scale_node)
    return nodes


def elementwise_ops(op_type, operator, block):
    """Convert elementwise operators From to ONNX. Supported elementwise 
       'op_type' includes 'Add', 'Div', 'Mul', 'Pow' and 'Sub'. 
    """

    inputs, attrs, outputs = op_io_info(operator)
    #rank_x = len(block.vars[get_old_name(inputs['X'][0])].shape)
    #rank_y = len(block.vars[get_old_name(inputs['Y'][0])].shape)
    #axis = rank_x - rank_y if attrs['axis'] == -1 else attrs['axis']
    return make_node(
        op_type,
        inputs=inputs['X'] + inputs['Y'],
        outputs=outputs['Out'])


def elu_op(operator, block):
    inputs, attrs, outputs = op_io_info(operator)
    return make_node(
        'Elu', inputs=inputs['X'], outputs=outputs['Out'], alpha=attrs['alpha'])


def equal_op():
    pass


def flatten_op():
    pass


def gru_op():
    pass


def gather_op(operator, block):
    inputs, attrs, outputs = op_io_info(operator)
    return make_node(
        'Gather', inputs=inputs['X'] + inputs['Index'], outputs=outputs['Out'])


def gemm_op():
    pass


def globallppool_op():
    pass


def hardsigmoid_op(operator, block):
    inputs, attrs, outputs = op_io_info(operator)
    return make_node(
        'HardSigmoid',
        inputs=inputs['X'],
        outputs=outputs['Out'],
        alpha=0.2,
        beta=0.5)
    pass


def hardmax_op():
    pass


def instancenormalization_op():
    pass


def lrn_op(operator, block):
    inputs, attrs, outputs = op_io_info(operator)
    return make_node(
        'LRN',
        inputs=inputs['X'],
        outputs=outputs['Out'],
        alpha=attrs['alpha'],
        beta=attrs['beta'],
        bias=attrs['k'],
        size=attrs['n'])


def lstm_op():
    pass


def leaky_relu_op(operator, block):
    inputs, attrs, outputs = op_io_info(operator)
    return make_node(
        'LeakyRelu',
        inputs=inputs['X'],
        outputs=outputs['Out'],
        alpha=attrs['alpha'])


def binary_logical_ops(op_type, operator, block):
    """Convert binary logical operators, i.e. 'And', 'Or' and 'Xor'.
    """

    inputs, _, outputs = op_io_info(operator)
    return make_node(
        op_type, inputs=inputs['X'] + inputs['Y'], outputs=outputs['Out'])


def unary_logical_ops(op_type, operator, block):
    """Convert unary logical operators, i.e. 'Not'.
    """

    inputs, _, outputs = op_io_info(operator)
    return make_node(op_type, inputs=inputs['X'], outputs=outputs['Out'])


def logsoftmax_op():
    pass


def lpnormalization_op():
    pass


def lppool_op():
    pass


def mul_op(operator, block):
    inputs, attrs, outputs = op_io_info(operator)

    # Get shape of inputs 
    x_shape = block.vars[get_old_name(inputs['X'][0])].shape
    y_shape = block.vars[get_old_name(inputs['Y'][0])].shape
    x_shape = paddle_onnx_shape(x_shape)
    y_shape = paddle_onnx_shape(y_shape)
    x_num_col_dims, y_num_col_dims = attrs['x_num_col_dims'], attrs[
        'y_num_col_dims']
    out_shape = x_shape[:x_num_col_dims] + y_shape[y_num_col_dims:]

    # Flatten input(X) and input(Y) into 2-D matries
    x_flat_out = [inputs['X'][0] + '@flatten_0']
    y_flat_out = [inputs['Y'][0] + '@flatten_0']

    # Because in TensorRT backend, Flatten op only accepts input tensor with 
    # dimension 3, here we use Reshape op to flatten the input tensor when 
    # ONNX is v1.0.1. 
    if __onnx_ver__ == '1.0.1':
        # In v1.0.1, shape is the attribute of Reshape op, not an input tensor.
        flatten_x_node = make_node(
            'Reshape',
            inputs=inputs['X'],
            outputs=x_flat_out,
            shape=[
                np.prod(x_shape[:x_num_col_dims]),
                np.prod(x_shape[x_num_col_dims:])
            ])
        flatten_y_node = make_node(
            'Reshape',
            inputs=inputs['Y'],
            outputs=y_flat_out,
            shape=[
                np.prod(y_shape[:y_num_col_dims]),
                np.prod(y_shape[y_num_col_dims:])
            ])
    else:
        flatten_x_node = make_node(
            'Flatten',
            inputs=inputs['X'],
            outputs=x_flat_out,
            axis=attrs['x_num_col_dims'])
        flatten_y_node = make_node(
            'Flatten',
            inputs=inputs['Y'],
            outputs=y_flat_out,
            axis=attrs['y_num_col_dims'])

    # Mat mul 
    matmul_out = [outputs['Out'][0] + '@matmul_0']
    matmul_node = make_node(
        'MatMul', inputs=x_flat_out + y_flat_out, outputs=matmul_out)

    nodes = (flatten_x_node, flatten_y_node, matmul_node)
    # Reshpe output
    if __onnx_ver__ == '1.0.1':
        output_node = make_node(
            'Reshape',
            inputs=matmul_out,
            shape=out_shape,
            outputs=outputs['Out'])
        nodes += (output_node, )
    else:
        output_shape_name = [outputs['Out'][0] + '@shape_0']
        output_shape_node = make_node(
            'Constant',
            inputs=[],
            outputs=output_shape_name,
            value=make_tensor(
                name=output_shape_name[0],
                data_type=TensorProto.INT64,
                dims=(len(out_shape), ),
                vals=out_shape))
        output_node = make_node(
            'Reshape',
            inputs=matmul_out + output_shape_name,
            outputs=outputs['Out'])
        nodes += (output_shape_node, output_node)

    return nodes


def max_op():
    pass


def maxroipool_op():
    pass


def mean_op():
    pass


def min_op():
    pass


def neg_op():
    pass


def prelu_op(operator, block):
    inputs, attrs, outputs = op_io_info(operator)
    return make_node(
        'PRelu', inputs=inputs['X'] + inputs['Alpha'], outputs=outputs['Out'])


def pad_op():
    pass


def pool2d_op(operator, block):
    inputs, attrs, outputs = op_io_info(operator)
    if attrs['global_pooling'] is False:
        op_type = {'max': 'MaxPool', 'avg': 'AveragePool'}
        pool2d = make_node(
            op_type[attrs['pooling_type']],
            inputs=inputs['X'],
            outputs=outputs['Out'],
            kernel_shape=attrs['ksize'],
            strides=attrs['strides'],
            pads=attrs['paddings'] + attrs['paddings'], )
    else:
        op_type = {'max': 'GlobalMaxPool', 'avg': 'GlobalAveragePool'}
        pool2d = make_node(
            op_type[attrs['pooling_type']],
            inputs=inputs['X'],
            outputs=outputs['Out'])
    return pool2d


def pow_op():
    pass


def rnn_op():
    pass


def randomnormal_op():
    pass


def randomnormallike_op():
    pass


def randomuniform_op():
    pass


def randomuniformlike_op():
    pass


def reduce_ops(op_type, operator, block):
    """Convert reduce operators in Fluid to ONNX. 'op_type' specifies the 
       target ONNX operator type, supporting 'Reduce{Max, Mean, Min, Sum}'
       right now.
    """

    inputs, attrs, outputs = op_io_info(operator)
    rank = len(block.vars[get_old_name(inputs['X'][0])].shape)
    dim = attrs['dim']
    axes = [dim if dim >= 0 else rank + dim]
    reduce_out = [outputs['Out'][0] + '@reduce_0'] if attrs[
        'reduce_all'] else outputs
    reduce_node = make_node(
        op_type,
        inputs=inputs['X'],
        outputs=reduce_out,
        axes=axes,
        keepdims=attrs['keep_dim'])
    if attrs['reduce_all'] is True:
        axes = range(rank) if attrs['keep_dim'] else range(rank - 1)
        reduce_all_node = make_node(
            op_type,
            inputs=reduce_out,
            outputs=outputs,
            axes=axes,
            keepdims=attrs['keep_dim'])
        return (reduce_node, reduce_all_node)
    return reduce_node


def reducel1_op():
    pass


def reducel2_op():
    pass


def reducelogsum_op():
    pass


def reducelogsumexp_op():
    pass


def reduceprod_op():
    pass


def reducesumsquare_op():
    pass


def reshape_op(operator, block):
    inputs, attrs, outputs = op_io_info(operator)
    shape_name = ""
    if 'Shape' in inputs and inputs['Shape'] is not None and len(inputs['Shape']) > 0:
        shape_name = inputs['Shape'] 
        return make_node('Reshape', inputs=[inputs['X'][0], shape_name], outputs=outputs['Out'])
    elif 'ShapeTensor' in inputs and inputs['ShapeTensor'] is not None and len(inputs['ShapeTensor']) > 0:
        shape_name = inputs['ShapeTensor']
        return make_node('Reshape', inputs=[inputs['X'][0], shape_name], outputs=outputs['Out'])
    else:
        shape_name = outputs['Out'][0] + "@shape_var"
        output_shape_node = make_node(
            'Constant',
            inputs=[],
            outputs=[shape_name],
            value=make_tensor(
                  name=shape_name,
                  data_type=TensorProto.INT64,
                  dims=(len(attrs['shape']), ),
                  vals=attrs['shape']))
        reshape_node = make_node('Reshape', inputs=[inputs['X'][0], shape_name], outputs=outputs['Out'])
        return (output_shape_node, reshape_node)


def selu_op():
    pass


def shape_op():
    pass


def size_op():
    pass


def slice_op():
    pass


def softmax_op(operator, block):
    inputs, attrs, outputs = op_io_info(operator)
    return make_node('Softmax', inputs=inputs['X'], outputs=outputs['Out'])


def spacetodepth_op():
    pass


def split_op(operator, block):
    inputs, attrs, outputs = op_io_info(operator)
    len_sec = len(attrs['sections'])
    if len_sec > 0:
        return make_node('Split', inputs=inputs['X'],
                outputs=outputs['Out'],
                axis=attrs['axis'],
                split=attrs['sections'])
    else:
        return make_node('Split', inputs=inputs['X'],
                outputs=outputs['Out'],
                axis=attrs['axis'])
       

def squeeze_op():
    pass


def sub_op():
    pass


def sum_op():
    pass


def tile_op():
    pass


def topk_op():
    pass


def transpose_op(operator, block):
    inputs, attrs, outputs = op_io_info(operator)
    node = make_node('Transpose', inputs=inputs['X'],
           outputs=outputs['Out'],
           perm=attrs['axis'])
    return node
    


def unsqueeze_op():
    pass


def thresholded_relu_op(operator, block):
    inputs, attrs, outputs = op_io_info(operator)
    print(attrs)
    return make_node(
        'ThresholdedRelu',
        inputs=inputs['X'],
        outputs=outputs['Out'],
        alpha=attrs['threshold'])

def scale_op(operator, block):
    inputs, attrs, outputs = op_io_info(operator)
    scale_var_name = [outputs['Out'][0] + "@scale"]
    node_scale = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=scale_var_name,
        value=onnx.helper.make_tensor(
             name=scale_var_name[0]+"@const",
             data_type=onnx.TensorProto.FLOAT,
             dims=(),
             vals=[attrs['scale']]))
    bais_var_name = [outputs['Out'][0] + "@bais"]
    bais = 0.0
    if 'bais' in attrs:
       bais = float(attrs['bais'])
    node_bais = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=bais_var_name,
        value=onnx.helper.make_tensor(
             name=bais_var_name[0]+"@const",
             data_type=onnx.TensorProto.FLOAT,
             dims=(),
             vals=[bais]))
    paddle_var = block.var(inputs["X"][0])
    tmp_var_name = outputs['Out'][0] + "@tmp"
    shape =  paddle_onnx_shape(paddle_var.shape)
    tmp_var = onnx.helper.make_tensor_value_info(tmp_var_name, PADDLE_TO_ONNX_DTYPE[paddle_var.dtype], shape)
    nodes = (node_scale, node_bais)
    if attrs['bias_after_scale'] == True:
        node_output_mul = onnx.helper.make_node(
            'Mul',
            inputs=[scale_var_name[0], inputs["X"][0]],
                    outputs=[tmp_var_name])
        node_output_scale = onnx.helper.make_node(
             'Add',
             inputs=[bais_var_name[0], tmp_var_name],
             outputs=outputs["Out"])
        nodes += (node_output_mul, node_output_scale)
    else:
        node_output_add = onnx.helper.make_node(
             'Add',
             inputs=[bais_var_name[0], inputs["X"][0]],
             outputs=[tmp_var_name])
        node_output_mul = onnx.helper.make_node(
            'Mul',
            inputs=[scale_var_name[0], tmp_var_name],
                    outputs=outputs["Out"])
        nodes += (node_output_add, node_output_mul)

    return nodes
       
       
def swish_op(operator, block):
    """
    The activation swish, x / (1 + exp(-beta * x))
    """
    inputs, attrs, outputs = op_io_info(operator) 
    paddle_var = block.var(inputs["X"][0])
    shape =  paddle_onnx_shape(paddle_var.shape)

    if 'beta' in attrs:
        beta = attrs['beta']
    if 'slope' in attrs:
        beta = attrs['slope']

    name_beta = [outputs['Out'][0] + "@swish_beta"]
    node_beta = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=name_beta,
        value=onnx.helper.make_tensor(
             name=name_beta[0]+"@const",
             data_type=onnx.TensorProto.FLOAT,
             dims=(),
             vals=[beta]))

    # var and node for beta * x 
    name_beta_x = [outputs['Out'][0] + "@beta_x"]
    var_beta_x = onnx.helper.make_tensor_value_info(name_beta_x[0], 
        PADDLE_TO_ONNX_DTYPE[paddle_var.dtype], shape)
    node_beta_x = onnx.helper.make_node(
        'Mul',
        inputs=[name_beta[0], inputs['X'][0]],
        outputs=name_beta_x)

    # var and node sigmoid(beta*x)
    name_sigmoid_x = [outputs['Out'][0] + "@sigmoid_x"]
    var_sigmoid_x = onnx.helper.make_tensor_value_info(name_sigmoid_x[0],
        PADDLE_TO_ONNX_DTYPE[paddle_var.dtype], shape)
    node_sigmoid_x = onnx.helper.make_node(
        'Sigmoid',
        inputs=name_beta_x,
        outputs=name_sigmoid_x)

    node_swish = onnx.helper.make_node(
        'Mul',
        inputs=inputs['X'] + name_sigmoid_x,
        outputs=outputs['Out'])
    return (node_beta, node_beta_x, node_sigmoid_x, node_swish)

def relu6_op(operator, block):
    """
    The activation function relu6, out=min(max(0,x),6) 
    And you can set the threshold of activation.
    """
    inputs, attrs, outputs = op_io_info(operator) 
    threshold = attrs['threshold']
    relu6_node = onnx.helper.make_node(
        'Clip',
        inputs=inputs['X'],
        outputs=outputs['Out'],
        max=threshold,
        min=0.0)
    return relu6_node 
    #return (node_min, node_max, node_select_max, node_select_min)
    #return (node_min, node_max, node_select_max)



# Based on the ONNX 1.0 operator list generated on March 26th, 2018.
# Reference for paddle operator availability taken from:
#     https://github.com/PaddlePaddle/Paddle/issues/8028

node_maker = {
    # Paddle op name : (ONNX op name, modifier)
    'abs': partial(activation_ops, 'Abs'),
    # 'ArgMax', NEEDS ATTENTION.
    # 'ArgMin', NEEDS ATTENTION.
    'batch_norm': batch_norm_op,
    'cast': cast_op,
    'ceil': partial(activation_ops, 'Ceil'),
    'clip': clip_op,
    'concat': concat_op,
    'constant': constant_op,
    'conv2d': conv2d_op,
    # Need to continue the mapping below.
    'conv2d_transpose': conv2d_transpose_op,
    # 'cos': partial(activation_ops, 'Cos'),
    '': 'DepthToSpace',
    'depthwise_conv2d': conv2d_op,
    'dropout': dropout_op,
    'elementwise_add': partial(elementwise_ops, 'Add'),
    'elementwise_div': partial(elementwise_ops, 'Div'),
    'elementwise_mul': partial(elementwise_ops, 'Mul'),
    'elementwise_pow': partial(elementwise_ops, 'Pow'),
    'elementwise_sub': partial(elementwise_ops, 'Sub'),
    'elu': elu_op,
    'equal': partial(compare_ops, 'Equal'),
    'exp': partial(activation_ops, 'Exp'),
    '': 'Flatten',
    'floor': partial(activation_ops, 'Floor'),
    '': 'GRU',
    'gather': gather_op,
    '': 'Gemm',
    '': 'GlobalLpPool',
    'greater_than': partial(compare_ops, 'Greater'),
    'hard_sigmoid': 'HardSigmoid',  # Caffe2 error
    # 'Hardmax', NEEDS ATTENTION.
    # 'InstanceNormalization', NEEDS ATTENTION.
    'less_than': partial(compare_ops, 'Less'),
    'lrn': lrn_op,
    '': 'LSTM',
    'leaky_relu': leaky_relu_op,
    'log': partial(activation_ops, 'Log'),
    'logical_and': partial(binary_logical_ops, 'And'),
    'logical_or': partial(binary_logical_ops, 'Or'),
    'logical_not': partial(unary_logical_ops, 'Not'),
    'logical_xor': partial(binary_logical_ops, 'Xor'),
    ',': 'LogSoftmax',
    '': 'LpNormalization',
    '': 'LpPool',
    '': 'MatMul',
    '': 'Max',
    # 'MaxPool', NEEDS ATTENTION.
    '': 'MaxRoiPool',
    'mean': ('Mean', mean_op),
    '': 'Min',
    'mul': mul_op,
    ',': 'Neg',
    'prelu': prelu_op,
    '': 'Pad',
    'pool2d': pool2d_op,
    ',': 'RNN',
    '': 'RandomNormal',
    # 'RandomNormalLike', NEEDS ATTENTION.
    # 'RandomUniform', NEEDS ATTENTION.
    # 'RandomUniformLike', NEEDS ATTENTION.
    'reciprocal': partial(activation_ops, 'Reciprocal'),
    '': 'ReduceL1',
    '': 'ReduceL2',
    ',': 'ReduceLogSum',
    ',': 'ReduceLogSumExp',
    'reduce_max': partial(reduce_ops, 'ReduceMax'),
    'reduce_mean': partial(reduce_ops, 'ReduceMean'),
    'reduce_min': partial(reduce_ops, 'ReduceMin'),
    '': partial(reduce_ops, 'ReduceProd'),  # Caffe2 error
    'reduce_sum': partial(reduce_ops, 'ReduceSum'),
    ',': 'ReduceSumSquare',
    'relu': partial(activation_ops, 'Relu'),
    '': 'Reshape',
    # 'Selu', NEEDS ATTENTION.
    '': 'Shape',
    'sigmoid': partial(activation_ops, 'Sigmoid'),
    # 'sin': partial(activation_ops, 'Sin'),
    '': 'Size',
    # 'Slice', NEEDS ATTENTION.
    'softmax': softmax_op,
    'softplus': partial(activation_ops, 'Softplus'),
    'softsign': partial(activation_ops, 'Softsign'),
    '': 'SpaceToDepth',
    '': 'Split',
    'sqrt': partial(activation_ops, 'Sqrt'),
    # 'Squeeze', NEEDS ATTENTION.
    '': 'Sum',
    'tanh': partial(activation_ops, 'Tanh'),
    '': 'Tile',
    '': 'TopK',
    '': 'Transpose',
    # 'Unsqueeze', NEEDS ATTENTION.
    # 'experimental ATen'
    # ',': 'experimental Affine'
    # 'experimental ConstantFill'
    # 'experimental Crop'
    # 'experimental FC'
    # 'experimental GRUUnit'
    # 'experimental GivenTensorFill'
    # 'assign': 'experimental Identity'
    # 'experimental If'
    # ',': 'experimental ImageScaler'
    # 'experimental Loop'
    # 'experimental LoopIndexTensor'
    # 'experimental MeanVarianceNormalization'
    # 'experimental ParametricSoftplus'
    # 'experimental Scale'
    # 'experimental ScaledTanh'
    'thresholded_relu': thresholded_relu_op,
    'scale': scale_op,
    'split': split_op,
    'reshape2': reshape_op,
    'transpose2': transpose_op,
    'swish': swish_op,
    'relu6': relu6_op 
    # 'experimental Upsample'
}
