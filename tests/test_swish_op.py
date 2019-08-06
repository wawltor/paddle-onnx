#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import unittest
import numpy as np
from scipy.special import expit
from op_test import OpTest

class TestSwish(OpTest):
    def setUp(self):
        self.op_type = "swish"
        self.init_dtype()
        X = np.random.uniform(0.1, 1, [11, 17]).astype(self.dtype)
        beta = 2.3
        out = X * expit(beta * X)
        self.inputs = {'X': X}
        self.attrs = {'beta': beta}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()

    def init_dtype(self):
        self.dtype = np.float32

if __name__ == '__main__':
    unittest.main()
