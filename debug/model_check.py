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

import sys 
import os

TEST="./tests/test_"
PY="_op.py"
ELEMENTWISE="elementwise"

def debug_model(op_list, var_list):
    # start check the op test 
    for op_name in op_list:
        print("start check the op: %s"%(op_name))
        op_test_name = TEST + op_name + PY
        run_script = "python " + op_test_name 
        os.system(run_script)
      
    
