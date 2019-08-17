import onnx
import onnxruntime
import numpy as np
from onnxruntime.backend import prepare
from onnx import shape_inference
import pickle
import time 

sess = onnxruntime.InferenceSession("tests/nms_test.onnx") 
with open("tests/inputs_test.pkl", "rb") as f:
   np_images = pickle.load(f)
f.close()
result = sess.run([], np_images)
value_len = 0
data_dict = {}
for i in range(0, 10):
    data_dict[i] = 0

res = result[-1]

with open("tests/outputs_test.pkl", "wb") as f:
    pickle.dump(res, f)
f.close()

