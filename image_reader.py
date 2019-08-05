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
import numpy as np
from PIL import Image

DATA_DIM = 224
img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

class ImageBaseReader():
    def __init__(self, input_data_dict):
        self.data_dict = input_data_dict 

    def _resize_short(self, img, target_size):
        percent = float(target_size) / min(img.size[0], img.size[1])
        resized_width = int(round(img.size[0] * percent))
        resized_height = int(round(img.size[1] * percent))
        img = img.resize((resized_width, resized_height), Image.LANCZOS)
        return img


    def _crop_image(self, img, target_size, center):
        width, height = img.size
        size = target_size
        if center == True:
            w_start = (width - size) / 2
            h_start = (height - size) / 2
        else:
            w_start = np.random.randint(0, width - size + 1)
            h_start = np.random.randint(0, height - size + 1)
        w_end = w_start + size
        h_end = h_start + size
        img = img.crop((w_start, h_start, w_end, h_end))
        return img

    def _process_image(self, img):
        img = self._resize_short(img, target_size=256)
        img = self._crop_image(img, target_size=DATA_DIM, center=True)

        if img.mode != 'RGB':
            img = img.convert('RGB')

        img = np.array(img).astype('float32').transpose((2, 0, 1)) / 255
        img -= img_mean
        img /= img_std
        return img

    def preprocess(self):
        result = {'image': []}
        path = self.data_dict['image']
        walk_list = os.walk(path)
        # walk the overall dir, get all files
        for path, list_dir, file_list in walk_list:
            for file_name in file_list:
                file_path = os.path.join(path, file_name)
                yield self._process_image(Image.open(file_path))
        
  
