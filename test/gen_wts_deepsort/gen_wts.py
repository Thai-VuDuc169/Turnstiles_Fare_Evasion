#### Refer: https://github.com/wang-xinyu/tensorrtx/blob/master/psenet/gen_tf_wts.py
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
from sys import prefix
import numpy as np
import struct

model_dir = r"/media/vee/Docs/thaivd4/Projects/Turnstiles_Fare_Evasion/models/deep_sort"
checkpoint_path = os.path.join(model_dir, r"mars-small128.ckpt-68577")

print(f'[INFO]: Loading pretraind tf model...')
reader = tf.train.NewCheckpointReader(checkpoint_path)
param_dict = reader.get_variable_to_shape_map()
print(f'[INFO]: Loaded successfully!!!')

with open(r"mars-small128.wts", "w") as f:
    keys = param_dict.keys()
    f.write("{}\n".format(len(keys)))

    for key in keys:
        weight = reader.get_tensor(key)
        print(key, weight.shape)
        if len(weight.shape) == 4:
            weight = np.transpose(weight, (3, 2, 0, 1)) 
            print(weight.shape)
        weight = np.reshape(weight, -1)
        f.write("{} {} ".format(key, len(weight)))
        for w in weight:
            f.write(" ")
            f.write(struct.pack(">f", float(w)).hex())
        f.write("\n")
