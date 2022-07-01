# from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

# print_tensors_in_checkpoint_file(file_name= r"/media/vee/Docs/thaivd4/Projects/Turnstiles_Fare_Evasion/models/deep_sort/mars-small128.ckpt-68577"
#                        , all_tensor_names= True , all_tensors= True, tensor_name= "")

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import os

model_dir = r"/media/vee/Docs/thaivd4/Projects/Turnstiles_Fare_Evasion/models/deep_sort" 
checkpoint_path = os.path.join(model_dir, r"mars-small128.ckpt-68577")
reader = tf.compat.v1.train.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()

for key in var_to_shape_map:
    print("tensor_name: ", key)
    print(reader.get_tensor(key)) # Remove this is you want to print only variable names