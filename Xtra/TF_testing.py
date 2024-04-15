''' verify installation of TensorFlow '''

import tensorflow as tf

# Verify the CPU setup:
print(tf.reduce_sum(tf.random.normal([1000, 1000])))
# If a tensor is returned, you've installed TensorFlow successfully.

# Verify the GPU setup:
print(tf.config.list_physical_devices('GPU'))   
# If a list of GPU devices is returned, you've installed TensorFlow successfully.

# nvidia-smi
