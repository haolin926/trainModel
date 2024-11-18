import tensorflow as tf
from tensorflow.python.client import device_lib
import os
def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(get_available_devices())

# List available physical devices (should show the GPU if detected)
print("Available devices:", tf.config.list_physical_devices())

print("Is GPU available:", tf.config.list_physical_devices('GPU'))