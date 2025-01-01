import os

from tensorflow.python.client import device_lib

def initialization():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

import tensorflow as tf
    
if __name__ == '__main__':
    initialization()