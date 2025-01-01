import os

from tensorflow.python.client import device_lib

def initialization():
    devices = device_lib.list_local_devices()
    return [x.name for x in devices if x.device_type == 'GPU']

import tensorflow as tf

if __name__ == '__main__':
    initialization()