import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

import tensorflow as tf

def initialization():
    gpus = tf.config.list_physical_devices('GPU')
    assert(len(gpus) > 0)
    
if __name__ == '__main__':
    initialization()