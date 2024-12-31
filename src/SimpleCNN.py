import tensorflow as tf
import keras
import numpy as np
from sklearn.model_selection import train_test_split

from Cleaning import Cleaning

def generate_simpleCNN():
    func_kwargs = {'loss':'mse', 'optimizer':'adam', 'metrics':['accuracy']}
    
    inp = keras.layers.Input(shape=(4032, 3024, 3), batch_size=2, dtype=tf.float32)
    
    conv1 = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(5, 5))
    pool1 = keras.layers.MaxPool2D(pool_size=(5, 5))
    actv1 = keras.layers.ReLU() 
    
    conv2 = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1))
    pool2 = keras.layers.MaxPool2D(pool_size = (2, 2))
    actv2 = keras.layers.ReLU()
    
    conv3 = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1))
    pool3 = keras.layers.MaxPool2D(pool_size = (2, 2))
    actv3 = keras.layers.ReLU()
    reshape3 = keras.layers.Reshape((-1,))
    
    regr4 = keras.layers.Dense(units=10)
    actv4 = keras.layers.ReLU()
    
    regr5 = keras.layers.Dense(units=1)
    actv5 = keras.layers.ReLU()
    
    model = keras.Sequential(layers=[
        inp,
        
        conv1,
        pool1,
        actv1,
        
        conv2,
        pool2,
        actv2,
        
        conv3,
        pool3,
        actv3,
        reshape3,
        
        regr4,
        actv4,
        regr5,
        actv5
    ])
    
    model.compile(**func_kwargs)
    return model
    

if __name__ == '__main__':
    model = generate_simpleCNN()
    model.summary()
    
    data = Cleaning()
    dataset = data.dataset
    
    X = []
    y = []
    
    sample_dataset = dataset.take(16) 
    iterable_dataset = iter(sample_dataset)
    
    for features, labels in iterable_dataset:
        X.append(features)
        y.append(labels)

    X = np.array(X)
    y = np.array(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)
    
    X_train = tf.convert_to_tensor(X_train)
    X_test = tf.convert_to_tensor(X_test)
    y_train = tf.convert_to_tensor(y_train)
    y_test = tf.convert_to_tensor(y_test)
        
    model.fit(X_train, y_train, batch_size=2)    
    
