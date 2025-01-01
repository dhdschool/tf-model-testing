import tensorflow as tf
import keras
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

from Cleaning import frog_dataset

def generate_simpleCNN():
    func_kwargs = {'loss':'mse', 'optimizer':'adam', 'metrics':['accuracy']}
    
    inp = keras.layers.Input(shape=(4032, 3024, 3), batch_size=2, dtype=tf.float32)
    
    conv1 = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(5, 5))
    pool1 = keras.layers.MaxPool2D(pool_size=(3, 3))
    actv1 = keras.layers.ReLU() 
    
    conv2 = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(3, 3))
    pool2 = keras.layers.MaxPool2D(pool_size = (3, 3))
    actv2 = keras.layers.ReLU()
    
    conv3 = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1))
    pool3 = keras.layers.MaxPool2D(pool_size = (3, 3))
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
    
    dataset = frog_dataset
    
    batch_size = 2
    train_epochs = 16
    test_epochs = 2
    
    dataset.batch(batch_size, drop_remainder=True)
    data_iter = iter(dataset)
    
    for epoch in range(train_epochs):
        X_train, y_train = next(data_iter)
        model.fit(X_train, y_train, batch_size=batch_size)    

    predictions = []
    errors = []
    true_vals = []
    for epoch in range(test_epochs):
        X_test, y_test = next(data_iter)

        pred = model(X_test)

        errors.append(model.evaluate(X_test, y_test, batch_size=1)[0])
        true_vals.append(y_test.numpy()[0])
        predictions.append(pred.numpy()[0, 0])
    
    print(errors)
    formatted = pd.DataFrame({'Mean Squared Error':errors, 'Prediction':predictions, 'Region Count':true_vals}, columns=['Mean Squared Error', 'Prediction', 'Region Count'])
    formatted['Count Error'] = formatted['Prediction'] - formatted['Region Count']
    
    print(formatted)
    print(f"MSE: {formatted['Mean Squared Error'].mean()}")
    print(f"ME: {formatted['Count Error'].mean()}")