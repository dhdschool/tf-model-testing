import tensorflow as tf
import keras
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from Cleaning import frog_dataset, IMG_SHAPE

def generate_simpleCNN():
    func_kwargs = {'loss':'mse', 'optimizer':'adam', 'metrics':['accuracy']}
    
    inp = keras.layers.Input(shape=IMG_SHAPE, batch_size=2, dtype=tf.float32)
    
    conv1 = keras.layers.Conv2D(filters=16, kernel_size=3, strides=3)
    pool1 = keras.layers.MaxPool2D(pool_size=3)
    actv1 = keras.layers.ReLU() 
    
    conv2 = keras.layers.Conv2D(filters=128, kernel_size=3, strides=2)
    pool2 = keras.layers.MaxPool2D(pool_size=2)
    actv2 = keras.layers.ReLU()
    
    conv3 = keras.layers.Conv2D(filters=256, kernel_size=2, strides=1)
    pool3 = keras.layers.MaxPool2D(pool_size=2)
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
    
    batch_size = 1
    train_batches = 55
    test_batches = 8
    
    dataset = dataset.batch(batch_size, drop_remainder=True)
    data_iter = iter(dataset)
    
    model.fit(dataset, epochs=train_batches, steps_per_epoch=1)    

    predictions = []
    errors = []
    true_vals = []
    for epoch in range(test_batches):
        X_test, y_test = next(data_iter)
                
        for batch in range(batch_size):
            errors.append(
                model.evaluate(
                    tf.expand_dims(X_test[batch, :, :, :], axis=0),
                    tf.expand_dims(y_test[batch], axis=0), 
                    batch_size=1)[0]
            )
            
        pred = model(X_test)
        true_vals.extend(y_test.numpy().flatten())
        predictions.extend(pred.numpy().flatten())
    
    print(errors)
    formatted = pd.DataFrame({'Mean Squared Error':errors, 'Prediction':predictions, 'Region Count':true_vals}, columns=['Mean Squared Error', 'Prediction', 'Region Count'])
    formatted['Count Error'] = formatted['Prediction'] - formatted['Region Count']
    formatted['Absolute Error'] = abs(formatted['Prediction'] - formatted['Region Count'])
    
    print(formatted)
    print()
    print(f"MSE: {formatted['Mean Squared Error'].mean()}")
    print(f"MAE: {formatted['Absolute Error'].mean()}")
    print(f"ME: {formatted['Count Error'].mean()}")
