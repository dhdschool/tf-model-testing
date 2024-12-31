import pandas as pd
from pathlib import Path
import os

import tensorflow as tf
import numpy as np
import cv2


gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]  # Limit to 4GB
    )
  except RuntimeError as e:
    print(e)

class Cleaning:
    def __init__(self):        
        self.img_path_strings = os.listdir('annotated-images/alloted-images')
        self.img_paths = list(map(lambda x: Path(f'annotated-images/alloted-images/{x}'), self.img_path_strings))
        
        self.csv_path_strings = os.listdir('annotated-images/csv-files')
        self.csv_paths = list(map(lambda x: Path(f'annotated-images/csv-files/{x}'), self.csv_path_strings))
      
        self.prepare_counting()
        self.prepare_images()   
        
    def prepare_counting(self):
        csv_count_dict = {}
        for csv_file in self.csv_paths:
            with csv_file.open() as file:
                file.readline()
                line1 = file.readline()
                
                line_list = line1.split(',')
                csv_count_dict[csv_file.stem] = line_list[3]
        
        img_name_dict = {}
        for img_file in self.img_paths:
            if img_file.stem in csv_count_dict:
                img_name_dict[img_file.stem] = img_file.name
        
        df = pd.DataFrame.from_dict(csv_count_dict, orient='index', columns=['region-count'])
        img_name_df = pd.DataFrame.from_dict(img_name_dict, orient='index', columns=['file-name'])
        
        df.index.name = 'file-id'
        img_name_df.index.name = 'file-id'
        
        df = pd.concat([df, img_name_df], axis=1)
        df['region-count'] = df['region-count'].astype(int)
        
        self.df = df
    
    def gen_image(self):
        for img_path in self.img_paths:
            x = cv2.imread(str(img_path.resolve()))
            y = self.df['region-count'].loc[img_path.stem]
            
            yield x, y
    
    def prepare_images(self):
        dataset = tf.data.Dataset.from_generator(self.gen_image, 
                                                 output_signature=(
                                                     tf.TensorSpec(shape=(4032, 3024, 3), dtype=tf.float32),
                                                     tf.TensorSpec(shape=(), dtype=tf.float32)
                                                )
                                            )                          
        self._dataset = dataset

    @property
    def dataset(self):
        return self._dataset
    
    @property
    def dataframe(self):
        return self.df

if __name__ == '__main__':
    obj = Cleaning()
    dataset = obj.dataset
    print(next(iter(dataset.take(1))))