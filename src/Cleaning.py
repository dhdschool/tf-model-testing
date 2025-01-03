import pandas as pd
from pathlib import Path
import os

import tensorflow as tf
import numpy as np
import cv2

from random import sample

from PIL import Image


gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]  # Limit to 4GB
    )
  except RuntimeError as e:
    print(e)

IMG_SIZE = (384, 512)
IMG_SHAPE = (512, 384, 3)

DATASET_PATH = 'annotated-images/image-data'
IMAGE_PATH = 'annotated-images/preprocessed-images'
PROCESSED_PATH = 'annotated-images/processed-images'
CSV_PATH = 'annotated-images/csv-files'
REGION_COUNT_PATH = 'annotated-images/region-count.csv'

class Cleaning:
    def __init__(self, 
                 image_path=IMAGE_PATH, 
                 csv_path=CSV_PATH, 
                 region_path = REGION_COUNT_PATH,
                 dataset_path = DATASET_PATH,
                 processed_path = PROCESSED_PATH,
                 img_size = IMG_SIZE,
                 img_shape = IMG_SHAPE):  
        
        self.processed_path_strings = os.listdir(processed_path)
        self.processed_path_strings.remove('.gitkeep')
        self.processed_paths = list(map(lambda x: Path(processed_path) / Path(x), self.processed_path_strings))
        
        self.img_path_strings = os.listdir(image_path)
        self.img_path_strings.remove('.gitkeep')
        self.img_paths = list(map(lambda x: Path(image_path) / Path(x), self.img_path_strings))
        
        self.csv_path_strings = os.listdir(csv_path)
        self.csv_path_strings.remove('.gitkeep')
        self.csv_paths = list(map(lambda x: Path(csv_path) / Path(x), self.csv_path_strings))

        self.csv_directory = csv_path
        self.image_directory = image_path
        self.processed_directory = processed_path
        self.output_path = region_path
        self.dataset_path = dataset_path
        
        self.img_shape = img_shape
        self.img_size = img_size
        
        self.output_signature = (
            tf.TensorSpec(shape=self.img_shape, dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32)
        )
    
    # Processes everything in the preprocessed-images folder and moves them to the processed-images folder
    def prepare_images(self):
        try:
            df = pd.read_csv(self.output_path, index_col=0)
        except FileNotFoundError:
            df = pd.DataFrame(data=[[]], columns=['region-count', 'file-name'])
            df.index.name = 'file-id'
        
        for img_path in self.img_paths:
            try:
                df = self.append_dataframe(img_path, df)
            except FileNotFoundError:
                continue
            
            self.prepare_single_image(img_path)
            self.prepare_dataset_entry(img_path, df)
            
            os.remove(img_path)
            
        df.to_csv(self.output_path)
        
    def append_dataframe(self, img_path, dataframe, count=None):
        if not count:
            csv_path = Path(self.csv_directory) / Path(img_path.stem + '.csv')
            try:
                with open(csv_path, 'r') as csv_file:
                    header = csv_file.readline()
                    line1 = csv_file.readline()
                    line_list = line1.split(',')
                    
                    count = int(line_list[3])
                
            except FileNotFoundError:
                print(f'Could not find {csv_path}, ')
                raise FileNotFoundError
            
        if img_path.stem in dataframe.index:
            dataframe.loc[img_path.stem] = [img_path.name, count]
        else:
            df_row = pd.DataFrame([[count, img_path.name]], index=[img_path.stem], columns=['region-count', 'file-name'])
            df_row.index.name = 'file-id'
        
            dataframe = pd.concat([dataframe, df_row])
            
        return dataframe
        
        
    def prepare_dataset_entry(self, img_path, data):
        x = cv2.imread(str(img_path.resolve()))
        y = np.array(data['region-count'].loc[img_path.stem], dtype=np.float32)
        
        x = x.astype(np.float32)
        
        assert x is not None
        assert y is not None
        
        x_path = Path(DATASET_PATH) / Path(img_path.stem + 'x.npy')
        y_path = Path(DATASET_PATH) / Path(img_path.stem + 'y.npy')
        
        np.save(str(x_path), x)
        np.save(str(y_path), y)
        
    def prepare_single_image(self, img_path):
        image_dir = Path(self.processed_directory)
        with Image.open(img_path) as img:
            if img_path.suffix != 'jpg':
                img = img.convert('RGB')
                new_file_path = image_dir / Path(img_path.stem + '.jpg')
            
            else:
                new_file_path = image_dir / Path(img_path.name)
                            
            img_new = Image.new(img.mode, img.size)
            img_new.putdata(img.getdata())
            
            img_new = img_new.resize(self.img_size)
            img_new.save(str(new_file_path))

    # This assumes that the images have been processed by prepare_images
    def prepare_counting_processed(self):
        csv_count_dict = {}
        for csv_file in self.csv_paths:
            with csv_file.open() as file:
                file.readline()
                line1 = file.readline()
                
                line_list = line1.split(',')
                csv_count_dict[csv_file.stem] = line_list[3]
        
        img_name_dict = {}
        for img_file in self.processed_paths:
            if img_file.stem in csv_count_dict:
                img_name_dict[img_file.stem] = img_file.name
        
        df = pd.DataFrame.from_dict(csv_count_dict, orient='index', columns=['region-count'])
        img_name_df = pd.DataFrame.from_dict(img_name_dict, orient='index', columns=['file-name'])
        
        df.index.name = 'file-id'
        img_name_df.index.name = 'file-id'
        
        df = pd.concat([df, img_name_df], axis=1)
        df['region-count'] = df['region-count'].astype(int)
        df.to_csv(self.output_path)
    
    # This assumes that a dataset exists following prepare_counting and that the images have already been processed
    def prepare_dataset_processed(self):
        df = pd.read_csv(self.output_path, index_col=0)
        for img_path in self.processed_paths:
            x = cv2.imread(str(img_path.resolve()))      
            
            y = np.array(df['region-count'].loc[img_path.stem], dtype=np.float32)
            x = x.astype(np.float32)                     

            assert x is not None
            assert y is not None
            
            x_path = Path(DATASET_PATH) / Path(img_path.stem + 'x.npy')
            y_path = Path(DATASET_PATH) / Path(img_path.stem + 'y.npy')
            
            np.save(str(x_path), x)
            np.save(str(y_path), y)

    # This assumes that a dataset has been prepared with prepare_dataset_processed or with prepare_images
    def dataset_generator(self):
        shuffled_img_paths = sample(self.processed_paths, len(self.processed_paths))
        x_dct = {}
        y_dct = {}
            
        for img_path in shuffled_img_paths:
            if len(x_dct) > 100:
                x_dct = {}
                y_dct = {}
            
           
            if img_path.stem not in x_dct:
                x_path = Path(DATASET_PATH) / Path(img_path.stem + 'x.npy')
                x = np.load(x_path)
                x = tf.convert_to_tensor(x, dtype=tf.float32)
                x = tf.map_fn(lambda x: x/255.0, x)
                x_dct[img_path.stem] = x
            else:
                x = x_dct[img_path.stem]

            if img_path.stem not in y_dct:
                y_path = Path(DATASET_PATH) / Path(img_path.stem + 'y.npy')
                y = np.load(y_path)
                y = tf.convert_to_tensor(y, dtype=tf.float32)
                y_dct[img_path.stem] = y
            else:
                y = y_dct[img_path.stem]
            
            
            yield x, y    

    
if __name__ == '__main__':
    obj = Cleaning()
    obj.prepare_dataset_processed() 
    dataset_test = tf.data.Dataset.from_generator(obj.dataset_generator, output_signature=obj.output_signature)
          
obj = Cleaning()
frog_dataset = tf.data.Dataset.from_generator(obj.dataset_generator, output_signature=obj.output_signature)

    
