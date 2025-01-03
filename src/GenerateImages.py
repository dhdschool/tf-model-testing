from Cleaning import IMG_SIZE, REGION_COUNT_PATH
from PIL import Image, ImageDraw
from pathlib import Path
import math

import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.ndimage import histogram

TEST_IMAGE_PATH = 'test/test-img.jpg'
TEXTURE_DIRECTORY = 'texture'
PETRI_DIRECTORY = 'petri-dish'
FROG_EGG_DIRECTORY = 'frog-egg'
FERT_DIRECTORY = 'fert'
UNFERT_DIRECTORY = 'unfert'
TEXTURE_SIZE = (32, 32)

class TextureManager:
    def __init__(self):
        self._head = Path(TEXTURE_DIRECTORY)
        self._petri_path = self._head / Path(PETRI_DIRECTORY)
        self._frog_path = self._head / Path(FROG_EGG_DIRECTORY)
        self._fert_path = self._frog_path / Path(FERT_DIRECTORY)
        self._unfert_path = self._frog_path / Path(UNFERT_DIRECTORY)
        
        self._texture_dict = {}
        
        
        self._dir_to_str = {
            self._petri_path:   'petri-dish',
            self._fert_path:    'fert',
            self._unfert_path:  'unfert'
        }
        
        self._str_to_dir = {k:v for k, v in zip(self._directories.values(), self._directories.keys())}
        
        self._sizes = {
            'petri-dish':   TEXTURE_SIZE,
            'fert':         TEXTURE_SIZE,
            'unfert':       TEXTURE_SIZE
        }
        
        for directory in self._directories:
            texture_path_strings = os.listdir(str(directory))
            texture_paths = list(map(lambda x: directory / Path(x), texture_path_strings))
            
            self._texture_dict[directory] = texture_paths
    
    # Keys are pathlib Path objects
    def __getitem__(self, key, local_path):
        if isinstance(key, Path):
            file_path = key / local_path
            texture_str = self._dir_to_str[file_path]
            
        elif isinstance(key, str):
            texture_str = key
            file_path = self._str_to_dir[texture_str]
        
        else:
            raise TypeError(f'Expected pathlib.Path or str, got {key.__class__}')
        
        with Image.open(file_path) as img:
            self._sizes[texture_str] = img.size
            return (img, texture_str, file_path, self._sizes[texture_str])

   
class Texture:
    texture_obj = TextureManager()
    
    def __init__(self, key, img_path):
        if not isinstance(img_path, Path):
            img_path = Path(img_path)
        
        img, texture_string, file_path, img_size = Texture.texture_obj[key, img_path]
        
        self._texture_img = img
        self._texture_string = texture_string
        self._path = file_path
        self._size = img_size

    @property 
    def img(self):
        return self._texture_img
    
    @property
    def texture_string(self):
        return self._texture_string
    
    @property
    def path(self):
        return self._path
    
    @property
    def size(self):
        return self._size
    

def generate_image(file_path=TEST_IMAGE_PATH):
    img = Image.new(mode='RGB', size=IMG_SIZE)
    canvas = ImageDraw.Draw(img)
    
    white = (255, 255, 255)
    black = (0, 0, 0)
    center = tuple(map(lambda x: x//2, IMG_SIZE))
    max_radius = sum(center)
    
    canvas.ink = white
    brown_to_grey_grad(canvas, center, max_radius)
    petri_dish(canvas, IMG_SIZE)
    
    img.save(file_path)

# Takes in an ImageDraw object 
def texture_shape(draw_func, *, xy, texture : Texture, box_size, func_kwargs={}):
  
    canvas = draw_func.__self__
    
    draw_func_name = draw_func.__name__
    draw_func = getattr(canvas.__class__, draw_func_name)
    
    
    background_img = Image.new('RGB', size=box_size)
    bitmask_img = Image.new('1', size=box_size)
    
    bitmask_canvas = ImageDraw.Draw(bitmask_img)
    
    grid_width = math.ceil(box_size[0] / texture.size[0])
    grid_height = math.ceil(box_size[1] / texture.size[1])
    
    for row in range(grid_height):
        for column in range(grid_width):
            upper_left = (row*texture.size[0], column*texture.size[1])
            lower_right = ((row+1)*texture.size[0], (column+1)*texture.size[1])
            
            background_img.paste(texture.img, box=(upper_left, lower_right))
    
    draw_func(bitmask_canvas, fill=1, outline=1 **func_kwargs)
    canvas.im.paste(background_img, box=xy, mask=bitmask_img)
    
    

def brown_to_grey_grad(canvas, center, max_r):
    max_r *= 1
    
    circles = np.linspace(0, 1, 200)
    
    offset_start = 15
    offset_end = 15
    
    r_start_min = 0
    b_start_min = 0
    g_start_min = 0
    
    r_start_max = r_start_min + offset_start
    b_start_max = b_start_min + offset_start
    g_start_max = g_start_min + offset_start
    
    r_start = np.random.choice(np.arange(r_start_min, r_start_max))
    b_start = np.random.choice(np.arange(b_start_min, b_start_max))
    g_start = np.random.choice(np.arange(g_start_min, g_start_max))
    
    r_end_max = 65
    b_end_max = 63
    g_end_max = 60
    
    r_end_min = r_end_max - offset_end
    b_end_min = b_end_max - offset_end
    g_end_min = g_end_max - offset_end
    
    r_end = np.random.choice(np.arange(r_end_min, r_end_max))
    b_end = np.random.choice(np.arange(b_end_min, b_end_max))
    g_end = np.random.choice(np.arange(g_end_min, g_end_max))
    
    for circle in circles:
        r = int((r_end) * (1 - circle) + r_start * (circle))
        g = int((g_end) * (1 - circle) + g_start * (circle))
        b = int((b_end) * (1 - circle) + b_start * (circle))
        canvas.circle(center, radius=max_r * (1 - circle), fill=(r,g,b), outline=(r,g,b))
    
def petri_dish(canvas, img_size):
    r = int(5 * img_size[0] / 11)
    
    color_start = np.array([61, 67, 77])
    color_end = np.array([105, 114, 130])
    
    proportion = np.random.choice(np.linspace(0, 1, 100))
    
    color = (color_end) * (1 - proportion) + (color_start) * (proportion)
    color = tuple(map(lambda x: int(x), color))
    
    
    lower_bounds_x = r
    upper_bounds_x = img_size[0] - r
    
    lower_bounds_y = r
    upper_bounds_y = img_size[1] - r
    
    x = np.random.choice(np.arange(lower_bounds_x, upper_bounds_x))
    y = np.random.choice(np.arange(lower_bounds_y, upper_bounds_y))
    
    canvas.circle((x, y), radius=r, fill=color)
    
    
def count_distribution(file_path=REGION_COUNT_PATH, show=False):
    df = pd.read_csv(REGION_COUNT_PATH, index_col=0)
    
    bin_count = 25
    hist_min = min(df['region-count'])
    hist_max = max(df['region-count'])
    
    bins = histogram(df['region-count'], 
              min=hist_min,
              max=hist_max,
              bins=bin_count)
    
    bin_edges = pd.Series(np.linspace(hist_min, hist_max, bin_count+1))
    bin_x = ((bin_edges + bin_edges.shift(1))/2)
    bin_x = bin_x.replace(0, pd.NA)
    bin_x = np.array(bin_x.dropna())
    
    polynomial = CubicSpline(bin_x, bins, bc_type='natural')  
    
    if show:  
        x = np.linspace(hist_min, hist_max, 200)
        y = polynomial(x)
        
        
        plt.hist(df['region-count'], bins=bin_count)
        plt.plot(x, y)
        plt.show()
        
    return polynomial


if __name__ == '__main__':
    count_distribution()
    
