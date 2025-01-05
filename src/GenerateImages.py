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
TEXTURE_DIRECTORY = 'textures'
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
        
        self._str_to_dir = {k:v for k, v in zip(self._dir_to_str.values(), self._dir_to_str.keys())}
        
        self._sizes = {
            'petri-dish':   TEXTURE_SIZE,
            'fert':         TEXTURE_SIZE,
            'unfert':       TEXTURE_SIZE
        }
        
        for directory in self._dir_to_str:
            texture_path_strings = os.listdir(str(directory))
            texture_paths = list(map(lambda x: directory / Path(x), texture_path_strings))
            
            self._texture_dict[directory] = texture_paths
        
        self.images = []
    
    # Keys are pathlib Path objects
    def __getitem__(self, key):
        assert len(key) == 2
        
        key, local_path = key
        if isinstance(key, Path):
            file_path = key / local_path
            texture_str = self._dir_to_str[file_path]
            
        elif isinstance(key, str):
            texture_str = key
            file_path = self._str_to_dir[texture_str] / local_path
        
        else:
            raise TypeError(f'Expected pathlib.Path or str, got {key.__class__}')
        
        
        img = Image.open(fp=str(file_path))
        self.images.append(img)
        
        self._sizes[texture_str] = img.size
        return img, texture_str, file_path, self._sizes[texture_str]

    def __del__(self):
        for img in self.images:
            img.close()
   
   
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
    def img_and_mask(self):
        
        angles = np.linspace(0, 360, 15)
        angle_choice = np.random.choice(angles)
        
        self._texture_copy = self._texture_img.rotate(angle_choice)
        mask = Image.new('1', self._size, color=1).rotate(angle_choice)
        
        return self._texture_copy, mask
        
    
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
    
    center = tuple(map(lambda x: x//2, IMG_SIZE))
    
    canvas.ink = white
    brown_to_grey_grad(canvas, center)
    petri_dish(canvas, 'dark-1.jpg', IMG_SIZE)
    
    img.save(file_path)

# Takes in an ImageDraw object 
def texture_shape(draw_func, *, xy : tuple[tuple[int, int], tuple[int, int]], texture : Texture, func_kwargs={}, border_adjuster=4):
    # Border adjuster changes how granular the texture is
    # For a 32x32 texture, border adjuster ranges from 0-8
    # Time complexity is O(2^n) w.r.t. border adjuster, i.e.
    # a=6 takes 2x as long as a=5, a=7 takes 4x, and a=8 takes 16x

    border_offset = math.ceil(max(texture.size[0], texture.size[1]) / math.sqrt(2)) + border_adjuster
    texture_size = (texture.size[0] - border_offset, texture.size[1]-border_offset)
    
    canvas = draw_func.__self__
    
    draw_func_name = draw_func.__name__
    draw_func = getattr(canvas.__class__, draw_func_name)
    
    box_size = (2*func_kwargs['radius'], 2*func_kwargs['radius'])
    
    background_img = Image.new('RGB', size=box_size, color=(0, 0, 0))
    bitmask_img = Image.new('1', size=box_size, color=0)
    
    bitmask_canvas = ImageDraw.Draw(bitmask_img)
    
    grid_width = math.ceil(box_size[0] / texture_size[0])
    grid_height = math.ceil(box_size[1] / texture_size[1])
    
    for row in range(grid_height):
        for column in range(grid_width):
            upper_left = (row*texture_size[0], column*texture_size[1])
            
            paste_img, paste_mask = texture.img_and_mask            
            background_img.paste(paste_img, box=upper_left, mask=paste_mask)
    
    draw_func(bitmask_canvas, fill=1, outline=1, **func_kwargs)
    canvas._image.paste(background_img, box=xy[0], mask=bitmask_img)
    
    

def brown_to_grey_grad(canvas, center, scaling_factor=1):
    max_r = sum(center) * scaling_factor
    
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

def hex_coordinates(circle_radius, hex_radius):
    max_hex_radii = circle_radius // hex_radius
    max_r = (max_hex_radii - 1) // 2 + 1
    
    

def petri_dish(canvas, texture_file, img_size):
    r = int(5 * img_size[0] / 11)
    
    lower_bounds_x = r
    upper_bounds_x = img_size[0] - r
    
    lower_bounds_y = r
    upper_bounds_y = img_size[1] - r
    
    x = np.random.choice(np.arange(lower_bounds_x, upper_bounds_x))
    y = np.random.choice(np.arange(lower_bounds_y, upper_bounds_y))
    
    petri_dish_bb = ((x-r, y-r), (x+r, y+r))
    petri_dish_texture = Texture('petri-dish', texture_file)
    petri_circle_args = {
        'xy': (r, r),
        'radius': r
    }
    
    texture_shape(canvas.circle, 
                  func_kwargs=petri_circle_args,
                  xy=petri_dish_bb,
                  texture=petri_dish_texture,
                  border_adjuster=5)

    
    
def count_distribution(file_path=REGION_COUNT_PATH, show=False, weight=False):
    df = pd.read_csv(REGION_COUNT_PATH, index_col=0)
    
    bin_count = 25
    hist_min = min(df['region-count'])
    hist_max = max(df['region-count'])
    
    bins = histogram(df['region-count'], 
              min=hist_min,
              max=hist_max,
              bins=bin_count)
    
    if weight:
        weights = np.arange(1, len(bins) + 1)
        bins = bins * weights
    
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
    count_distribution(show=True, weight=True)    
