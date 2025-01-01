from Cleaning import IMG_SIZE, REGION_COUNT_PATH
from PIL import Image, ImageDraw

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.ndimage import histogram

TEST_IMAGE_PATH = 'test/test-img.jpg'

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
    
